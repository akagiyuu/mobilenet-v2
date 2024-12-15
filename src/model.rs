
use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use nn::{
    loss::CrossEntropyLossConfig,
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig}, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
};

use crate::{
    data::CatBatch,
    module::{
        conv2d_norm::{Conv2dNorm, Conv2dNormConfig},
        inverted_residual::{InvertedResidual, InvertedResidualConfig},
    },
};

const INVERTED_RESIDUAL_SETTINGS: [[usize; 4]; 7] = [
    // (t = expansion factor; c = channels; n = num blocks; s = stride)
    // t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
];

#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
}

impl<B: Backend> Classifier<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.dropout.forward(x);
        self.linear.forward(x)
    }
}

#[derive(Module, Debug)]
enum Conv2dBlock<B: Backend> {
    Conv(Conv2dNorm<B>),
    InvertedResidual(InvertedResidual<B>),
}

impl<B: Backend> Conv2dBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Conv2dBlock::Conv(conv2d_norm) => conv2d_norm.forward(x),
            Conv2dBlock::InvertedResidual(inverted_residual) => inverted_residual.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d_blocks: Vec<Conv2dBlock<B>>,

    avg_pool: AdaptiveAvgPool2d,
    classifier: Classifier<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self
            .conv2d_blocks
            .iter()
            .fold(x, |x, conv2d_block| conv2d_block.forward(x));

        let x = self.avg_pool.forward(x);
        let x = x.flatten(1, 3);
        self.classifier.forward(x)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<CatBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: CatBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CatBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: CatBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub classes: usize,
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let mut input_channel = 32;
        let output_channel = 1280;

        let mut conv2d_blocks = vec![];

        conv2d_blocks.push(Conv2dBlock::Conv(
            Conv2dNormConfig::new(
                [3, input_channel],
                [3, 3],
                [2, 2],
                PaddingConfig2d::Explicit(1, 1),
            )
            .init(device),
        ));

        for [t, c, n, s] in INVERTED_RESIDUAL_SETTINGS {
            let pre_c = input_channel;
            input_channel = c;

            conv2d_blocks.push(Conv2dBlock::InvertedResidual(
                InvertedResidualConfig::new([pre_c, c], [s, s], t).init(device),
            ));

            for _ in 0..n - 1 {
            conv2d_blocks.push(Conv2dBlock::InvertedResidual(
                InvertedResidualConfig::new([c, c], [1, 1], t).init(device),
            ));
            }
        }

        conv2d_blocks.push(Conv2dBlock::Conv(
            Conv2dNormConfig::new(
                [input_channel, output_channel],
                [1, 1],
                [1, 1],
                PaddingConfig2d::Valid,
            )
            .init(device),
        ));

        let avg_pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let classifier = Classifier {
            linear: LinearConfig::new(output_channel, self.classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        };

        Model {
            conv2d_blocks,
            avg_pool,
            classifier,
        }
    }
}
