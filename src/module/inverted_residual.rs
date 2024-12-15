use burn::{nn::conv::Conv2d, prelude::*};
use nn::{
    conv::Conv2dConfig, BatchNorm, BatchNormConfig, PaddingConfig2d,
};

use super::conv2d_norm::{Conv2dNorm, Conv2dNormConfig};

#[derive(Module, Debug)]
pub struct PointwiseLinear<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

impl<B: Backend> PointwiseLinear<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        self.norm.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    pub pointwise: Option<Conv2dNorm<B>>,
    pub depthwise: Conv2dNorm<B>,
    pub pointwise_linear: PointwiseLinear<B>,

    pub is_identity: bool,
}

impl<B: Backend> InvertedResidual<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let original = x.clone();

        let x = match &self.pointwise {
            Some(pointwise) => pointwise.forward(x),
            None => x,
        };

        let x = self.depthwise.forward(x);
        let x = self.pointwise_linear.forward(x);

        if self.is_identity {
            original + x
        } else {
            x
        }
    }
}

#[derive(Config, Debug)]
pub struct InvertedResidualConfig {
    pub channels: [usize; 2],
    pub stride: [usize; 2],
    pub expand_ratio: usize,
}

impl InvertedResidualConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        let hidden_dim = self.channels[0] * self.expand_ratio;
        let pointwise = if self.expand_ratio > 1 {
            Some(
                Conv2dNormConfig::new(
                    [self.channels[0], hidden_dim],
                    [1, 1],
                    [1, 1],
                    PaddingConfig2d::Explicit(0, 0),
                )
                .init(device),
            )
        } else {
            None
        };

        InvertedResidual {
            pointwise,

            depthwise: Conv2dNormConfig::new(
                [hidden_dim, hidden_dim],
                [3, 3],
                self.stride,
                PaddingConfig2d::Explicit(1, 1),
            )
            .with_groups(hidden_dim)
            .init(device),

            pointwise_linear: PointwiseLinear {
                conv: Conv2dConfig::new([hidden_dim, self.channels[1]], [1, 1])
                    .with_bias(false)
                    .init(device),
                norm: BatchNormConfig::new(self.channels[1]).init(device),
            },

            is_identity: self.stride == [1, 1] && self.channels[0] == self.channels[1],
        }
    }
}
