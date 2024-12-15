use burn::{module::Module, nn::conv::Conv2d, prelude::*};
use nn::{conv::Conv2dConfig, BatchNorm, BatchNormConfig, PaddingConfig2d};

use super::relu6::ReLU6;

#[derive(Module, Debug)]
pub struct Conv2dNorm<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: ReLU6,
}

impl<B: Backend> Conv2dNorm<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct Conv2dNormConfig {
    channels: [usize; 2],

    kernel_size: [usize; 2],

    stride: [usize; 2],

    padding: PaddingConfig2d,

    #[config(default = "1")]
    groups: usize,

    #[config(default = false)]
    bias: bool,
}

impl Conv2dNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dNorm<B> {
        Conv2dNorm {
            conv: Conv2dConfig::new(self.channels, self.kernel_size)
                .with_stride(self.stride)
                .with_padding(self.padding.clone())
                .with_groups(self.groups)
                .with_bias(self.bias)
                .init(device),
            norm: BatchNormConfig::new(self.channels[1]).init(device),
            activation: ReLU6,
        }
    }
}
