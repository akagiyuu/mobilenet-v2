use burn::{
    module::Module,
    prelude::Backend,
    tensor::{activation::relu, Tensor},
};

#[derive(Module, Debug, Clone)]
pub struct ReLU6;

impl ReLU6 {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        relu(input).clamp_max(6)
    }
}
