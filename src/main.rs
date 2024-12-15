use std::path::Path;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    optim::AdamConfig,
};
use model::ModelConfig;
use training::{train, TrainingConfig};

pub mod data;
pub mod model;
pub mod module;
pub mod training;

const LABELS: [&str; 5] = [
    "bengal",
    "domestic_shorthair",
    "maine_coon",
    "ragdoll",
    "siamese",
];

fn main() {
    type Backend = Wgpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = WgpuDevice::default();
    let artifact_dir = Path::new("artifact");

    train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(LABELS.len(), 0.5), AdamConfig::new()).with_epoch_count(30),
        device,
    );
}
