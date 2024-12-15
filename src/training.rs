use std::{fs, path::Path};

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

use crate::{
    data::{CatBatcher, CatDataset},
    model::ModelConfig,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,

    pub optimizer: AdamConfig,

    #[config(default = 10)]
    pub epoch_count: usize,

    #[config(default = 8)]
    pub batch_size: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-4)]
    pub learning_rate: f64,

    #[config(default = 4)]
    worker_count: usize,
}

fn create_artifact_dir(artifact_dir: &Path) {
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &Path, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);

    config.save(artifact_dir.join("config.json")).unwrap();

    B::seed(config.seed);

    let batcher_train = CatBatcher::<B>::new(device.clone());
    let batcher_valid = CatBatcher::<B::InnerBackend>::new(device.clone());

    let (train, test) = CatDataset::load(Path::new("data"));

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.worker_count)
        .build(train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.worker_count)
        .build(test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.epoch_count)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(artifact_dir.join("model"), &CompactRecorder::new())
        .unwrap();
}
