use std::{fs, path::Path};

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    prelude::*,
};
use image::{imageops::FilterType, ImageReader};
use rand::{seq::SliceRandom, thread_rng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};

use crate::LABELS;

const WIDTH: usize = 224;
const HEIGHT: usize = 224;
const CHANNEL_COUNT: usize = 3;

// const MEAN: [f32; 3] = [0.5830539325104355, 0.5555322207007403, 0.4937775657137797];
//
// const STD: [f32; 3] = [
//     0.24543463541345029,
//     0.23733689397361069,
//     0.26497867766966515,
// ];

#[derive(Debug, Clone)]
pub struct Cat {
    pub image: [[[u8; WIDTH]; HEIGHT]; CHANNEL_COUNT],
    pub label: u32,
}

pub struct CatDataset {
    pub dataset: InMemDataset<Cat>,
}

impl Dataset<Cat> for CatDataset {
    fn get(&self, index: usize) -> Option<Cat> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl CatDataset {
    pub fn load(data_dir: &Path) -> (Self, Self) {
        let mut items: Vec<_> = LABELS
            .par_iter()
            .enumerate()
            .flat_map(|(id, name)| {
                let cat_paths = fs::read_dir(data_dir.join(name))
                    .unwrap()
                    .flatten()
                    .map(|entry| entry.path());
                cat_paths.par_bridge().flat_map(move |path| {
                    let Ok(image_raw) = ImageReader::open(path).unwrap().decode() else {
                        return None;
                    };

                    let image_raw = image_raw.resize_exact(224, 224, FilterType::Triangle);
                    let image_raw = image_raw.to_rgb8();

                    let mut image = [[[0; WIDTH]; HEIGHT]; CHANNEL_COUNT];

                    for (i, pixel) in image_raw.pixels().enumerate() {
                        let h = i / WIDTH;
                        let w = i % WIDTH;
                        let [r, g, b] = pixel.0;
                        image[0][w][h] = r;
                        image[1][w][h] = g;
                        image[2][w][h] = b;
                    }

                    Some(Cat {
                        image,
                        label: id as u32,
                    })
                })
            })
            .collect();

        items.shuffle(&mut thread_rng());

        let (train_items, test_items) = items.split_at(items.len() * 8 / 10);

        (
            CatDataset {
                dataset: InMemDataset::new(train_items.to_vec()),
            },
            CatDataset {
                dataset: InMemDataset::new(test_items.to_vec()),
            },
        )
    }
}

#[derive(Clone, Debug)]
pub struct CatBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct CatBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CatBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<Cat, CatBatch<B>> for CatBatcher<B> {
    fn batch(&self, items: Vec<Cat>) -> CatBatch<B> {
        // let mean = Tensor::<B, 1>::from_floats(MEAN, &self.device).reshape([1, 3, 1, 1]);
        // let std = Tensor::<B, 1>::from_floats(STD, &self.device).reshape([1, 3, 1, 1]);
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 3>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 3, 224, 224]))
            .map(|tensor| tensor / 255.)
            // .map(|tensor| (tensor - mean.clone()) / std.clone())
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(item.label as i64).elem::<B::IntElem>()],
                    &self.device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        CatBatch { images, targets }
    }
}
