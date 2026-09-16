//! ArcFace w600k_r50 inference. Aligned inputs are 112×112 uint8 RGB;
//! embeddings are 512 unnormalized float32 values, as in InsightFace.
pub mod alignment;
mod cnn;
pub mod hub;
mod model;
mod onnx;
mod plan;
use anyhow::{Result, ensure};
use hrx::{
    image::ImageOps,
    inference::{Inference, ModelContext},
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
};
use std::path::Path;
pub const SIZE: usize = 112;
pub const EMBEDDING: usize = 512;
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub device: i32,
    pub max_batch: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            max_batch: 16,
        }
    }
}
/// Shared immutable weights with bounded, private asynchronous inference slots.
pub struct ArcFace {
    cnn: cnn::Cnn,
    images: ImageOps,
}
/// Resident aligned RGB crops and per-face geometry status. Invalid geometry
/// produces black pixels but is an error, not a successful aligned face.
pub struct AlignedFaces {
    crops: Inference,
    status: DeviceTensor,
}
impl AlignedFaces {
    pub fn crops(&self) -> &DeviceTensor {
        &self.crops.outputs()[0]
    }
    pub fn status(&self) -> &DeviceTensor {
        &self.status
    }
}
/// Resident embeddings with geometry status preserved from device alignment.
pub struct FaceInference {
    inference: Inference,
    status: DeviceTensor,
    context: ModelContext,
}
impl FaceInference {
    pub fn embeddings(&self) -> &DeviceTensor {
        &self.inference.outputs()[0]
    }
    pub fn status(&self) -> &DeviceTensor {
        &self.status
    }
    pub fn completion(&self) -> &hrx::Completion {
        self.inference.completion()
    }
    /// Drain inference, check geometry and return only final embeddings. Four
    /// status bytes per face are downloaded; landmarks and crops stay resident.
    pub fn wait(self) -> Result<Vec<[f32; EMBEDDING]>> {
        let status = self.context.download(&self.status)?;
        let embeddings = self.inference.download()?.wait()?.remove(0);
        let status = status.wait()?;
        ensure!(
            status.as_chunks::<4>().0.iter().all(|s| *s == [0, 0, 0, 0]),
            "invalid or degenerate alignment landmarks"
        );
        Ok(embeddings
            .as_chunks::<{ EMBEDDING * 4 }>()
            .0
            .iter()
            .map(|row| {
                std::array::from_fn(|i| f32::from_le_bytes(row[i * 4..][..4].try_into().unwrap()))
            })
            .collect())
    }
}
impl ArcFace {
    /// Load the pinned pretrained model from the Hugging Face cache, fetching it
    /// if needed. Set `HF_HUB_OFFLINE=1` for cached weights only.
    /// Use [`Self::load`] to supply a local file instead.
    pub fn from_pretrained(options: Options) -> Result<Self> {
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        Self::load(hub::weights(false)?, options)
    }

    /// Validate and pack the model, compile kernels, and allocate resident storage.
    pub fn load(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            gpu_index: options.device,
            ..Default::default()
        })?;
        Self::load_in(path, &context, options.max_batch)
    }
    /// Load into an explicit context shared with detection and other models.
    pub fn load_in(
        path: impl AsRef<Path>,
        context: &ModelContext,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!((1..=64).contains(&max_batch), "max_batch must be 1..=64");
        Ok(Self {
            cnn: cnn::Cnn::new(model::load(path.as_ref())?, context, max_batch)?,
            images: ImageOps::new(context, 8)?,
        })
    }
    /// Scheduling and allocation domain for this model's device inputs and outputs.
    pub fn context(&self) -> &ModelContext {
        self.cnn.engine.context()
    }
    /// Submit aligned device RGB crops; outputs are resident, unnormalized f32 embeddings.
    /// Three live requests/retained outputs per shape apply bounded backpressure.
    pub fn submit(&self, crops: &DeviceTensor) -> Result<Inference> {
        self.context().validate(crops)?;
        let desc = crops.desc();
        ensure!(
            desc.dtype() == DType::U8
                && desc.layout() == Layout::Nhwc
                && desc.is_contiguous()
                && desc.shape().len() == 4
                && desc.shape()[1..] == [SIZE, SIZE, 3],
            "expected contiguous NHWC uint8 112×112 RGB crops"
        );
        Ok(self
            .cnn
            .prepare(desc.shape()[0])?
            .submit(std::slice::from_ref(crops))?)
    }
    /// Fit resident F32 `[faces,5,2]` landmarks and sample resident RGB crops.
    /// The image has shape `[1,height,width,3]`; no pixels cross back to the CPU.
    /// One nonempty batch up to max_batch is accepted, with bounded backpressure.
    pub fn align(&self, image: &DeviceTensor, landmarks: &DeviceTensor) -> Result<AlignedFaces> {
        self.context().validate(image)?;
        self.context().validate(landmarks)?;
        let shape = landmarks.desc().shape();
        ensure!(
            shape.len() == 3
                && shape[1..] == [5, 2]
                && (1..=self.cnn.max_batch).contains(&shape[0]),
            "alignment requires 1..=max_batch faces"
        );
        let fit = self.images.similarity_2d(landmarks, &alignment::TEMPLATE)?;
        Ok(AlignedFaces {
            crops: self.images.affine_rgb(
                image,
                &fit.outputs()[1],
                SIZE,
                SIZE,
                hrx::image::RgbSampling::BlackTiesEven,
            )?,
            status: fit.outputs()[2].clone(),
        })
    }
    /// Align and encode a resident image, keeping crop pixels and embeddings on GPU.
    /// Fitting, sampling and inference run on GPU; dependent submissions do not wait.
    pub fn submit_image(
        &self,
        image: &DeviceTensor,
        landmarks: &DeviceTensor,
    ) -> Result<FaceInference> {
        let crops = self.align(image, landmarks)?;
        Ok(FaceInference {
            inference: self.submit(crops.crops())?,
            status: crops.status,
            context: self.context().clone(),
        })
    }
    /// Encode a contiguous batch of aligned 112×112 RGB crops, chunking as needed.
    pub fn embeddings(&self, crops: &[u8]) -> Result<Vec<[f32; EMBEDDING]>> {
        ensure!(
            crops.len().is_multiple_of(SIZE * SIZE * 3),
            "expected complete 112×112 RGB crops"
        );
        let mut out = vec![[0.; EMBEDDING]; crops.len() / (SIZE * SIZE * 3)];
        for (input, out) in crops
            .chunks(self.cnn.max_batch * SIZE * SIZE * 3)
            .zip(out.chunks_mut(self.cnn.max_batch))
        {
            self.cnn
                .prepare(out.len())?
                .acquire_blocking()?
                .submit_host(&[input])?
                .download()?
                .read_into(&mut [bytemuck::cast_slice_mut(out)])?;
        }
        Ok(out)
    }
    /// Synchronized warm host latency including staging, execution and readback.
    pub fn benchmark(&self, crops: &[u8], samples: usize) -> Result<hrx::benchmark::Distribution> {
        ensure!(
            !crops.is_empty() && crops.len() <= self.cnn.max_batch * SIZE * SIZE * 3,
            "benchmark requires one nonempty batch"
        );
        ensure!(samples >= 10, "use at least ten timing samples");
        for _ in 0..10 {
            self.embeddings(crops)?;
        }
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = std::time::Instant::now();
            self.embeddings(crops)?;
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        Ok(hrx::benchmark::Distribution::from_samples(times)?)
    }
    /// Align and embed faces from a packed RGB image and five landmarks per face.
    pub fn embed(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        landmarks: &[[[f32; 2]; 5]],
    ) -> Result<Vec<[f32; EMBEDDING]>> {
        if landmarks.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(width > 0 && height > 0, "invalid RGB image size");
        let image = self.context().upload(
            TensorDesc::new(DType::U8, vec![1, height, width, 3])?.with_layout(Layout::Nhwc)?,
            image,
        )?;
        let mut out = vec![[0.; EMBEDDING]; landmarks.len()];
        for (points, output) in landmarks
            .chunks(self.cnn.max_batch)
            .zip(out.chunks_mut(self.cnn.max_batch))
        {
            let landmarks = self.context().upload(
                TensorDesc::new(DType::F32, vec![points.len(), 5, 2])?,
                bytemuck::cast_slice(points),
            )?;
            output.copy_from_slice(&self.submit_image(&image, &landmarks)?.wait()?);
        }
        Ok(out)
    }
}
/// Cosine similarity; rejects non-finite or zero-length embeddings.
pub fn similarity(a: &[f32; EMBEDDING], b: &[f32; EMBEDDING]) -> Result<f64> {
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "non-finite embedding"
    );
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    let norm = |a: &[f32]| a.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
    let d = (norm(a) * norm(b)).sqrt();
    ensure!(d > 0., "zero embedding");
    Ok(dot / d)
}

#[cfg(test)]
mod tests;
