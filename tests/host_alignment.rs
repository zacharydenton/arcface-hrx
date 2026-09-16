//! Qualification against externally exported production crops. Fixtures contain
//! decoded RGB, five-point landmarks and host-aligned reference RGB, so decoder
//! differences cannot masquerade as alignment changes.
use anyhow::{Result, ensure};
use arcface_hrx::{ArcFace, Options, similarity};
use hrx::tensor::{DType, Layout, TensorDesc};
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Fixture {
    name: String,
    width: usize,
    height: usize,
    rgb: String,
    reference: String,
    crop_us: u64,
    landmarks: Vec<[[f32; 2]; 5]>,
}

#[test]
#[ignore = "requires gfx1151, cached weights and ARCFACE_HOST_ALIGNMENT_FIXTURES"]
fn production_host_alignment_quality_gate() -> Result<()> {
    let directory = PathBuf::from(std::env::var("ARCFACE_HOST_ALIGNMENT_FIXTURES")?);
    let fixtures: Vec<Fixture> =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
    ensure!(!fixtures.is_empty(), "no alignment fixtures");
    let model = ArcFace::from_pretrained(Options {
        max_batch: 64,
        ..Default::default()
    })?;
    let mut minimum = 1f64;
    let mut reference_embeddings = Vec::new();
    let mut device_embeddings = Vec::new();
    for fixture in fixtures {
        let rgb = std::fs::read(directory.join(fixture.rgb))?;
        let crops = std::fs::read(directory.join(fixture.reference))?;
        ensure!(
            !fixture.landmarks.is_empty() && fixture.landmarks.len() <= 64,
            "invalid fixture face count"
        );
        ensure!(
            crops.len() == fixture.landmarks.len() * 112 * 112 * 3,
            "invalid reference crop bytes"
        );
        let reference = model.embeddings(&crops)?;
        // Prepare the image/landmark shape and model once before warm timing.
        let _ = model.embed(&rgb, fixture.width, fixture.height, &fixture.landmarks)?;
        let start = Instant::now();
        let actual = model.embed(&rgb, fixture.width, fixture.height, &fixture.landmarks)?;
        let device_time = start.elapsed();
        let start = Instant::now();
        let _ = model.embeddings(&crops)?;
        let host_embedding_time = start.elapsed();
        let image = model.context().upload(
            TensorDesc::new(DType::U8, vec![1, fixture.height, fixture.width, 3])?
                .with_layout(Layout::Nhwc)?,
            &rgb,
        )?;
        let landmarks = model.context().upload(
            TensorDesc::new(DType::F32, vec![fixture.landmarks.len(), 5, 2])?,
            bytemuck::cast_slice(&fixture.landmarks),
        )?;
        let aligned = model.align(&image, &landmarks)?;
        let gpu_crops = model.context().download(aligned.crops())?.wait()?;
        let max_channel_delta = crops
            .iter()
            .zip(&gpu_crops)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let mean_channel_delta = crops
            .iter()
            .zip(&gpu_crops)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / crops.len() as f64;
        let cosine = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| similarity(a, b))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(1f64, f64::min);
        minimum = minimum.min(cosine);
        eprintln!(
            "{}: {}x{}, faces={}, cosine={cosine:.12}, max_channel_delta={max_channel_delta}, mean_channel_delta={mean_channel_delta:.6}, host_crop={}us, host_embedding={host_embedding_time:?}, device_alignment_and_embedding={device_time:?}",
            fixture.name,
            fixture.width,
            fixture.height,
            actual.len(),
            fixture.crop_us
        );
        reference_embeddings.extend(reference);
        device_embeddings.extend(actual);
    }
    let mut max_score_change = 0f64;
    for a in 0..reference_embeddings.len() {
        for b in a + 1..reference_embeddings.len() {
            let before = similarity(&reference_embeddings[a], &reference_embeddings[b])?;
            let after = similarity(&device_embeddings[a], &device_embeddings[b])?;
            max_score_change = max_score_change.max((before - after).abs());
        }
    }
    eprintln!("minimum cosine={minimum:.12}, maximum pairwise score change={max_score_change:.12}");
    ensure!(
        minimum > 0.99995,
        "host alignment embedding drift: cosine={minimum}"
    );
    ensure!(
        max_score_change < 0.001,
        "host alignment pairwise score drift: {max_score_change}"
    );
    Ok(())
}
