use super::*;
#[test]
#[ignore = "requires gfx1151"]
fn rgb_conversion_and_padding() -> Result<()> {
    use hrx::{
        loom::Specialization,
        model::{Command, Dispatch, ModelSession},
    };

    let mut engine = ModelSession::open_for(0, "gfx1151")?;
    let input = engine.allocate_shared(32 * 3)?;
    let output = engine.allocate_shared(32 * 8 * 2)?;
    let mut spec = Specialization::new("arcface_hwc_u8_to_nhwc_f16");
    spec.set_config("arcface.hwc_u8_to_nhwc_f16.size", "16");
    let kernels =
        unsafe { engine.compile(&[(include_str!("../kernels/hwc_u8_to_nhwc_f16.loom"), spec)])? };
    unsafe {
        engine.record(
            1,
            &[Command::Dispatch(Dispatch::indices(
                kernels[0],
                [2],
                [2, 1, 1],
                vec![input.read(), output.write()],
            ))],
        )?;
    }
    // Unequal channels catch accidental RGB/BGR reversal. Replay with changed
    // colors also checks that every channel, including padding, is overwritten.
    for seed in [0, 73] {
        let rgb: Vec<u8> = (0..32 * 3).map(|i| ((i * 37 + seed) % 256) as u8).collect();
        engine.upload(input, &rgb)?;
        engine.upload(output, &vec![0xff; output.len()])?;
        engine.replay(1)?;
        let mut bytes = vec![0u8; output.len()];
        engine.read_many(&mut [(output, &mut bytes)])?;
        for (p, pixel) in bytes.as_chunks::<16>().0.iter().enumerate() {
            for (c, value) in pixel.as_chunks::<2>().0.iter().enumerate() {
                let expected = if c < 3 {
                    (rgb[p * 3 + c] as f32 - 127.5) / 127.5
                } else {
                    0.
                };
                assert_eq!(
                    half::f16::from_le_bytes([value[0], value[1]]),
                    half::f16::from_f32(expected),
                    "pixel {p}, channel {c}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn input_and_alignment_validation() {
    assert!(alignment::transform(&[[0.; 2]; 5]).is_err());
    assert!(alignment::transform(&[[f32::NAN; 2]; 5]).is_err());
    assert!(alignment::crop(&[], 112, 112, &alignment::TEMPLATE).is_err());
    assert!(similarity(&[0.; 512], &[1.; 512]).is_err());
    assert!((similarity(&[1.; 512], &[1.; 512]).unwrap() - 1.).abs() < 1e-12);
    let t = alignment::transform(&alignment::TEMPLATE).unwrap();
    for (row, want) in t.iter().zip([[1., 0., 0.], [0., 1., 0.]]) {
        for (v, w) in row.iter().zip(want) {
            assert!((v - w).abs() < 1e-4);
        }
    }
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    let norm = |a: &[f32]| a.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
    dot / (norm(a) * norm(b)).sqrt()
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn native_reference_and_replay() -> Result<()> {
    use tract_onnx::prelude::*;
    let path = model_path()?;
    let model = ArcFace::load(
        &path,
        Options {
            device: 0,
            max_batch: 4,
        },
    )?;
    let reference = tract_onnx::onnx()
        .model_for_path(&path)?
        .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())?
        .into_optimized()?
        .into_runnable()?;
    let crops: Vec<u8> = (0..4 * 112 * 112 * 3)
        .map(|i| ((i * 13 + i / 379) % 256) as u8)
        .collect();
    let batch = model.embeddings(&crops)?;
    for (i, crop) in crops.chunks(112 * 112 * 3).enumerate() {
        let blob = tract_ndarray::Array4::from_shape_fn((1, 3, 112, 112), |(_, c, y, x)| {
            (crop[(y * 112 + x) * 3 + c] as f32 - 127.5) / 127.5
        });
        let expected = reference.run(tvec![blob.into_tensor().into()])?;
        let data = expected[0].to_array_view::<f32>()?;
        let want = data.as_slice().unwrap();
        assert!(
            cosine(&batch[i], want) > 0.99995,
            "image {i} cosine {}",
            cosine(&batch[i], want)
        );
        assert_eq!(model.embeddings(crop)?[0], batch[i]);
    }
    assert_eq!(model.embeddings(&crops)?, batch);
    assert!(model.embeddings(&[0; 5]).is_err());
    Ok(())
}
#[test]
#[ignore = "requires pretrained weights; CPU model import"]
fn importer_liveness() -> Result<()> {
    let p = model::load(std::path::Path::new(&model_path()?))?;
    assert_eq!(p.ops.len(), 56);
    assert_eq!(p.buffers, [200704, 1605632, 1605632, 401408]);
    for l in p.ops {
        assert_ne!(l.src_buf, l.dst_buf);
        assert_ne!(l.extra_buf, l.dst_buf);
    }
    Ok(())
}
#[test]
fn malformed_onnx_returns_errors() {
    use onnx_protobuf::{GraphProto, Message, ModelProto, NodeProto, ValueInfoProto};
    assert!(onnx::Network::from_bytes(&[], 112).is_err());
    assert!(onnx::Network::from_bytes(&[0x3a, 0xff], 112).is_err());
    let node = NodeProto {
        op_type: "Conv".into(),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node],
        input: vec![ValueInfoProto {
            name: "x".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let model = ModelProto {
        graph: Some(graph).into(),
        ..Default::default()
    };
    assert!(onnx::Network::from_bytes(&model.write_to_bytes().unwrap(), 112).is_err());
}
#[test]
fn spatial_operators_reject_flattened_input() -> Result<()> {
    use onnx_protobuf::{
        AttributeProto, GraphProto, Message, ModelProto, NodeProto, ValueInfoProto,
        attribute_proto::AttributeType,
    };
    let ints = |name: &str, values: &[i64]| AttributeProto {
        name: name.into(),
        type_: AttributeType::INTS.into(),
        ints: values.to_vec(),
        ..Default::default()
    };
    let text = |name: &str, value: &str| AttributeProto {
        name: name.into(),
        type_: AttributeType::STRING.into(),
        s: value.as_bytes().to_vec(),
        ..Default::default()
    };
    let dir = tempfile::tempdir()?;
    for (op, attribute) in [
        ("Transpose", vec![ints("perm", &[2, 3, 0, 1])]),
        (
            "MaxPool",
            vec![ints("kernel_shape", &[2, 2]), ints("strides", &[2, 2])],
        ),
        (
            "AveragePool",
            vec![ints("kernel_shape", &[2, 2]), ints("strides", &[2, 2])],
        ),
        (
            "Resize",
            vec![
                text("mode", "nearest"),
                text("coordinate_transformation_mode", "asymmetric"),
                text("nearest_mode", "floor"),
            ],
        ),
    ] {
        let value = |name: &str| ValueInfoProto {
            name: name.into(),
            ..Default::default()
        };
        let graph = GraphProto {
            input: vec![value("x")],
            output: vec![value("y")],
            node: vec![
                NodeProto {
                    op_type: "Flatten".into(),
                    input: vec!["x".into()],
                    output: vec!["flat".into()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Shape".into(),
                    input: vec!["x".into()],
                    output: vec!["sizes".into()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: op.into(),
                    input: if op == "Resize" {
                        vec!["flat".into(), "".into(), "".into(), "sizes".into()]
                    } else {
                        vec!["flat".into()]
                    },
                    output: vec!["y".into()],
                    attribute,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(graph).into(),
            ..Default::default()
        };
        let path = dir.path().join(format!("{op}.onnx"));
        std::fs::write(&path, model.write_to_bytes()?)?;
        let error = ArcFace::load(&path, Options::default())
            .err()
            .expect("invalid rank must fail");
        assert_eq!(
            error.to_string(),
            format!("{op}: expected rank-4 input, got rank 2")
        );
    }
    Ok(())
}
#[test]
fn landmark_fixture() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_arcface.json"))?;
    for face in fixture["faces"].as_array().unwrap() {
        let points: [[f32; 2]; 5] = serde_json::from_value(face["kps"].clone())?;
        let want: [[f64; 3]; 2] = serde_json::from_value(face["M"].clone())?;
        let got = alignment::transform(&points)?;
        for (a, b) in got.iter().flatten().zip(want.iter().flatten()) {
            let delta = (f64::from(*a) - b).abs();
            assert!(delta < 2e-4, "alignment delta {delta}");
        }
    }
    Ok(())
}

fn compare_alignment_crops(
    label: &str,
    rgb: &[u8],
    width: usize,
    height: usize,
    points: &[[f32; 2]; 5],
    gpu: Option<(&hrx::image::ImageOps, &hrx::inference::ModelContext)>,
) -> Result<()> {
    let got = alignment::crop(rgb, width, height, points)?;
    if let Some((ops, context)) = gpu {
        let input = context.upload(
            TensorDesc::new(DType::U8, vec![1, height, width, 3])?.with_layout(Layout::Nhwc)?,
            rgb,
        )?;
        let landmarks = context.upload(
            TensorDesc::new(DType::F32, vec![1, 5, 2])?,
            bytemuck::cast_slice(points),
        )?;
        let fitted = ops.similarity_2d(&landmarks, &alignment::TEMPLATE)?;
        let device = ops
            .affine_rgb(
                &input,
                &fitted.outputs()[1],
                112,
                112,
                hrx::image::RgbSampling::BlackTiesEven,
            )?
            .download()?
            .wait()?
            .remove(0);
        let max = device
            .iter()
            .zip(&got)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let changed = device.iter().zip(&got).filter(|(a, b)| a != b).count();
        eprintln!("{label}: GPU/CPU FP32 {changed} changed channels, max delta {max}");
        assert!(max <= 1, "{label}: GPU/CPU crop error {max}");
    }
    let old = alignment::reference::crop(rgb, width, height, points)?;
    let differences: Vec<u8> = got.iter().zip(&old).map(|(a, b)| a.abs_diff(*b)).collect();
    let changed = differences.iter().filter(|d| **d != 0).count();
    let max = *differences.iter().max().unwrap();
    let mean = differences.iter().map(|d| *d as f64).sum::<f64>() / differences.len() as f64;
    eprintln!(
        "{label}: {changed}/{} changed channels, max delta {max}, mean delta {mean:.6}",
        got.len()
    );
    // Qualification bounds for these fixtures, not a universal FP32 guarantee.
    // At half-integer pixel values, a tiny coordinate difference can change
    // many rounded channels by one; changed-channel count is diagnostic only.
    assert!(max <= 1, "{label}: max channel delta {max}");
    Ok(())
}

#[test]
fn fp32_alignment_crop_parity() -> Result<()> {
    alignment_crop_parity(None)
}

#[test]
#[ignore = "requires gfx1151 and Loom compiler"]
fn gpu_alignment_crop_parity() -> Result<()> {
    let context = ModelContext::new(hrx::execution::RuntimeOptions::default())?;
    let ops = ImageOps::new(&context, 8)?;
    alignment_crop_parity(Some((&ops, &context)))
}

fn alignment_crop_parity(gpu: Option<(&ImageOps, &ModelContext)>) -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_arcface.json"))?;
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let (w, h) = image.dimensions();
    for (i, face) in fixture["faces"].as_array().unwrap().iter().enumerate() {
        compare_alignment_crops(
            &format!("fixture face {i}"),
            &image,
            w as usize,
            h as usize,
            &serde_json::from_value(face["kps"].clone())?,
            gpu,
        )?;
    }
    for (label, width, height, scale, angle, tx, ty) in [
        ("identity", 112, 112, 1., 0., 0., 0.),
        ("fractional", 256, 256, 1., 0., 0.375, 0.125),
        ("rotation", 256, 256, 1.5, 0.7, 100., 5.),
        ("top-left border", 112, 112, 1., 0., -40.25, -30.75),
        ("bottom-right border", 112, 112, 1., 0., 40.25, 30.75),
        ("outside", 112, 112, 1., 0., -300., -300.),
        ("large x", 8192, 256, 1.5, 0.2, 8000., 0.),
        ("large y", 256, 8192, 1.5, -0.2, 0., 8000.),
        ("small face", 112, 112, 0.0001, 0.3, 50., 50.),
    ] {
        let angle: f32 = angle;
        let (sin, cos) = angle.sin_cos();
        let points = alignment::TEMPLATE.map(|[x, y]| {
            [
                scale * (cos * x - sin * y) + tx,
                scale * (sin * x + cos * y) + ty,
            ]
        });
        let rgb: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 37 + i / 379) % 256) as u8)
            .collect();
        compare_alignment_crops(label, &rgb, width, height, &points, gpu)?;
    }
    let near_collinear = [
        [20., 20.],
        [40., 20.],
        [60., 20.],
        [80., 20.],
        [100., 20.00001],
    ];
    let rgb: Vec<u8> = (0..112 * 112 * 3).map(|i| (i % 256) as u8).collect();
    compare_alignment_crops("near collinear", &rgb, 112, 112, &near_collinear, gpu)?;
    // Unrepresentable/overflowing or collapsed geometry must fail, not emit NaNs.
    for points in [
        [[f32::INFINITY; 2]; 5],
        [[f32::MAX; 2]; 5],
        [[8192.; 2]; 5],
        alignment::TEMPLATE.map(|[x, y]| [x * 1e30, y * 1e30]),
    ] {
        assert!(alignment::crop(&rgb, 112, 112, &points).is_err());
    }
    Ok(())
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn insightface_fixture() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_arcface.json"))?;
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let (w, h) = image.dimensions();
    let rgb = image.into_raw();
    let model = ArcFace::load(
        model_path()?,
        Options {
            max_batch: 2,
            ..Options::default()
        },
    )?;
    let mut expected = vec![];
    let mut landmarks = vec![];
    for face in fixture["faces"].as_array().unwrap() {
        landmarks.push(serde_json::from_value::<[[f32; 2]; 5]>(
            face["kps"].clone(),
        )?);
        expected.push(serde_json::from_value::<Vec<f32>>(
            face["embedding"].clone(),
        )?);
    }
    let before = model.context().runtime().statistics().downloaded_bytes;
    let got = model.embed(&rgb, w as usize, h as usize, &landmarks)?;
    let readback = model.context().runtime().statistics().downloaded_bytes - before;
    assert_eq!(
        readback, 0,
        "host-visible embeddings and geometry status need no explicit readback copy"
    );
    assert_eq!(got, model.embed(&rgb, w as usize, h as usize, &landmarks)?);
    let image = model.context().upload(
        TensorDesc::new(DType::U8, vec![1, h as usize, w as usize, 3])?
            .with_layout(Layout::Nhwc)?,
        &rgb,
    )?;
    for count in [0, 3] {
        let points = model
            .context()
            .allocate(TensorDesc::new(DType::F32, vec![count, 5, 2])?)?;
        assert!(model.submit_image(&image, &points).is_err());
    }
    let points = model
        .context()
        .upload(TensorDesc::new(DType::F32, vec![1, 5, 2])?, &[0; 40])?;
    assert!(model.submit_image(&image, &points)?.wait().is_err());
    let mut legacy = Vec::new();
    for points in &landmarks {
        let crop = alignment::reference::crop(&rgb, w as usize, h as usize, points)?;
        legacy.push(model.embeddings(&crop)?[0]);
    }
    for (i, (a, b)) in got.iter().zip(&legacy).enumerate() {
        let similarity = cosine(a, b);
        eprintln!("FP32/FP64 face {i}: embedding cosine {similarity:.12}");
        assert!(similarity > 0.99995, "FP32/FP64 cosine {similarity}");
    }
    for (a, b) in got.iter().zip(&expected) {
        assert!(cosine(a, b) > 0.99995, "cosine {}", cosine(a, b));
    }
    for i in 0..got.len() {
        for j in 0..got.len() {
            assert!((cosine(&got[i], &got[j]) - cosine(&expected[i], &expected[j])).abs() < 0.001);
            assert!((cosine(&got[i], &got[j]) - cosine(&legacy[i], &legacy[j])).abs() < 0.001);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn convolution_tiles_preserve_embeddings_across_batch_boundaries() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_arcface.json"))?;
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let landmarks: Vec<[[f32; 2]; 5]> = fixture["faces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|face| serde_json::from_value(face["kps"].clone()))
        .collect::<std::result::Result<_, _>>()?;
    let model = ArcFace::load(
        model_path()?,
        Options {
            max_batch: 64,
            ..Default::default()
        },
    )?;
    let run = |points: &[[[f32; 2]; 5]]| {
        model.embed(
            &image,
            image.width() as usize,
            image.height() as usize,
            points,
        )
    };
    let reference = landmarks
        .iter()
        .map(|points| Ok(run(std::slice::from_ref(points))?[0]))
        .collect::<Result<Vec<_>>>()?;
    // Odd/tail tile counts, multiple workgroups and the maximum supported batch.
    // Change face order on replay to catch stale input/output tile storage.
    for batch in [1, 2, 3, 6, 16, 32, 64] {
        let points: Vec<_> = (0..batch).map(|i| landmarks[i % landmarks.len()]).collect();
        let _ = run(&points)?;
        let points: Vec<_> = (0..batch)
            .map(|i| landmarks[(i + 1) % landmarks.len()])
            .collect();
        let before = model.context().runtime().statistics();
        let actual = run(&points)?;
        let after = model.context().runtime().statistics();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.copied_bytes, before.copied_bytes);
        assert_eq!(after.submissions - before.submissions, 1);
        for (i, embedding) in actual.iter().enumerate() {
            assert_eq!(
                embedding,
                &reference[(i + 1) % reference.len()],
                "batch {batch}, row {i}"
            );
        }
    }
    Ok(())
}

fn model_path() -> Result<std::path::PathBuf> {
    match std::env::var_os("ARCFACE_MODEL") {
        Some(path) => Ok(path.into()),
        None => hub::weights(false),
    }
}
