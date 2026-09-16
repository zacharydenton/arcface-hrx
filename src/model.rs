use crate::{
    onnx::{Network, Node},
    plan::*,
};
use anyhow::{Context, Result, ensure};
use std::{collections::HashMap, path::Path};
fn bn(net: &Network, n: &Node) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(
        n.op_type() == "BatchNormalization",
        "expected BatchNormalization"
    );
    let g = net.tensor(&n.inputs()[1])?.f64s()?;
    let b = net.tensor(&n.inputs()[2])?.f64s()?;
    let m = net.tensor(&n.inputs()[3])?.f64s()?;
    let v = net.tensor(&n.inputs()[4])?.f64s()?;
    let mut scale = vec![];
    let mut shift = vec![];
    for i in 0..g.len() {
        ensure!(
            v[i] + n.float("epsilon", 1e-5) > 0.,
            "invalid BatchNorm variance"
        );
        let s = g[i] / (v[i] + n.float("epsilon", 1e-5)).sqrt();
        scale.push(s);
        shift.push(b[i] - m[i] * s);
    }
    Ok((scale, shift))
}
pub(crate) fn load(path: &Path) -> Result<Plan> {
    let net = Network::load(path, 112)?;
    ensure!(
        net.outputs().len() == 1 && net.shape(&net.outputs()[0])? == [1, 512],
        "expected w600k_r50 embedding output"
    );
    ensure!(
        net.nodes().iter().filter(|n| n.op_type() == "Conv").count() == 53,
        "expected 53 w600k_r50 convolutions"
    );
    let mut weights = HashMap::new();
    let mut aliases = HashMap::from([(net.inputs()[0].clone(), "nhwc_input".into())]);
    let mut ops = vec![Op {
        kind: "convert",
        name: "convert".into(),
        src: net.inputs()[0].clone(),
        dst: "nhwc_input".into(),
        h: 112,
        w: 112,
        ho: 112,
        wo: 112,
        bytes: 112 * 112 * 8 * 2,
        ..Default::default()
    }];
    let mut index = 0;
    for node in net.nodes() {
        match node.op_type() {
            "Conv" => {
                let name = format!("c{index:02}");
                index += 1;
                let shape = net.tensor(&node.inputs()[1])?.shape()?;
                let (co, ci, taps) = (shape[0], shape[1], shape[2] * shape[3]);
                let mut w = net.tensor(&node.inputs()[1])?.f64s()?;
                let b = net.tensor(&node.inputs()[2])?.f64s()?;
                let parent = net
                    .producer(&node.inputs()[0])
                    .filter(|n| n.op_type() == "BatchNormalization");
                let node_output = node.output()?;
                let add = net
                    .consumers(node_output)
                    .find(|n| n.op_type() == "Add" && n.inputs()[0] == node_output);
                let prelu = net.consumers(node_output).find(|n| n.op_type() == "PRelu");
                let stride = node.integers("strides", &[1, 1])[0] as usize;
                let variant = if taps == 1 {
                    "plain"
                } else if add.is_some() {
                    "add"
                } else if parent.is_some() {
                    "bnprelu"
                } else {
                    "prelu"
                };
                ensure!(
                    (variant == "bnprelu") == parent.is_some()
                        && (variant == "prelu" || variant == "bnprelu") == prelu.is_some(),
                    "unsupported convolution fusion {name}"
                );
                let cp = align(ci, 8);
                let k = align(9 * cp, 32);
                let n = align(co, 64);
                let mut bias = vec![0.; n * if parent.is_some() { 9 } else { 1 }];
                let src = if let Some(bn_node) = parent {
                    ensure!(
                        stride == 1 && net.consumers(bn_node.output()?).count() == 1,
                        "unsupported BN-convolution fold"
                    );
                    let (scale, shift) = bn(&net, bn_node)?;
                    for class in 0..9 {
                        for o in 0..co {
                            let mut value = b[o];
                            for c in 0..ci {
                                for y in 0..3 {
                                    for x in 0..3 {
                                        if (class / 3 != 0 || y != 0)
                                            && (class / 3 != 2 || y != 2)
                                            && (class % 3 != 0 || x != 0)
                                            && (class % 3 != 2 || x != 2)
                                        {
                                            value += w[(o * ci + c) * 9 + y * 3 + x] * shift[c];
                                        }
                                    }
                                }
                            }
                            bias[class * n + o] = value;
                        }
                    }
                    for o in 0..co {
                        for c in 0..ci {
                            for t in 0..9 {
                                w[(o * ci + c) * 9 + t] *= scale[c];
                            }
                        }
                    }
                    bn_node.inputs()[0].clone()
                } else {
                    bias[..co].copy_from_slice(&b);
                    node.inputs()[0].clone()
                };
                if taps == 1 {
                    ensure!(stride == 2, "expected stride-2 shortcut");
                    let mut expanded = vec![0.; co * ci * 9];
                    for i in 0..co * ci {
                        expanded[i * 9 + 4] = w[i];
                    }
                    w = expanded;
                }
                emit16(&mut weights, name.clone(), &pack(&w, co, ci, 9, cp, n, k));
                emit32(&mut weights, format!("{name}_b"), &bias);
                let mut extra = String::new();
                if let Some(a) = add {
                    ensure!(
                        net.consumers(node.output()?).count() == 1,
                        "unfused convolution consumer"
                    );
                    extra = a.inputs()[1].clone();
                    aliases.insert(a.output()?.into(), node.output()?.into());
                }
                if let Some(p) = prelu {
                    ensure!(
                        net.consumers(node.output()?).count() == 1,
                        "unfused PRelu consumer"
                    );
                    let mut slopes = vec![0.; n];
                    slopes[..co].copy_from_slice(&net.tensor(&p.inputs()[1])?.f64s()?);
                    emit32(&mut weights, format!("{name}_slope"), &slopes);
                    aliases.insert(p.output()?.into(), node.output()?.into());
                }
                let s = net.shape(&src)?;
                let out = net.shape(node.output()?)?;
                ops.push(Op {
                    kind: "conv",
                    variant,
                    name,
                    src,
                    dst: node.output()?.into(),
                    extra,
                    h: s[2],
                    w: s[3],
                    stride,
                    cin_pad: cp,
                    cin_stride: storage(ci),
                    k,
                    n,
                    ho: out[2],
                    wo: out[3],
                    tile: if n == 128 { 128 } else { 64 },
                    slope: prelu.is_some(),
                    bytes: out[2] * out[3] * n * 2,
                    ..Default::default()
                });
            }
            "Gemm" => {
                let flat = net
                    .producer(&node.inputs()[0])
                    .context("missing head Flatten")?;
                ensure!(flat.op_type() == "Flatten", "expected head Flatten");
                let before = net.producer(&flat.inputs()[0]).context("missing head BN")?;
                let after = net
                    .consumers(node.output()?)
                    .next()
                    .context("missing final BN")?;
                ensure!(
                    after.output()? == net.outputs()[0]
                        && net.consumers(node.output()?).count() == 1,
                    "unsupported head output"
                );
                ensure!(
                    net.shape(&before.inputs()[0])? == [1, 512, 7, 7],
                    "expected 512×7×7 head input"
                );
                let (a1, s1) = bn(&net, before)?;
                let (a2, s2) = bn(&net, after)?;
                let w = net.tensor(&node.inputs()[1])?.f64s()?;
                let bias = net.tensor(&node.inputs()[2])?.f64s()?;
                let (k, n) = (25088, 512);
                ensure!(
                    net.tensor(&node.inputs()[1])?.shape()? == [n, k],
                    "unsupported head shape"
                );
                let mut packed = vec![0.; n * k];
                let mut b = vec![0.; n];
                for o in 0..n {
                    let mut value = bias[o];
                    for c in 0..512 {
                        for p in 0..49 {
                            let v = w[o * k + c * 49 + p];
                            value += v * s1[c];
                            packed[o * k + p * 512 + c] = v * a1[c] * a2[o];
                        }
                    }
                    b[o] = value * a2[o] + s2[o];
                }
                emit16(&mut weights, "fc".into(), &packed);
                emit32(&mut weights, "fc_b".into(), &b);
                ops.push(Op {
                    kind: "head",
                    name: "fc".into(),
                    src: before.inputs()[0].clone(),
                    dst: "partials".into(),
                    k,
                    n,
                    ho: 1,
                    wo: 1,
                    tile: 64,
                    splits: 28,
                    bytes: 28 * 512 * 4,
                    ..Default::default()
                });
                ops.push(Op {
                    kind: "reduce",
                    name: "fc".into(),
                    src: "partials".into(),
                    dst: "embedding".into(),
                    n,
                    ho: 1,
                    wo: 1,
                    splits: 28,
                    bytes: 512 * 4,
                    ..Default::default()
                });
            }
            "BatchNormalization" | "PRelu" | "Add" | "Flatten" => {}
            other => anyhow::bail!("unsupported ArcFace operator {other}"),
        }
    }
    finish(ops, aliases, weights, &["embedding".into()])
}
