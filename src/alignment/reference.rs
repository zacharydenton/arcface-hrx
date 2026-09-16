// Test-only frozen pre-FP32 implementation; not a production fallback.
//! InsightFace five-point similarity alignment, with bilinear sampling and black borders.
//! The template and Umeyama procedure retain the attribution in THIRD_PARTY_NOTICES.md.
use anyhow::{Result, ensure};
use nalgebra::{Matrix2, Vector2};
pub const TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];
pub fn transform(points: &[[f32; 2]; 5]) -> Result<[[f64; 3]; 2]> {
    ensure!(
        points.iter().flatten().all(|x| x.is_finite()),
        "landmarks must be finite"
    );
    let mean = |p: &[[f32; 2]; 5]| {
        Vector2::new(
            p.iter().map(|p| p[0]).sum::<f32>() / 5.,
            p.iter().map(|p| p[1]).sum::<f32>() / 5.,
        )
    };
    let src = mean(points);
    let dst = mean(&TEMPLATE);
    let mut covariance = Matrix2::<f32>::zeros();
    let mut variance = [0f32; 2];
    for i in 0..5 {
        let s = Vector2::from(points[i]) - src;
        let d = Vector2::from(TEMPLATE[i]) - dst;
        covariance += d * s.transpose();
        variance[0] += s[0] * s[0];
        variance[1] += s[1] * s[1];
    }
    covariance /= 5.;
    let variance = variance[0] / 5. + variance[1] / 5.;
    ensure!(
        variance.is_finite() && variance > 0.,
        "degenerate landmarks"
    );
    let svd = covariance.svd(true, true);
    let u = svd.u.unwrap().cast::<f64>();
    let vt = svd.v_t.unwrap().cast::<f64>();
    let sign = if covariance.determinant() < 0. {
        -1.
    } else {
        1.
    };
    let mut d = Matrix2::identity();
    d[(1, 1)] = sign;
    let scale =
        (svd.singular_values[0] as f64 + sign * svd.singular_values[1] as f64) / variance as f64;
    let a = u * d * vt * scale;
    let t = dst.cast::<f64>() - a * src.cast::<f64>();
    ensure!(
        a.iter().chain(t.iter()).all(|x| x.is_finite()) && a.determinant().abs() > 1e-20,
        "degenerate transform"
    );
    Ok([[a[(0, 0)], a[(0, 1)], t[0]], [a[(1, 0)], a[(1, 1)], t[1]]])
}
/// Warp a packed RGB image into a 112×112 RGB crop.
pub fn crop(image: &[u8], width: usize, height: usize, points: &[[f32; 2]; 5]) -> Result<Vec<u8>> {
    ensure!(
        width > 0
            && height > 0
            && width.checked_mul(height).and_then(|n| n.checked_mul(3)) == Some(image.len()),
        "invalid RGB image size"
    );
    let t = transform(points)?;
    let det = t[0][0] * t[1][1] - t[0][1] * t[1][0];
    let a = t[1][1] / det;
    let b = -t[0][1] / det;
    let c = -t[1][0] / det;
    let d = t[0][0] / det;
    let tx = -a * t[0][2] - b * t[1][2];
    let ty = -c * t[0][2] - d * t[1][2];
    let mut out = vec![0; 112 * 112 * 3];
    // Use full-precision bilinear coordinates, matching the OpenCV reference
    // used to capture the fixtures. Pixel values round to nearest, ties to even.
    for y in 0..112 {
        for x in 0..112 {
            let xx = a * x as f64 + b * y as f64 + tx;
            let yy = c * x as f64 + d * y as f64 + ty;
            let sx = xx.floor() as i64;
            let sy = yy.floor() as i64;
            let fx = xx - xx.floor();
            let fy = yy - yy.floor();
            for ch in 0..3 {
                let mut value = 0.;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = sx + dx;
                        let py = sy + dy;
                        if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                            value += image[(py as usize * width + px as usize) * 3 + ch] as f64
                                * if dx == 0 { 1. - fx } else { fx }
                                * if dy == 0 { 1. - fy } else { fy };
                        }
                    }
                }
                out[(y * 112 + x) * 3 + ch] = value.round_ties_even().clamp(0., 255.) as u8;
            }
        }
    }
    Ok(out)
}
