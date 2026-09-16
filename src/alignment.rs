//! InsightFace five-point similarity alignment, with bilinear sampling and black borders.
//! The template and Umeyama procedure retain the attribution in THIRD_PARTY_NOTICES.md.
use anyhow::{Result, ensure};
#[cfg(test)]
pub(crate) mod reference;
pub const TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];
pub fn transform(points: &[[f32; 2]; 5]) -> Result<[[f32; 3]; 2]> {
    Ok(hrx::image::fit_similarity_2d(points, &TEMPLATE)?)
}
/// Warp a packed RGB image into a 112×112 RGB crop.
pub fn crop(image: &[u8], width: usize, height: usize, points: &[[f32; 2]; 5]) -> Result<Vec<u8>> {
    ensure!(
        width > 0
            && height > 0
            && width.checked_mul(height).and_then(|n| n.checked_mul(3)) == Some(image.len()),
        "invalid RGB image size"
    );
    let [[a, b, tx], [c, d, ty]] = hrx::image::invert_affine(transform(points)?)?;
    let mut out = vec![0; 112 * 112 * 3];
    // FP32 throughout, without quantizing bilinear weights. Black borders and
    // nearest, ties-to-even pixel rounding retain the reference contract.
    for y in 0..112 {
        for x in 0..112 {
            let xx = a * x as f32 + b * y as f32 + tx;
            let yy = c * x as f32 + d * y as f32 + ty;
            // Skip fully out-of-bounds samples before integer conversion. This
            // also prevents saturated coordinates overflowing at sx/sy + 1.
            if !(xx > -1. && yy > -1. && xx < width as f32 && yy < height as f32) {
                continue;
            }
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
                            value += image[(py as usize * width + px as usize) * 3 + ch] as f32
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
