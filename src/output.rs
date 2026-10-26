use crate::input::Input;
use anyhow::Result;
use std::path::Path;
pub struct Map {
    pub name: String,
    pub channels: usize,
    pub values: Vec<f32>,
}
impl Map {
    pub fn new(name: &str, c: usize, size: usize) -> Self {
        Self {
            name: name.into(),
            channels: c,
            values: vec![0.; size * size * c],
        }
    }
}
// Area integration, matching the upstream reconstruction of cropped output.
fn area(src: &[f32], size: usize, c: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.; w * h * c];
    let sx = size as f64 / w as f64;
    let sy = size as f64 / h as f64;
    for y in 0..h {
        let y0 = y as f64 * sy;
        let y1 = (y + 1) as f64 * sy;
        for x in 0..w {
            let x0 = x as f64 * sx;
            let x1 = (x + 1) as f64 * sx;
            for yy in y0.floor() as usize..(y1.ceil() as usize).min(size) {
                let wy = (y1.min((yy + 1) as f64) - y0.max(yy as f64)).max(0.);
                for xx in x0.floor() as usize..(x1.ceil() as usize).min(size) {
                    let wx = (x1.min((xx + 1) as f64) - x0.max(xx as f64)).max(0.);
                    for ch in 0..c {
                        out[(y * w + x) * c + ch] +=
                            (src[(yy * size + xx) * c + ch] as f64 * wx * wy / (sx * sy)) as f32;
                    }
                }
            }
        }
    }
    out
}
pub fn save(map: &Map, input: &Input, out: &Path, formats: &[String]) -> Result<()> {
    let [ys, ye, xs, xe] = input.roi;
    let w = xe - xs;
    let h = ye - ys;
    let c = map.channels;
    let mut crop = area(&map.values, input.size, c, w, h);
    let normal = map.name == "normal";
    for p in crop.chunks_exact_mut(c) {
        if normal {
            let n = p.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in p {
                *v /= n;
            }
        } else {
            for v in p {
                *v = v.clamp(-1., 1.) * 0.5 + 0.5;
            }
        }
    }
    let mut rgb = vec![0.; input.width * input.height * 3];
    for y in 0..h {
        for x in 0..w {
            let dst = (y + ys) * input.width + x + xs;
            for ch in 0..3 {
                rgb[dst * 3 + ch] =
                    crop[(y * w + x) * c + ch.min(c - 1)] * input.original_mask[dst];
            }
        }
    }
    for format in formats {
        match format.as_str() {
            "exr" => {
                let img = image::Rgb32FImage::from_raw(
                    input.width as u32,
                    input.height as u32,
                    rgb.clone(),
                )
                .unwrap();
                img.save(out.join(format!("{}.exr", map.name)))?;
            }
            "png" => {
                let mut bytes = Vec::with_capacity(input.width * input.height * 4);
                for (i, p) in rgb.chunks_exact(3).enumerate() {
                    for v in p {
                        let v = if normal { v * 0.5 + 0.5 } else { *v };
                        bytes.push((v.clamp(0., 1.) * 255.).round() as u8);
                    }
                    bytes.push((input.original_mask[i].clamp(0., 1.) * 255.).round() as u8);
                }
                image::RgbaImage::from_raw(input.width as u32, input.height as u32, bytes)
                    .unwrap()
                    .save(out.join(format!("{}.png", map.name)))?;
            }
            _ => unreachable!("validated format"),
        }
    }
    Ok(())
}
