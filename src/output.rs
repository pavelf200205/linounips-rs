use crate::input::Input;
use anyhow::{Context, Result, ensure};
use std::path::Path;

/// Compare decoded data rather than compressed file bytes or visually rounded PNGs.
/// EXR is required so a PNG-only match cannot conceal floating-point differences.
pub fn compare(reference: &Path, candidate: &Path) -> Result<()> {
    let files = |dir: &Path| -> Result<std::collections::BTreeSet<String>> {
        let mut names = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("exr" | "png")
                )
            {
                names.insert(entry.file_name().to_string_lossy().into_owned());
            }
        }
        Ok(names)
    };
    let names = files(reference)?;
    ensure!(
        names.iter().any(|n| n.ends_with(".exr")),
        "Reference must contain EXR outputs for exact float comparison"
    );
    ensure!(names == files(candidate)?, "Output file sets differ");
    let mut failed = false;
    for name in names {
        let a = image::open(reference.join(&name))
            .with_context(|| format!("Decode reference {name}"))?;
        let b = image::open(candidate.join(&name))
            .with_context(|| format!("Decode candidate {name}"))?;
        ensure!(
            a.width() == b.width() && a.height() == b.height() && a.color() == b.color(),
            "Image shape/type differs: {name}"
        );
        let differences = if name.ends_with(".exr") {
            let a = a.to_rgba32f();
            let b = b.to_rgba32f();
            a.as_raw()
                .iter()
                .zip(b.as_raw())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        } else {
            a.as_bytes()
                .iter()
                .zip(b.as_bytes())
                .filter(|(x, y)| x != y)
                .count()
        };
        println!("{name}: {differences} unequal components");
        failed |= differences != 0;
    }
    ensure!(!failed, "Output equality FAILED (zero tolerance)");
    println!("All decoded outputs are identical.");
    Ok(())
}

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
    let inv_area = 1.0 / (sx * sy);
    let x_spans: Vec<(usize, Vec<f64>)> = (0..w)
        .map(|x| {
            let x0 = x as f64 * sx;
            let x1 = (x + 1) as f64 * sx;
            let x_start = x0.floor() as usize;
            let x_end = (x1.ceil() as usize).min(size);
            let weights = (x_start..x_end)
                .map(|xx| (x1.min((xx + 1) as f64) - x0.max(xx as f64)).max(0.) * inv_area)
                .collect();
            (x_start, weights)
        })
        .collect();

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(h.max(1));
    let rows_per_worker = h.div_ceil(workers);
    let row_stride = w * c;
    let x_spans_ref = &x_spans;
    std::thread::scope(|s| {
        for (chunk_idx, out_chunk) in out.chunks_mut(rows_per_worker * row_stride).enumerate() {
            let y_base = chunk_idx * rows_per_worker;
            let y_count = out_chunk.len() / row_stride;
            s.spawn(move || {
                for local_y in 0..y_count {
                    let y = y_base + local_y;
                    let y0 = y as f64 * sy;
                    let y1 = (y + 1) as f64 * sy;
                    let yy_start = y0.floor() as usize;
                    let yy_end = (y1.ceil() as usize).min(size);
                    let dst_row = &mut out_chunk[local_y * row_stride..(local_y + 1) * row_stride];
                    for (x, (x_start, wx_list)) in x_spans_ref.iter().enumerate() {
                        let dst = &mut dst_row[x * c..(x + 1) * c];
                        for yy in yy_start..yy_end {
                            let wy = (y1.min((yy + 1) as f64) - y0.max(yy as f64)).max(0.);
                            let row_off = yy * size;
                            for (k, &wx_scaled) in wx_list.iter().enumerate() {
                                let xx = *x_start + k;
                                let w_xy = wx_scaled * wy;
                                let src_px = &src[(row_off + xx) * c..(row_off + xx + 1) * c];
                                for ch in 0..c {
                                    dst[ch] += (src_px[ch] as f64 * w_xy) as f32;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
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
            let mask_val = input.original_mask[dst];
            let src_base = (y * w + x) * c;
            for ch in 0..3 {
                rgb[dst * 3 + ch] = crop[src_base + ch.min(c - 1)] * mask_val;
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
