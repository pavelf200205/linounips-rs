use crate::host::Packed;
use anyhow::{Context, Result, ensure};
use burn::tensor::FloatDType;
use std::path::{Path, PathBuf};
// Legacy NumPy RandomState MT19937: stable shuffles without altering pixel groups.
pub struct Rng {
    state: [u32; 624],
    index: usize,
}
impl Rng {
    pub fn new(seed: u32) -> Self {
        let mut s = [0u32; 624];
        s[0] = seed;
        for i in 1..624 {
            s[i] = 1812433253u32
                .wrapping_mul(s[i - 1] ^ (s[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state: s,
            index: 624,
        }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    pub fn uniform(&mut self) -> f32 {
        let a = (self.next() >> 5) as f64;
        let b = (self.next() >> 6) as f64;
        ((a * 67108864. + b) / 9007199254740992.) as f32
    }
    pub fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let mut mask = i as u32;
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            let j = loop {
                let j = self.next() & mask;
                if j <= i as u32 {
                    break j as usize;
                }
            };
            v.swap(i, j);
        }
    }
}
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1. {
        (1.25 * x - 2.25) * x * x + 1.
    } else if x < 2. {
        ((-0.75 * x + 3.75) * x - 6.) * x + 3.
    } else {
        0.
    }
}
// OpenCV INTER_CUBIC convention (a=-0.75, half-pixel coordinates).
pub fn resize_cubic(src: &[f32], w: usize, h: usize, c: usize, size: usize) -> Vec<f32> {
    let mut temp = vec![0.; h * size * c];
    for y in 0..h {
        for ox in 0..size {
            let f = (ox as f64 + 0.5) * w as f64 / size as f64 - 0.5;
            let base = f.floor() as isize;
            for ch in 0..c {
                let mut sum = 0.;
                for j in -1..=2 {
                    let ix = (base + j).clamp(0, w as isize - 1) as usize;
                    sum += src[(y * w + ix) * c + ch] as f64 * cubic(f - (base + j) as f64);
                }
                temp[(y * size + ox) * c + ch] = sum as f32;
            }
        }
    }
    let mut out = vec![0.; size * size * c];
    for oy in 0..size {
        let f = (oy as f64 + 0.5) * h as f64 / size as f64 - 0.5;
        let base = f.floor() as isize;
        for x in 0..size {
            for ch in 0..c {
                let mut sum = 0.;
                for j in -1..=2 {
                    let iy = (base + j).clamp(0, h as isize - 1) as usize;
                    sum += temp[(iy * size + x) * c + ch] as f64 * cubic(f - (base + j) as f64);
                }
                out[(oy * size + x) * c + ch] = sum as f32;
            }
        }
    }
    out
}
pub struct Input {
    pub images: Vec<Packed>,
    pub mask: Vec<bool>,
    pub original_mask: Vec<f32>,
    pub size: usize,
    pub width: usize,
    pub height: usize,
    pub roi: [usize; 4],
    pub files: Vec<PathBuf>,
}
impl Input {
    pub fn load(
        dir: &Path,
        limit: isize,
        prefix: &str,
        max_resolution: usize,
        dtype: FloatDType,
        normalization: &str,
        rng: &mut Rng,
    ) -> Result<Self> {
        let mut files = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.file_name().unwrap().to_string_lossy().starts_with(prefix)
                    && p.file_name().unwrap() != "mask.png"
                    && matches!(
                        p.extension()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_ascii_lowercase()
                            .as_str(),
                        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "exr"
                    )
            })
            .collect::<Vec<_>>();
        files.sort();
        if limit > 0 {
            files.truncate(limit as usize)
        }
        ensure!(!files.is_empty(), "No matching images in {}", dir.display());
        let first = image::open(&files[0])?;
        let w = first.width() as usize;
        let h = first.height() as usize;
        drop(first);
        let maskpath = dir.join("mask.png");
        let original_mask = if maskpath.exists() {
            let m = image::open(maskpath)?;
            ensure!(
                m.width() as usize == w && m.height() as usize == h,
                "Mask dimensions differ from images"
            );
            m.to_luma32f().into_raw()
        } else {
            vec![1.; w * h]
        };
        let mut ymin = h;
        let mut ymax = 0;
        let mut xmin = w;
        let mut xmax = 0;
        for y in 0..h {
            for x in 0..w {
                if original_mask[y * w + x] > 0. {
                    ymin = ymin.min(y);
                    ymax = ymax.max(y);
                    xmin = xmin.min(x);
                    xmax = xmax.max(x);
                }
            }
        }
        ensure!(ymin < h, "Mask contains no foreground");
        let mut roi = [0, h, 0, w];
        if ymin > 8 && xmin > 8 && ymax + 8 <= h && xmax + 8 <= w {
            let dh = ymax - ymin;
            let dw = xmax - xmin;
            if dh > dw {
                roi = [
                    ymin - 8,
                    ymax + 8,
                    xmin.saturating_sub((dh - dw) / 2 + 8),
                    (xmax + (dh - dw) / 2 + 8).min(w),
                ];
            } else {
                roi = [
                    ymin.saturating_sub((dw - dh) / 2 + 8),
                    (ymax + (dw - dh) / 2 + 8).min(h),
                    xmin - 8,
                    xmax + 8,
                ];
            }
        }
        let [ys, ye, xs, xe] = roi;
        let rh = ye - ys;
        let rw = xe - xs;
        let size = ((rh.max(rw) / 512) * 512)
            .max(512)
            .min(max_resolution / 512 * 512);
        let crop = |data: &[f32], c: usize| {
            let mut out = Vec::with_capacity(rw * rh * c);
            for y in ys..ye {
                out.extend_from_slice(&data[(y * w + xs) * c..(y * w + xe) * c]);
            }
            out
        };
        let mask: Vec<bool> = resize_cubic(&crop(&original_mask, 1), rw, rh, 1, size)
            .into_iter()
            .map(|v| v > 0.5)
            .collect();
        let count = mask.iter().filter(|v| **v).count();
        ensure!(count > 0, "Resized mask is empty");
        let mut images = Vec::new();
        for file in &files {
            let im = image::open(file).with_context(|| format!("Read {}", file.display()))?;
            ensure!(
                im.width() as usize == w && im.height() as usize == h,
                "Image dimensions differ"
            );
            let depth = match im.color() {
                image::ColorType::Rgb8
                | image::ColorType::Rgba8
                | image::ColorType::L8
                | image::ColorType::La8 => 255.,
                image::ColorType::Rgb16
                | image::ColorType::Rgba16
                | image::ColorType::L16
                | image::ColorType::La16 => 65535.,
                _ => 0.,
            };
            let data = im.to_rgb32f().into_raw();
            drop(im);
            let mut data = resize_cubic(&crop(&data, 3), rw, rh, 3, size);
            if depth > 0. {
                for v in &mut data {
                    *v = (*v * depth).round().clamp(0., depth) / depth;
                }
            }
            let mut mean = 0f64;
            let mut max = 0f32;
            for (i, m) in mask.iter().enumerate() {
                if *m {
                    let v = (data[3 * i] + data[3 * i + 1] + data[3 * i + 2]) / 3.;
                    mean += v as f64;
                    max = max.max(v);
                }
            }
            let scale = if normalization == "eval" {
                let a = rng.uniform();
                (1. - a) * (mean / count as f64) as f32 + a * max
            } else {
                max
            };
            for v in &mut data {
                *v /= scale + 1e-6;
            }
            images.push(Packed::new(data, dtype));
        }
        Ok(Self {
            images,
            mask,
            original_mask,
            size,
            width: w,
            height: h,
            roi,
            files,
        })
    }
    // tile_stride is an interleaved grid, not contiguous image crops.
    pub fn block(&self, k: usize, dtype: FloatDType) -> (Vec<Packed>, Vec<bool>) {
        let s = self.size / 512;
        let by = k / s;
        let bx = k % s;
        let mut mask = vec![false; 512 * 512];
        let mut images = Vec::new();
        for image in &self.images {
            let mut v = vec![0.; 3 * 512 * 512];
            for y in 0..512 {
                for x in 0..512 {
                    let p = (y * s + by) * self.size + x * s + bx;
                    mask[y * 512 + x] = self.mask[p];
                    for c in 0..3 {
                        v[(c * 512 + y) * 512 + x] = image.get(p * 3 + c);
                    }
                }
            }
            images.push(Packed::new(v, dtype));
        }
        (images, mask)
    }
}
