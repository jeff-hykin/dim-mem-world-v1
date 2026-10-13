//! The three SigLIP2 members, per patch. Each frame becomes one grid of L2-normalized embeddings per member (every
//! vision token pooled ALONE by the model's attention head, MaskCLIP-style, which lands it in the text-aligned space);
//! a question becomes one text embedding per member, and a patch's score is their dot product.
//!
//! Preprocessing matches transformers' processors: the fixed-resolution member squashes the frame to its square
//! (`image_size`), NaFlex members keep the aspect ratio inside a patch budget (`naflex_size`) and see the frame as a
//! row-major list of 16x16x3 patches, with the 16x16 position grid resized to the frame's grid (`resize_pos`).
use crate::onnx;
use anyhow::{bail, Context, Result};
use dimos_recording::image::Rgb;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The three members, in the order they're shown and stored (the release's file prefixes).
pub const MEMBERS: [&str; 3] = ["base224", "base_naflex", "so400m_naflex"];

/// `<key>.json` from the release.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spec {
    pub repo: String,
    pub naflex: bool,
    pub dim: usize,
    pub patch: u32,
    pub text_length: usize,
    #[serde(default)]
    pub image_size: Option<u32>,
    #[serde(default)]
    pub max_patches: Option<u32>,
    #[serde(default)]
    pub pos_grid_side: Option<usize>,
}

/// One frame's embeddings for one member: `rows * cols` cells, row-major, `dim` floats each.
#[derive(Debug, Clone)]
pub struct Grid {
    pub rows: u32,
    pub cols: u32,
    pub dim: usize,
    pub data: Vec<f32>,
}

pub struct Member {
    pub key: String,
    pub spec: Spec,
    /// built for one patch count at a time (CoreML compiles fixed shapes; a NaFlex count follows the frame's shape)
    vision: Mutex<Option<(usize, Arc<onnx::Model>)>>,
    vision_path: std::path::PathBuf,
    text: Mutex<Option<onnx::Model>>,
    text_path: std::path::PathBuf,
    tokenizer: tokenizers::Tokenizer,
    pos_grid: Option<Vec<f32>>,
}

impl Member {
    /// Loads `<dir>/<key>.{json,vision.onnx,text.onnx,tokenizer.json,pos_grid.bin}`. The text tower is loaded on first use
    /// (generation never needs it).
    pub fn load(dir: &Path, key: &str) -> Result<Member> {
        let spec: Spec = serde_json::from_slice(&std::fs::read(dir.join(format!("{key}.json"))).with_context(|| format!("{key}.json"))?)?;
                let tokenizer = tokenizers::Tokenizer::from_file(dir.join(format!("{key}.tokenizer.json"))).map_err(|e| anyhow::anyhow!("{key} tokenizer: {e}"))?;
        let pos_grid = if spec.naflex {
            let bytes = std::fs::read(dir.join(format!("{key}.pos_grid.bin")))?;
            Some(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        } else {
            None
        };
        Ok(Member { key: key.into(), spec, vision: Mutex::new(None), vision_path: dir.join(format!("{key}.vision.onnx")), text: Mutex::new(None), text_path: dir.join(format!("{key}.text.onnx")), tokenizer, pos_grid })
    }

    pub fn device(&self) -> onnx::Device {
        self.vision.lock().unwrap().as_ref().map(|(_, model)| model.device()).unwrap_or_else(onnx::wanted_device)
    }

    /// The vision session for `n` patches (0 = the fixed-size member).
    fn vision(&self, n: usize) -> Result<Arc<onnx::Model>> {
        let mut guard = self.vision.lock().unwrap();
        if let Some((built_for, model)) = guard.as_ref() {
            if *built_for == n {
                return Ok(model.clone());
            }
        }
        let fixed: Vec<(&str, i64)> = if n == 0 { vec![("batch", 1)] } else { vec![("n", n as i64)] };
        let model = Arc::new(onnx::Model::load(&self.vision_path, &fixed)?);
        *guard = Some((n, model.clone()));
        Ok(model)
    }

    pub fn grid_shape(&self, width: u32, height: u32) -> (u32, u32) {
        let patch = self.spec.patch;
        match (self.spec.naflex, self.spec.image_size) {
            (true, _) => {
                let (h, w) = naflex_size(height, width, patch, self.spec.max_patches.unwrap_or(1024));
                (h / patch, w / patch)
            }
            (false, Some(size)) => (size / patch, size / patch),
            _ => (0, 0),
        }
    }

    pub fn embed_grid(&self, rgb: &Rgb) -> Result<Grid> {
        let patch = self.spec.patch;
        let outputs = if self.spec.naflex {
            let (h, w) = naflex_size(rgb.height, rgb.width, patch, self.spec.max_patches.unwrap_or(1024));
            let (rows, cols) = (h / patch, w / patch);
            let resized = resize_rgb(rgb, w, h);
            let patches = patchify(&resized, patch);
            let side = self.spec.pos_grid_side.context("NaFlex member without pos_grid_side")?;
            let pos = resize_pos(self.pos_grid.as_ref().context("NaFlex member without pos grid")?, side, self.spec.dim, rows as usize, cols as usize);
            let n = (rows * cols) as usize;
            let (shape, data) = self.vision(n)?.run(|session| {
                let patch_values = Tensor::from_array(([1usize, n, (patch * patch * 3) as usize], patches.clone()))?;
                let pos_values = Tensor::from_array(([1usize, n, self.spec.dim], pos.clone()))?;
                let out = session.run(ort::inputs!["patches" => patch_values, "pos" => pos_values])?;
                let (shape, data) = out["embeddings"].try_extract_tensor::<f32>()?;
                Ok((shape.to_vec(), data.to_vec()))
            })?;
            (rows, cols, shape, data)
        } else {
            let size = self.spec.image_size.context("fixed member without image_size")?;
            let resized = resize_rgb(rgb, size, size);
            let pixels = planar(&resized);
            let (shape, data) = self.vision(0)?.run(|session| {
                let input = Tensor::from_array(([1usize, 3, size as usize, size as usize], pixels.clone()))?;
                let out = session.run(ort::inputs!["pixel_values" => input])?;
                let (shape, data) = out["embeddings"].try_extract_tensor::<f32>()?;
                Ok((shape.to_vec(), data.to_vec()))
            })?;
            (size / patch, size / patch, shape, data)
        };
        let (rows, cols, shape, data) = outputs;
        let dim = *shape.last().context("empty output")? as usize;
        if data.len() != (rows * cols) as usize * dim {
            bail!("{}: expected {}x{} cells, got shape {:?}", self.key, rows, cols, shape);
        }
        Ok(Grid { rows, cols, dim, data })
    }

    pub fn tokens(&self, texts: &[&str]) -> Result<Vec<i64>> {
        let length = self.spec.text_length;
        let mut ids = Vec::with_capacity(texts.len() * length);
        for text in texts {
            let encoding = self.tokenizer.encode(*text, true).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
            let mut row: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).take(length).collect();
            row.resize(length, 0);
            ids.extend(row);
        }
        Ok(ids)
    }

    /// L2-normalized text embeddings, one per text.
    pub fn embed_text(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let ids = self.tokens(texts)?;
        let mut guard = self.text.lock().unwrap();
        if guard.is_none() {
            // texts come in batches of any size, so the text tower stays on the CPU (~0.1 s a question)
            *guard = Some(onnx::Model::load_on(&self.text_path, &[], onnx::Device::Cpu)?);
        }
        guard.as_ref().unwrap().run(|session| {
            let input = Tensor::from_array(([texts.len(), self.spec.text_length], ids.clone()))?;
            let out = session.run(ort::inputs!["input_ids" => input])?;
            let (shape, data) = out["embedding"].try_extract_tensor::<f32>()?;
            let dim = *shape.last().context("empty text output")? as usize;
            Ok(data.chunks_exact(dim).map(|row| row.to_vec()).collect())
        })
    }
}

/// transformers' `get_image_size_for_max_num_patches`: the largest (height, width), each a multiple of `patch`, at one
/// common scale of the frame, whose patch count fits `budget`.
pub fn naflex_size(height: u32, width: u32, patch: u32, budget: u32) -> (u32, u32) {
    let scaled = |scale: f64, size: u32| -> u32 {
        let size = (size as f64 * scale / patch as f64).ceil() * patch as f64;
        (size as u32).max(patch)
    };
    let eps = 1e-5;
    let (mut low, mut high) = (eps / 10.0, 100.0f64);
    while high - low >= eps {
        let scale = (low + high) / 2.0;
        let count = (scaled(scale, height) / patch) as f64 * (scaled(scale, width) / patch) as f64;
        if count <= budget as f64 {
            low = scale;
        } else {
            high = scale;
        }
    }
    (scaled(low, height), scaled(low, width))
}

/// PIL-style bilinear (a triangle filter widened when shrinking), what the processors' `resample=2` does.
pub fn resize_rgb(rgb: &Rgb, width: u32, height: u32) -> Rgb {
    if rgb.width == width && rgb.height == height {
        return rgb.clone();
    }
    let source = image::RgbImage::from_raw(rgb.width, rgb.height, rgb.data.clone()).expect("rgb buffer size");
    let resized = image::imageops::resize(&source, width, height, image::imageops::FilterType::Triangle);
    Rgb { width, height, data: resized.into_raw() }
}

fn normalized(value: u8) -> f32 {
    (value as f32 / 255.0 - 0.5) / 0.5
}

/// HWC u8 -> CHW float, (x/255 - 0.5) / 0.5.
fn planar(rgb: &Rgb) -> Vec<f32> {
    let n = (rgb.width * rgb.height) as usize;
    let mut out = vec![0.0f32; 3 * n];
    for (i, pixel) in rgb.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * n + i] = normalized(pixel[c]);
        }
    }
    out
}

/// transformers' `convert_image_to_patches`: (rows*cols, patch*patch*3), each patch's pixels row by row, channels last.
pub fn patchify(rgb: &Rgb, patch: u32) -> Vec<f32> {
    let (rows, cols) = (rgb.height / patch, rgb.width / patch);
    let p = patch as usize;
    let mut out = Vec::with_capacity((rows * cols) as usize * p * p * 3);
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            for py in 0..p {
                let y = r * p + py;
                let start = (y * rgb.width as usize + c * p) * 3;
                out.extend(rgb.data[start..start + p * 3].iter().map(|&v| normalized(v)));
            }
        }
    }
    out
}

/// One axis of PyTorch's antialiased bilinear resize (`F.interpolate(..., mode="bilinear", align_corners=False,
/// antialias=True)`): per output index, (first input index, normalized weights).
fn aa_weights(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor() as usize).min(input);
            let mut weights: Vec<f64> = (xmin..xmax).map(|j| (1.0 - ((j as f64 - center + 0.5) * invscale).abs()).max(0.0)).collect();
            let total: f64 = weights.iter().sum();
            if total > 0.0 {
                weights.iter_mut().for_each(|w| *w /= total);
            }
            (xmin, weights.into_iter().map(|w| w as f32).collect())
        })
        .collect()
}

/// The (side, side, dim) position grid resized to (rows, cols): (rows*cols, dim), row-major, as the NaFlex graph's
/// `pos` input wants it.
pub fn resize_pos(grid: &[f32], side: usize, dim: usize, rows: usize, cols: usize) -> Vec<f32> {
    let along_y = aa_weights(side, rows);
    let along_x = aa_weights(side, cols);
    // first along x: (side, cols, dim)
    let mut wide = vec![0.0f32; side * cols * dim];
    for y in 0..side {
        for (x, (start, weights)) in along_x.iter().enumerate() {
            let out = &mut wide[(y * cols + x) * dim..(y * cols + x + 1) * dim];
            for (k, w) in weights.iter().enumerate() {
                let source = &grid[(y * side + start + k) * dim..(y * side + start + k + 1) * dim];
                out.iter_mut().zip(source).for_each(|(o, s)| *o += w * s);
            }
        }
    }
    let mut result = vec![0.0f32; rows * cols * dim];
    for (y, (start, weights)) in along_y.iter().enumerate() {
        for x in 0..cols {
            let out = &mut result[(y * cols + x) * dim..(y * cols + x + 1) * dim];
            for (k, w) in weights.iter().enumerate() {
                let source = &wide[((start + k) * cols + x) * dim..((start + k) * cols + x + 1) * dim];
                out.iter_mut().zip(source).for_each(|(o, s)| *o += w * s);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naflex_size_matches_transformers() {
        // the RealSense frame at a 1024-patch budget is 24x42 patches (transformers, checked in export_models.py)
        assert_eq!(naflex_size(480, 848, 16, 1024), (384, 672));
        assert_eq!(naflex_size(480, 848, 16, 256), (192, 336));
        let (h, w) = naflex_size(720, 1280, 16, 1024);
        assert!((h / 16) * (w / 16) <= 1024);
    }

    #[test]
    fn upsampling_weights_are_plain_bilinear() {
        // 2 -> 4: output 0 sits left of input 0's center (all of input 0), output 1 is 3/4 input 0 + 1/4 input 1
        let weights = aa_weights(2, 4);
        assert_eq!(weights[0], (0, vec![1.0]));
        assert_eq!(weights[1].0, 0);
        assert!((weights[1].1[0] - 0.75).abs() < 1e-6 && (weights[1].1[1] - 0.25).abs() < 1e-6);
        // shrinking 4 -> 2 widens the triangle to 2 inputs each side (PyTorch: 0.75, 0.75, 0.25, normalized)
        let shrink = aa_weights(4, 2);
        assert_eq!(shrink[0].0, 0);
        let expected = [0.75 / 1.75, 0.75 / 1.75, 0.25 / 1.75];
        assert!(shrink[0].1.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-6), "{:?}", shrink[0]);
    }

    #[test]
    fn a_constant_grid_resizes_to_itself() {
        let grid = vec![2.0f32; 16 * 16 * 3];
        let out = resize_pos(&grid, 16, 3, 24, 42);
        assert_eq!(out.len(), 24 * 42 * 3);
        assert!(out.iter().all(|v| (v - 2.0).abs() < 1e-5));
    }

    #[test]
    fn patches_are_rows_of_pixels_channels_last() {
        // a 2x4 image, patch 2: two patches; the second starts at column 2
        let data: Vec<u8> = (0..24).map(|v| v as u8 * 10).collect();
        let rgb = Rgb { width: 4, height: 2, data };
        let patches = patchify(&rgb, 2);
        assert_eq!(patches.len(), 2 * 12);
        assert_eq!(patches[0], normalized(0));
        // second patch, first pixel = column 2 of row 0 = byte 6
        assert_eq!(patches[12], normalized(60));
        // first patch, second row = row 1 column 0 = byte 12
        assert_eq!(patches[6], normalized(120));
    }
}
