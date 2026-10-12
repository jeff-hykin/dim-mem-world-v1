//! OWLv2 (onnx-community/owlv2-base-patch16-ensemble-ONNX): boxes for a text query in one frame. It runs only on the
//! few frames the three SigLIP2 members agree on, and its boxes are what get placed in 3D.
//!
//! Preprocessing as Owlv2ImageProcessor: pad the frame to a square at the bottom/right with gray, resize to 960,
//! CLIP mean/std. Boxes come back as (cx, cy, w, h) in [0, 1] of the padded square.
use crate::onnx;
use anyhow::{Context, Result};
use dimos_recording::image::Rgb;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

pub const SIZE: u32 = 960;
pub const TEXT_LENGTH: usize = 16;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Detection {
    /// [x0, y0, x1, y1] in the frame's pixels
    pub bbox: [f32; 4],
    pub score: f32,
}

pub struct Owl {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
}

impl Owl {
    pub fn load(dir: &Path) -> Result<Owl> {
        let session = onnx::session(&dir.join("owlv2.onnx"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("owlv2.tokenizer.json")).map_err(|e| anyhow::anyhow!("owlv2 tokenizer: {e}"))?;
        Ok(Owl { session: Mutex::new(session), tokenizer })
    }

    fn tokens(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).take(TEXT_LENGTH).collect();
        let mut mask = vec![1i64; ids.len()];
        ids.resize(TEXT_LENGTH, 0);
        mask.resize(TEXT_LENGTH, 0);
        Ok((ids, mask))
    }

    /// Every box scoring at least `threshold` for `text`, best first.
    pub fn detect(&self, rgb: &Rgb, text: &str, threshold: f32) -> Result<Vec<Detection>> {
        let (ids, mask) = self.tokens(text)?;
        let pixels = preprocess(rgb);
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, TEXT_LENGTH], ids))?,
            "pixel_values" => Tensor::from_array(([1usize, 3, SIZE as usize, SIZE as usize], pixels))?,
            "attention_mask" => Tensor::from_array(([1usize, TEXT_LENGTH], mask))?,
        ])?;
        let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (_, boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>()?;
        let side = rgb.width.max(rgb.height) as f32;
        let mut found: Vec<Detection> = logits
            .iter()
            .enumerate()
            .filter_map(|(i, &logit)| {
                let score = 1.0 / (1.0 + (-logit).exp());
                if score < threshold {
                    return None;
                }
                let b = &boxes[i * 4..i * 4 + 4];
                let (cx, cy, w, h) = (b[0] * side, b[1] * side, b[2] * side, b[3] * side);
                let bbox = [(cx - w / 2.0).max(0.0), (cy - h / 2.0).max(0.0), (cx + w / 2.0).min(rgb.width as f32), (cy + h / 2.0).min(rgb.height as f32)];
                (bbox[2] > bbox[0] && bbox[3] > bbox[1]).then_some(Detection { bbox, score })
            })
            .collect();
        found.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(suppress(found, 0.3))
    }
}

/// Pad to a square (gray, bottom/right), resize to 960, CLIP-normalize, CHW.
pub fn preprocess(rgb: &Rgb) -> Vec<f32> {
    let side = rgb.width.max(rgb.height);
    let mut square = image::RgbImage::from_pixel(side, side, image::Rgb([128, 128, 128]));
    let source = image::RgbImage::from_raw(rgb.width, rgb.height, rgb.data.clone()).expect("rgb buffer size");
    image::imageops::replace(&mut square, &source, 0, 0);
    let resized = image::imageops::resize(&square, SIZE, SIZE, image::imageops::FilterType::Triangle);
    let n = (SIZE * SIZE) as usize;
    let mut out = vec![0.0f32; 3 * n];
    for (i, pixel) in resized.pixels().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Greedy non-maximum suppression over boxes already sorted best first.
pub fn suppress(sorted: Vec<Detection>, overlap: f32) -> Vec<Detection> {
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in sorted {
        if kept.iter().all(|k| iou(&k.bbox, &candidate.bbox) < overlap) {
            kept.push(candidate);
        }
    }
    kept
}

/// Whether a directory holds what `Owl::load` reads.
pub fn present(dir: &Path) -> bool {
    dir.join("owlv2.onnx").exists() && dir.join("owlv2.tokenizer.json").exists()
}

pub fn decode_jpeg(bytes: &[u8]) -> Result<Rgb> {
    let decoded = image::load_from_memory(bytes).context("decoding a stored frame")?.to_rgb8();
    Ok(Rgb { width: decoded.width(), height: decoded.height(), data: decoded.into_raw() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_boxes_keep_the_best() {
        let a = Detection { bbox: [0.0, 0.0, 10.0, 10.0], score: 0.9 };
        let b = Detection { bbox: [1.0, 1.0, 11.0, 11.0], score: 0.8 };
        let c = Detection { bbox: [50.0, 50.0, 60.0, 60.0], score: 0.7 };
        assert_eq!(suppress(vec![a, b, c], 0.3), vec![a, c]);
        assert!((iou(&a.bbox, &a.bbox) - 1.0).abs() < 1e-6);
        assert_eq!(iou(&a.bbox, &c.bbox), 0.0);
    }

    #[test]
    fn a_wide_frame_pads_below_with_gray() {
        let rgb = Rgb { width: 4, height: 2, data: vec![255; 4 * 2 * 3] };
        let pixels = preprocess(&rgb);
        let n = (SIZE * SIZE) as usize;
        // top-left is the white frame, bottom-left is the gray pad
        assert!(pixels[0] > 1.5);
        let gray = (128.0 / 255.0 - MEAN[0]) / STD[0];
        assert!((pixels[n - SIZE as usize] - gray).abs() < 0.05);
    }
}
