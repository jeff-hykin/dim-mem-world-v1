//! OWLv2 (google/owlv2-base-patch16-ensemble), split in two (tools/export_models.py):
//!
//! - the image half, run ONCE per keyframe at generation and stored in the recording: for each of its 60x60 patches a
//!   box, a normalized class embedding (512), and a logit shift and scale;
//! - the text half, run per question: one normalized embedding.
//!
//! A box's score for a question is then `sigmoid((class . text + shift) * scale)`: a dot product per patch, so a
//! question can score every keyframe in milliseconds instead of re-running the image tower (4 s a frame on a CPU).
//!
//! Preprocessing as Owlv2ImageProcessor: pad the frame to a square at the bottom/right with gray, resize to 960,
//! CLIP mean/std. Boxes are (cx, cy, w, h) in [0, 1] of the padded square.
use crate::blob;
use crate::onnx;
use anyhow::{bail, Context, Result};
use dimos_recording::image::Rgb;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

pub const SIZE: u32 = 960;
pub const PATCHES: usize = 3600;
pub const DIM: usize = 512;
pub const TEXT_LENGTH: usize = 16;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Detection {
    /// [x0, y0, x1, y1] in the frame's pixels
    pub bbox: [f32; 4],
    pub score: f32,
}

/// One frame's question-independent OWLv2 output.
#[derive(Debug, Clone)]
pub struct Features {
    /// PATCHES x DIM, each row unit length
    pub class_embeds: Vec<f32>,
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
    /// PATCHES x (cx, cy, w, h) in [0, 1] of the padded square
    pub boxes: Vec<f32>,
}

impl Features {
    /// fp16, as stored in the recording: class embeds, shift, scale, boxes.
    pub fn encode(&self) -> Vec<u8> {
        let mut all = Vec::with_capacity(PATCHES * (DIM + 6));
        all.extend_from_slice(&self.class_embeds);
        all.extend_from_slice(&self.shift);
        all.extend_from_slice(&self.scale);
        all.extend_from_slice(&self.boxes);
        blob::f16_bytes(&all)
    }

    pub fn decode(bytes: &[u8]) -> Result<Features> {
        let all = blob::f16_values(bytes);
        if all.len() != PATCHES * (DIM + 6) {
            bail!("OWLv2 features of the wrong size ({} values)", all.len());
        }
        let (class_embeds, rest) = all.split_at(PATCHES * DIM);
        let (shift, rest) = rest.split_at(PATCHES);
        let (scale, boxes) = rest.split_at(PATCHES);
        Ok(Features { class_embeds: class_embeds.to_vec(), shift: shift.to_vec(), scale: scale.to_vec(), boxes: boxes.to_vec() })
    }

    /// Every box scoring at least `threshold` for the question embedding, in pixels of a `width` x `height` frame
    /// (the frame the features came from, at any scale), best first, overlaps suppressed.
    pub fn detect(&self, query: &[f32], width: u32, height: u32, threshold: f32) -> Vec<Detection> {
        let side = width.max(height) as f32;
        let mut found: Vec<Detection> = (0..PATCHES)
            .filter_map(|i| {
                let similarity = crate::memory::dot(&self.class_embeds[i * DIM..(i + 1) * DIM], query);
                let logit = (similarity + self.shift[i]) * self.scale[i];
                let score = 1.0 / (1.0 + (-logit).exp());
                if score < threshold {
                    return None;
                }
                let b = &self.boxes[i * 4..i * 4 + 4];
                let (cx, cy, w, h) = (b[0] * side, b[1] * side, b[2] * side, b[3] * side);
                let bbox = [(cx - w / 2.0).max(0.0), (cy - h / 2.0).max(0.0), (cx + w / 2.0).min(width as f32), (cy + h / 2.0).min(height as f32)];
                (bbox[2] > bbox[0] && bbox[3] > bbox[1]).then_some(Detection { bbox, score })
            })
            .collect();
        found.sort_by(|a, b| b.score.total_cmp(&a.score));
        suppress(found, 0.3)
    }
}

pub struct Owl {
    /// on a Mac, the native CoreML model (dropped for ONNX Runtime on the CPU if it ever fails)
    #[cfg(target_os = "macos")]
    native: Mutex<Option<std::sync::Arc<crate::coreml::Native>>>,
    /// ONNX Runtime's session, built on first use
    vision: Mutex<Option<std::sync::Arc<onnx::Model>>>,
    vision_path: std::path::PathBuf,
    text: Mutex<Option<onnx::Model>>,
    text_path: std::path::PathBuf,
    tokenizer: tokenizers::Tokenizer,
}

impl Owl {
    pub fn load(dir: &Path) -> Result<Owl> {
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("owlv2.tokenizer.json")).map_err(|e| anyhow::anyhow!("owlv2 tokenizer: {e}"))?;
        Ok(Owl {
            #[cfg(target_os = "macos")]
            native: Mutex::new(crate::coreml::load_if_wanted(dir, "owlv2.vision")),
            vision: Mutex::new(None),
            vision_path: dir.join("owlv2.vision.onnx"),
            text: Mutex::new(None),
            text_path: dir.join("owlv2.text.onnx"),
            tokenizer,
        })
    }

    pub fn device(&self) -> onnx::Device {
        #[cfg(target_os = "macos")]
        if self.native.lock().unwrap().is_some() {
            return onnx::Device::Coreml;
        }
        self.vision.lock().unwrap().as_ref().map(|model| model.device()).unwrap_or_else(onnx::wanted_device)
    }

    fn session(&self) -> Result<std::sync::Arc<onnx::Model>> {
        let mut guard = self.vision.lock().unwrap();
        if guard.is_none() {
            *guard = Some(std::sync::Arc::new(onnx::Model::load(&self.vision_path, &[])?));
        }
        Ok(guard.as_ref().unwrap().clone())
    }

    /// The image half for one frame.
    pub fn features(&self, rgb: &Rgb) -> Result<Features> {
        let pixels = preprocess(rgb);
        let check = |name: &str, data: Vec<f32>, len: usize| -> Result<Vec<f32>> {
            if data.len() != len {
                bail!("OWLv2 {name}: {} values, expected {len}", data.len());
            }
            Ok(data)
        };
        #[cfg(target_os = "macos")]
        {
            let native = self.native.lock().unwrap().clone();
            if let Some(native) = native {
                match native.run(&[("pixel_values", &[1, 3, SIZE as usize, SIZE as usize], &pixels)], &["class_embeds", "logit_shift", "logit_scale", "boxes"]) {
                    Ok(outputs) => {
                        let [class_embeds, shift, scale, boxes]: [Vec<f32>; 4] = outputs.try_into().map_err(|_| anyhow::anyhow!("OWLv2: wrong output count"))?;
                        return Ok(Features { class_embeds: check("class_embeds", class_embeds, PATCHES * DIM)?, shift: check("logit_shift", shift, PATCHES)?, scale: check("logit_scale", scale, PATCHES)?, boxes: check("boxes", boxes, PATCHES * 4)? });
                    }
                    Err(error) => {
                        eprintln!("[memworld] OWLv2 failed on CoreML, moving it to the CPU: {error:#}");
                        *self.native.lock().unwrap() = None;
                    }
                }
            }
        }
        self.session()?.run(|session| {
            let out = session.run(ort::inputs!["pixel_values" => Tensor::from_array(([1usize, 3, SIZE as usize, SIZE as usize], pixels.clone()))?])?;
            let take = |name: &str, len: usize| -> Result<Vec<f32>> { check(name, out[name].try_extract_tensor::<f32>()?.1.to_vec(), len) };
            Ok(Features { class_embeds: take("class_embeds", PATCHES * DIM)?, shift: take("logit_shift", PATCHES)?, scale: take("logit_scale", PATCHES)?, boxes: take("boxes", PATCHES * 4)? })
        })
    }

    pub fn tokens(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).take(TEXT_LENGTH).collect();
        let mut mask = vec![1i64; ids.len()];
        ids.resize(TEXT_LENGTH, 0);
        mask.resize(TEXT_LENGTH, 0);
        Ok((ids, mask))
    }

    /// The text half: a unit-length embedding of the question.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let (ids, mask) = self.tokens(text)?;
        let mut guard = self.text.lock().unwrap();
        if guard.is_none() {
            *guard = Some(onnx::Model::load_on(&self.text_path, &[("batch", 1)], onnx::Device::Cpu)?);
        }
        guard.as_ref().unwrap().run(|session| {
            let out = session.run(ort::inputs![
                "input_ids" => Tensor::from_array(([1usize, TEXT_LENGTH], ids.clone()))?,
                "attention_mask" => Tensor::from_array(([1usize, TEXT_LENGTH], mask.clone()))?,
            ])?;
            let (_, data) = out["embedding"].try_extract_tensor::<f32>()?;
            Ok(data.to_vec())
        })
    }

    /// Both halves on one frame (what a frame without stored features costs).
    pub fn detect(&self, rgb: &Rgb, text: &str, threshold: f32) -> Result<Vec<Detection>> {
        let query = self.embed_text(text)?;
        Ok(self.features(rgb)?.detect(&query, rgb.width, rgb.height, threshold))
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
        assert!(pixels[0] > 1.5);
        let gray = (128.0 / 255.0 - MEAN[0]) / STD[0];
        assert!((pixels[n - SIZE as usize] - gray).abs() < 0.05);
    }

    #[test]
    fn stored_features_score_like_the_head_and_scale_to_any_frame_size() {
        // one patch matches the question (class embed = the question), the rest point elsewhere
        let mut features = Features { class_embeds: vec![0.0; PATCHES * DIM], shift: vec![0.0; PATCHES], scale: vec![1.0; PATCHES], boxes: vec![0.5; PATCHES * 4] };
        for i in 0..PATCHES {
            features.class_embeds[i * DIM + 1] = 1.0;
        }
        features.class_embeds[7 * DIM + 1] = 0.0;
        features.class_embeds[7 * DIM] = 1.0;
        features.scale[7] = 10.0;
        features.boxes[7 * 4..7 * 4 + 4].copy_from_slice(&[0.25, 0.25, 0.1, 0.1]);
        let mut query = vec![0.0; DIM];
        query[0] = 1.0;
        let back = Features::decode(&features.encode()).unwrap();
        // 848 x 480: the padded square's side is 848
        let found = back.detect(&query, 848, 480, 0.9);
        assert_eq!(found.len(), 1);
        assert!((found[0].score - 1.0 / (1.0 + (-10.0f32).exp())).abs() < 1e-4);
        let expected = [0.2 * 848.0, 0.2 * 848.0, 0.3 * 848.0, 0.3 * 848.0];
        assert!(found[0].bbox.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1.0), "{:?}", found[0]);
        // the same features on the half-size JPEG give half-size boxes
        let half = back.detect(&query, 424, 240, 0.9);
        assert!((half[0].bbox[0] - expected[0] / 2.0).abs() < 1.0);
    }
}
