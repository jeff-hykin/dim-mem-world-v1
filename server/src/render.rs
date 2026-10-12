//! What the page draws from a memory: its point cloud (every keyframe's depth, colored by its JPEG, thinned to a
//! 5 cm grid) and keyframe images with an answer's box drawn on them.
use anyhow::Result;
use memworld::memory::Memory;
use memworld::owl::decode_jpeg;
use std::collections::HashMap;

const VOXEL_M: f64 = 0.05;
const MAX_POINTS: usize = 1_500_000;

/// `u32 count`, `count` x `f32 x, y, z`, `count` x `u8 r, g, b`, little endian.
pub fn cloud(memory: &Memory) -> Result<Vec<u8>> {
    let mut cells: HashMap<(i32, i32, i32), ([f32; 3], [u8; 3])> = HashMap::new();
    for frame in &memory.frames {
        let Some(depth) = frame.depth_mm() else { continue };
        let Ok(rgb) = decode_jpeg(&frame.jpeg) else { continue };
        let camera = &frame.camera;
        for v in (0..camera.height).step_by(3) {
            for u in (0..camera.width).step_by(3) {
                let mm = depth[(v * camera.width + u) as usize];
                // past 5 m the RealSense's depth is too noisy to draw
                if mm == 0 || mm > 5000 {
                    continue;
                }
                let p = frame.pose * camera.unproject(u as f64 + 0.5, v as f64 + 0.5, mm as f64 / 1000.0);
                let key = ((p.x / VOXEL_M).floor() as i32, (p.y / VOXEL_M).floor() as i32, (p.z / VOXEL_M).floor() as i32);
                let at = ((v * rgb.width + u) * 3) as usize;
                cells.entry(key).or_insert(([p.x as f32, p.y as f32, p.z as f32], [rgb.data[at], rgb.data[at + 1], rgb.data[at + 2]]));
            }
        }
        if cells.len() > MAX_POINTS {
            break;
        }
    }
    let count = cells.len();
    let mut out = Vec::with_capacity(4 + count * 15);
    out.extend_from_slice(&(count as u32).to_le_bytes());
    let points: Vec<_> = cells.into_values().collect();
    for (p, _) in &points {
        for value in p {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    for (_, color) in &points {
        out.extend_from_slice(color);
    }
    Ok(out)
}

/// A stored keyframe as JPEG, with `bbox` (pixels) outlined when given.
pub fn boxed_jpeg(jpeg: &[u8], bbox: Option<[f32; 4]>) -> Result<Vec<u8>> {
    let Some(bbox) = bbox else { return Ok(jpeg.to_vec()) };
    let rgb = decode_jpeg(jpeg)?;
    let mut image = image::RgbImage::from_raw(rgb.width, rgb.height, rgb.data).expect("decoded size");
    let (w, h) = (image.width() as i64, image.height() as i64);
    let color = image::Rgb([255, 140, 0]);
    let thickness = 3;
    let (x0, y0, x1, y1) = (bbox[0] as i64, bbox[1] as i64, bbox[2] as i64, bbox[3] as i64);
    for t in 0..thickness {
        for x in (x0 - t).max(0)..=(x1 + t).min(w - 1) {
            for y in [y0 - t, y1 + t] {
                if (0..h).contains(&y) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
        for y in (y0 - t).max(0)..=(y1 + t).min(h - 1) {
            for x in [x0 - t, x1 + t] {
                if (0..w).contains(&x) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 88).encode(image.as_raw(), image.width(), image.height(), image::ExtendedColorType::Rgb8)?;
    Ok(out)
}
