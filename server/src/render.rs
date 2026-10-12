//! What the page draws from a memory: the lidar's voxel map, colored by height like the Map Editor, and keyframe images
//! with an answer's box drawn on them.
use anyhow::Result;
use memworld::memory::Memory;
use memworld::owl::decode_jpeg;

/// `u32 count`, `count` x `f32 x, y, z`, `count` x `u8 r, g, b`, little endian.
pub fn cloud(memory: &Memory) -> Result<Vec<u8>> {
    let points = memory.map.clone().unwrap_or_default();
    let colors = memworld::lidar::height_colors(&points);
    let mut out = Vec::with_capacity(4 + points.len() * 15);
    out.extend_from_slice(&(points.len() as u32).to_le_bytes());
    for p in &points {
        for value in p {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    for color in &colors {
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
