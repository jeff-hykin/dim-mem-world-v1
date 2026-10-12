//! Camera messages: sensor_msgs Image / CompressedImage / CameraInfo, from dimos's LCM (`.db` rows and `lcm` mcap
//! channels) or ROS 2 CDR (`cdr` mcap channels), turned into what a vision model and a back-projection need: an RGB8
//! frame, a depth frame in meters, and pinhole intrinsics.
use crate::cdr::Cdr;
use crate::lcm::{read_header, Header, Reader};
use anyhow::{bail, Context, Result};

pub const IMAGE_SCHEMA: &str = "sensor_msgs/msg/Image";
pub const COMPRESSED_IMAGE_SCHEMA: &str = "sensor_msgs/msg/CompressedImage";
pub const CAMERA_INFO_SCHEMA: &str = "sensor_msgs/msg/CameraInfo";

/// An image as stored: raw pixels with an `encoding` (rgb8, bgr8, mono16, 16UC1, 32FC1, ...) or a compressed `format`
/// (jpeg, png).
#[derive(Debug, Clone)]
pub struct StoredImage {
    pub header: Header,
    pub width: u32,
    pub height: u32,
    /// raw: the ROS encoding; compressed: "jpeg" / "png" (whatever the message's format string starts with)
    pub encoding: String,
    pub compressed: bool,
    pub big_endian: bool,
    pub step: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CameraInfo {
    pub header: Header,
    pub width: u32,
    pub height: u32,
    /// row-major 3x3: fx 0 cx / 0 fy cy / 0 0 1
    pub k: [f64; 9],
}

impl CameraInfo {
    pub fn fx(&self) -> f64 {
        self.k[0]
    }
    pub fn fy(&self) -> f64 {
        self.k[4]
    }
    pub fn cx(&self) -> f64 {
        self.k[2]
    }
    pub fn cy(&self) -> f64 {
        self.k[5]
    }
    /// The same camera at another resolution (intrinsics scale with the image).
    pub fn scaled(&self, width: u32, height: u32) -> CameraInfo {
        let sx = width as f64 / self.width.max(1) as f64;
        let sy = height as f64 / self.height.max(1) as f64;
        let mut k = self.k;
        k[0] *= sx;
        k[2] *= sx;
        k[4] *= sy;
        k[5] *= sy;
        CameraInfo { header: self.header.clone(), width, height, k }
    }
}

/// An 8-bit RGB frame, row-major.
#[derive(Debug, Clone)]
pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Depth in meters, row-major; 0 = no reading.
#[derive(Debug, Clone)]
pub struct Depth {
    pub width: u32,
    pub height: u32,
    pub meters: Vec<f32>,
}

impl Depth {
    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.meters[(y * self.width + x) as usize]
    }
}

// LCM: array lengths are hoisted to the front of the struct (after the fingerprint).

pub fn decode_lcm_image(payload: &[u8]) -> Result<StoredImage> {
    let mut reader = Reader::new(payload);
    reader.take(8)?;
    let length = reader.length()?;
    let header = read_header(&mut reader)?;
    let height = reader.i32()? as u32;
    let width = reader.i32()? as u32;
    let encoding = reader.string()?;
    let big_endian = reader.u8()? != 0;
    let step = reader.i32()? as u32;
    let data = reader.take(length)?.to_vec();
    Ok(StoredImage { header, width, height, encoding, compressed: false, big_endian, step, data })
}

pub fn decode_lcm_compressed_image(payload: &[u8]) -> Result<StoredImage> {
    let mut reader = Reader::new(payload);
    reader.take(8)?;
    let length = reader.length()?;
    let header = read_header(&mut reader)?;
    let format = reader.string()?;
    let data = reader.take(length)?.to_vec();
    compressed(header, &format, data)
}

pub fn decode_lcm_camera_info(payload: &[u8]) -> Result<CameraInfo> {
    let mut reader = Reader::new(payload);
    reader.take(8)?;
    let d_length = reader.length()?;
    let header = read_header(&mut reader)?;
    let height = reader.i32()? as u32;
    let width = reader.i32()? as u32;
    let _model = reader.string()?;
    for _ in 0..d_length {
        reader.f64()?;
    }
    let k = reader.f64_array::<9>()?;
    Ok(CameraInfo { header, width, height, k })
}

pub fn decode_cdr_image(payload: &[u8]) -> Result<StoredImage> {
    let mut cdr = Cdr::new(payload)?;
    let header = cdr.header()?;
    let height = cdr.u32()?;
    let width = cdr.u32()?;
    let encoding = cdr.string()?;
    let big_endian = cdr.u8()? != 0;
    let step = cdr.u32()?;
    let length = cdr.u32()? as usize;
    let data = cdr.take(length)?.to_vec();
    Ok(StoredImage { header, width, height, encoding, compressed: false, big_endian, step, data })
}

pub fn decode_cdr_compressed_image(payload: &[u8]) -> Result<StoredImage> {
    let mut cdr = Cdr::new(payload)?;
    let header = cdr.header()?;
    let format = cdr.string()?;
    let length = cdr.u32()? as usize;
    let data = cdr.take(length)?.to_vec();
    compressed(header, &format, data)
}

pub fn decode_cdr_camera_info(payload: &[u8]) -> Result<CameraInfo> {
    let mut cdr = Cdr::new(payload)?;
    let header = cdr.header()?;
    let height = cdr.u32()?;
    let width = cdr.u32()?;
    let _model = cdr.string()?;
    let d_length = cdr.u32()? as usize;
    for _ in 0..d_length {
        cdr.f64()?;
    }
    let k = cdr.f64s::<9>()?;
    Ok(CameraInfo { header, width, height, k })
}

/// A compressed image's size comes from its own header (jpeg SOF / png IHDR), read without decoding it.
fn compressed(header: Header, format: &str, data: Vec<u8>) -> Result<StoredImage> {
    let lower = format.to_ascii_lowercase();
    let encoding = if lower.contains("png") { "png" } else if lower.contains("jpeg") || lower.contains("jpg") { "jpeg" } else { bail!("unsupported compressed format {format:?}") };
    let (width, height) = image::ImageReader::new(std::io::Cursor::new(&data)).with_guessed_format()?.into_dimensions().context("reading a compressed image's size")?;
    // compressedDepth (ROS image_transport) prefixes the png with a 12-byte config: not supported here
    Ok(StoredImage { header, width, height, encoding: encoding.into(), compressed: true, big_endian: false, step: 0, data })
}

pub fn decode_image(encoding: &str, schema: &str, payload: &[u8]) -> Result<StoredImage> {
    let compressed = schema.contains("CompressedImage");
    match (encoding == "cdr", compressed) {
        (true, true) => decode_cdr_compressed_image(payload),
        (true, false) => decode_cdr_image(payload),
        (false, true) => decode_lcm_compressed_image(payload),
        (false, false) => decode_lcm_image(payload),
    }
}

pub fn decode_camera_info(encoding: &str, payload: &[u8]) -> Result<CameraInfo> {
    if encoding == "cdr" {
        decode_cdr_camera_info(payload)
    } else {
        decode_lcm_camera_info(payload)
    }
}

impl StoredImage {
    /// Whether this holds depth rather than color: 16-bit or float single-channel, or a 16-bit png.
    pub fn is_depth(&self) -> bool {
        matches!(self.encoding.as_str(), "16UC1" | "mono16" | "32FC1" | "DEPTH16" | "Z16")
    }

    pub fn to_rgb(&self) -> Result<Rgb> {
        let (w, h) = (self.width, self.height);
        if self.compressed {
            let decoded = image::load_from_memory(&self.data).context("decoding a compressed image")?.to_rgb8();
            return Ok(Rgb { width: decoded.width(), height: decoded.height(), data: decoded.into_raw() });
        }
        let pixels = (w * h) as usize;
        let step = if self.step == 0 { w as usize * channels(&self.encoding) } else { self.step as usize };
        let row = |y: usize| &self.data[y * step..];
        let mut data = Vec::with_capacity(pixels * 3);
        let wide = w as usize;
        match self.encoding.to_ascii_lowercase().as_str() {
            "rgb8" | "rgb" => (0..h as usize).for_each(|y| data.extend_from_slice(&row(y)[..wide * 3])),
            "bgr8" | "bgr" => (0..h as usize).for_each(|y| row(y)[..wide * 3].chunks_exact(3).for_each(|p| data.extend_from_slice(&[p[2], p[1], p[0]]))),
            "rgba8" | "rgba" => (0..h as usize).for_each(|y| row(y)[..wide * 4].chunks_exact(4).for_each(|p| data.extend_from_slice(&p[..3]))),
            "bgra8" | "bgra" => (0..h as usize).for_each(|y| row(y)[..wide * 4].chunks_exact(4).for_each(|p| data.extend_from_slice(&[p[2], p[1], p[0]]))),
            "mono8" | "8uc1" | "gray" => (0..h as usize).for_each(|y| row(y)[..wide].iter().for_each(|&v| data.extend_from_slice(&[v, v, v]))),
            other => bail!("not a color encoding: {other}"),
        }
        if data.len() != pixels * 3 {
            bail!("image is {}x{} but holds {} bytes", w, h, self.data.len());
        }
        Ok(Rgb { width: w, height: h, data })
    }

    /// Depth in meters: 16-bit encodings are millimeters, 32FC1 is meters; 0 and non-finite mean no reading.
    pub fn to_depth(&self) -> Result<Depth> {
        let (w, h) = (self.width, self.height);
        if self.compressed {
            let decoded = image::load_from_memory(&self.data).context("decoding a compressed depth image")?.to_luma16();
            return Ok(Depth { width: decoded.width(), height: decoded.height(), meters: decoded.into_raw().into_iter().map(|mm| mm as f32 / 1000.0).collect() });
        }
        let step = if self.step == 0 { w as usize * channels(&self.encoding) } else { self.step as usize };
        let mut meters = Vec::with_capacity((w * h) as usize);
        match self.encoding.as_str() {
            "16UC1" | "mono16" | "DEPTH16" | "Z16" => {
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let at = y * step + x * 2;
                        let bytes = [self.data[at], self.data[at + 1]];
                        let mm = if self.big_endian { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) };
                        // 65535 is RealSense's "no reading" sentinel
                        meters.push(if mm == u16::MAX { 0.0 } else { mm as f32 / 1000.0 });
                    }
                }
            }
            "32FC1" => {
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let at = y * step + x * 4;
                        let bytes = [self.data[at], self.data[at + 1], self.data[at + 2], self.data[at + 3]];
                        let m = if self.big_endian { f32::from_be_bytes(bytes) } else { f32::from_le_bytes(bytes) };
                        meters.push(if m.is_finite() && m > 0.0 { m } else { 0.0 });
                    }
                }
            }
            other => bail!("not a depth encoding: {other}"),
        }
        Ok(Depth { width: w, height: h, meters })
    }
}

fn channels(encoding: &str) -> usize {
    match encoding.to_ascii_lowercase().as_str() {
        "rgb8" | "bgr8" | "rgb" | "bgr" => 3,
        "rgba8" | "bgra8" | "rgba" | "bgra" | "32fc1" => 4,
        "16uc1" | "mono16" | "depth16" | "z16" => 2,
        _ => 1,
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::lcm::Writer;

    pub const IMAGE_FINGERPRINT: [u8; 8] = *b"S\\\xfa\xce\x1fOW\x17";

    /// An LCM sensor_msgs.Image the way dimos writes one.
    pub fn lcm_image(sec: i32, frame: &str, width: u32, height: u32, encoding: &str, data: &[u8]) -> Vec<u8> {
        let mut writer = Writer::new(&IMAGE_FINGERPRINT);
        writer.i32(data.len() as i32);
        writer.header(&Header { stamp_sec: sec, stamp_nsec: 0, frame_id: frame.into() });
        writer.i32(height as i32).i32(width as i32).string(encoding).u8(0);
        let step = data.len() as u32 / height.max(1);
        writer.i32(step as i32);
        let mut bytes = writer.bytes.clone();
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn lcm_rgb_and_depth_round_trip() {
        let rgb = lcm_image(5, "cam", 2, 1, "bgr8", &[1, 2, 3, 4, 5, 6]);
        let image = decode_lcm_image(&rgb).unwrap();
        assert_eq!((image.width, image.height, image.header.frame_id.as_str()), (2, 1, "cam"));
        assert_eq!(image.to_rgb().unwrap().data, vec![3, 2, 1, 6, 5, 4]);
        let mut depth = Vec::new();
        for mm in [1500u16, u16::MAX] {
            depth.extend_from_slice(&mm.to_le_bytes());
        }
        let image = decode_lcm_image(&lcm_image(5, "depth", 2, 1, "16UC1", &depth)).unwrap();
        assert!(image.is_depth());
        assert_eq!(image.to_depth().unwrap().meters, vec![1.5, 0.0]);
    }

    #[test]
    fn intrinsics_scale_with_the_image() {
        let info = CameraInfo { header: Header::at(0.0, "c"), width: 848, height: 480, k: [427.0, 0.0, 425.0, 0.0, 426.0, 242.0, 0.0, 0.0, 1.0] };
        let half = info.scaled(424, 240);
        assert_eq!((half.fx(), half.cx(), half.fy(), half.cy()), (213.5, 212.5, 213.0, 121.0));
    }
}
