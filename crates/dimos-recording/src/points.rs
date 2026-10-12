//! A PointCloud2's x/y/z (and intensity when present) as plain f32 arrays, whatever its field layout.
use crate::lcm::PointCloud2;
use anyhow::{bail, Result};

/// sensor_msgs/PointField datatypes
const INT8: u8 = 1;
const UINT8: u8 = 2;
const INT16: u8 = 3;
const UINT16: u8 = 4;
const INT32: u8 = 5;
const UINT32: u8 = 6;
const FLOAT32: u8 = 7;
const FLOAT64: u8 = 8;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cloud {
    pub ts: f64,
    pub frame_id: String,
    pub points: Vec<[f32; 3]>,
    pub intensity: Option<Vec<f32>>,
}

fn reader(datatype: u8, big: bool) -> Option<fn(&[u8], bool) -> f32> {
    fn get<const N: usize>(bytes: &[u8]) -> [u8; N] {
        bytes[..N].try_into().unwrap()
    }
    Some(match datatype {
        FLOAT32 => |b: &[u8], big: bool| if big { f32::from_be_bytes(get(b)) } else { f32::from_le_bytes(get(b)) },
        FLOAT64 => |b: &[u8], big: bool| (if big { f64::from_be_bytes(get(b)) } else { f64::from_le_bytes(get(b)) }) as f32,
        INT8 => |b: &[u8], _| b[0] as i8 as f32,
        UINT8 => |b: &[u8], _| b[0] as f32,
        INT16 => |b: &[u8], big: bool| (if big { i16::from_be_bytes(get(b)) } else { i16::from_le_bytes(get(b)) }) as f32,
        UINT16 => |b: &[u8], big: bool| (if big { u16::from_be_bytes(get(b)) } else { u16::from_le_bytes(get(b)) }) as f32,
        INT32 => |b: &[u8], big: bool| (if big { i32::from_be_bytes(get(b)) } else { i32::from_le_bytes(get(b)) }) as f32,
        UINT32 => |b: &[u8], big: bool| (if big { u32::from_be_bytes(get(b)) } else { u32::from_le_bytes(get(b)) }) as f32,
        _ => return None,
    })
    .filter(|_| big || !big)
}

fn size_of(datatype: u8) -> usize {
    match datatype {
        INT8 | UINT8 => 1,
        INT16 | UINT16 => 2,
        FLOAT64 => 8,
        _ => 4,
    }
}

/// Non-finite points (a lidar's "no return") are dropped.
pub fn extract(cloud: &PointCloud2) -> Result<Cloud> {
    let field = |name: &str| cloud.fields.iter().find(|field| field.name == name);
    let (Some(x), Some(y), Some(z)) = (field("x"), field("y"), field("z")) else {
        bail!("point cloud has no x/y/z fields");
    };
    let step = cloud.point_step as usize;
    if step == 0 {
        bail!("point_step is 0");
    }
    let count = (cloud.width as usize * cloud.height.max(1) as usize).min(cloud.data.len() / step);
    let big = cloud.is_bigendian;
    let read = |field: &crate::lcm::PointField| -> Result<(usize, fn(&[u8], bool) -> f32)> {
        let read = reader(field.datatype, big).ok_or_else(|| anyhow::anyhow!("unsupported field type {}", field.datatype))?;
        if field.offset as usize + size_of(field.datatype) > step {
            bail!("field {} runs past point_step", field.name);
        }
        Ok((field.offset as usize, read))
    };
    let (xo, xr) = read(x)?;
    let (yo, yr) = read(y)?;
    let (zo, zr) = read(z)?;
    let intensity_field = field("intensity").and_then(|field| read(field).ok());
    let mut points = Vec::with_capacity(count);
    let mut intensity = intensity_field.map(|_| Vec::with_capacity(count));
    for index in 0..count {
        let point = &cloud.data[index * step..(index + 1) * step];
        let p = [xr(&point[xo..], big), yr(&point[yo..], big), zr(&point[zo..], big)];
        if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
            continue;
        }
        points.push(p);
        if let (Some(values), Some((offset, read))) = (intensity.as_mut(), intensity_field) {
            values.push(read(&point[offset..], big));
        }
    }
    Ok(Cloud { ts: cloud.header.ts(), frame_id: cloud.header.frame_id.clone(), points, intensity })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcm::{decode_point_cloud2, encode_xyz_cloud, Header};

    #[test]
    fn extracts_and_drops_non_finite() {
        let payload = encode_xyz_cloud(&Header::at(1.0, "lidar"), &[[1.0, 2.0, 3.0], [f32::NAN, 0.0, 0.0]], Some(&[5.0, 6.0]));
        let cloud = extract(&decode_point_cloud2(&payload).unwrap()).unwrap();
        assert_eq!(cloud.points, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(cloud.intensity, Some(vec![5.0]));
        assert_eq!(cloud.frame_id, "lidar");
    }
}
