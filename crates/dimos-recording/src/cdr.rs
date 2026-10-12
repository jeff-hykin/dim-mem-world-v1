//! ROS 2 CDR decoding for the types the Live Viewer's recorder writes as `cdr` (sensor_msgs/msg/PointCloud2,
//! tf2_msgs/msg/TFMessage, nav_msgs/msg/Odometry, geometry_msgs/msg/PoseStamped), into the same structs the LCM
//! decoders produce. A 4-byte encapsulation header (byte 1: 0 = big, 1 = little endian) precedes the body; every
//! primitive is aligned to its own size, counted from the end of that header. ROS 2 headers carry no `seq`.
use crate::lcm::{Header, Odometry, PointCloud2, PointField, Pose, PoseStamped, TfEdge};
use anyhow::{bail, Result};

pub const POINT_CLOUD2_SCHEMA: &str = "sensor_msgs/msg/PointCloud2";
pub const TF_SCHEMA: &str = "tf2_msgs/msg/TFMessage";
pub const ODOMETRY_SCHEMA: &str = "nav_msgs/msg/Odometry";
pub const POSE_STAMPED_SCHEMA: &str = "geometry_msgs/msg/PoseStamped";

pub(crate) struct Cdr<'a> {
    data: &'a [u8],
    offset: usize,
    little: bool,
}

impl<'a> Cdr<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 4 {
            bail!("truncated cdr message");
        }
        Ok(Cdr { data: &data[4..], offset: 0, little: data[1] & 1 == 1 })
    }

    pub(crate) fn align(&mut self, size: usize) {
        self.offset = self.offset.div_ceil(size) * size;
    }

    pub(crate) fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.offset + count > self.data.len() {
            bail!("truncated cdr message");
        }
        let slice = &self.data[self.offset..self.offset + count];
        self.offset += count;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        self.align(4);
        let bytes: [u8; 4] = self.take(4)?.try_into()?;
        Ok(if self.little { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) })
    }

    pub(crate) fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    pub(crate) fn f64(&mut self) -> Result<f64> {
        self.align(8);
        let bytes: [u8; 8] = self.take(8)?.try_into()?;
        Ok(if self.little { f64::from_le_bytes(bytes) } else { f64::from_be_bytes(bytes) })
    }

    pub(crate) fn f64s<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut values = [0.0; N];
        for value in values.iter_mut() {
            *value = self.f64()?;
        }
        Ok(values)
    }

    pub(crate) fn string(&mut self) -> Result<String> {
        let length = self.u32()? as usize;
        let raw = self.take(length)?;
        Ok(String::from_utf8_lossy(raw.strip_suffix(&[0]).unwrap_or(raw)).into_owned())
    }

    pub(crate) fn header(&mut self) -> Result<Header> {
        Ok(Header { stamp_sec: self.i32()?, stamp_nsec: self.u32()? as i32, frame_id: self.string()? })
    }

    pub(crate) fn pose(&mut self) -> Result<Pose> {
        Ok(Pose { position: self.f64s()?, orientation: self.f64s()? })
    }
}

pub fn decode_point_cloud2(payload: &[u8]) -> Result<PointCloud2> {
    let mut cdr = Cdr::new(payload)?;
    let header = cdr.header()?;
    let height = cdr.u32()?;
    let width = cdr.u32()?;
    let field_count = cdr.u32()? as usize;
    if field_count > 1024 {
        bail!("nonsense field count");
    }
    let mut fields = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let name = cdr.string()?;
        let offset = cdr.u32()?;
        let datatype = cdr.u8()?;
        let count = cdr.u32()?;
        fields.push(PointField { name, offset, datatype, count });
    }
    let is_bigendian = cdr.u8()? != 0;
    let point_step = cdr.u32()?;
    let row_step = cdr.u32()?;
    let length = cdr.u32()? as usize;
    let data = cdr.take(length)?.to_vec();
    let is_dense = cdr.u8()? != 0;
    Ok(PointCloud2 { header, height, width, fields, is_bigendian, point_step, row_step, data, is_dense })
}

pub fn decode_tf(payload: &[u8]) -> Result<Vec<TfEdge>> {
    let mut cdr = Cdr::new(payload)?;
    let count = cdr.u32()? as usize;
    if count > 4096 {
        bail!("nonsense transform count");
    }
    (0..count)
        .map(|_| Ok(TfEdge { header: cdr.header()?, child: cdr.string()?, translation: cdr.f64s()?, rotation: cdr.f64s()? }))
        .collect()
}

pub fn decode_odometry(payload: &[u8]) -> Result<Odometry> {
    let mut cdr = Cdr::new(payload)?;
    Ok(Odometry { header: cdr.header()?, child_frame_id: cdr.string()?, pose: cdr.pose()? })
}

pub fn decode_pose_stamped(payload: &[u8]) -> Result<PoseStamped> {
    let mut cdr = Cdr::new(payload)?;
    Ok(PoseStamped { header: cdr.header()?, pose: cdr.pose()? })
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A little-endian CDR writer, enough to build fixtures the way the Live Viewer's recorder does.
    #[derive(Default)]
    pub struct CdrWriter {
        body: Vec<u8>,
    }

    impl CdrWriter {
        fn align(&mut self, size: usize) {
            while self.body.len() % size != 0 {
                self.body.push(0);
            }
        }
        pub fn u8(&mut self, value: u8) -> &mut Self {
            self.body.push(value);
            self
        }
        pub fn u32(&mut self, value: u32) -> &mut Self {
            self.align(4);
            self.body.extend_from_slice(&value.to_le_bytes());
            self
        }
        pub fn f64(&mut self, value: f64) -> &mut Self {
            self.align(8);
            self.body.extend_from_slice(&value.to_le_bytes());
            self
        }
        pub fn string(&mut self, value: &str) -> &mut Self {
            self.u32(value.len() as u32 + 1);
            self.body.extend_from_slice(value.as_bytes());
            self.body.push(0);
            self
        }
        pub fn header(&mut self, sec: u32, nsec: u32, frame: &str) -> &mut Self {
            self.u32(sec).u32(nsec).string(frame)
        }
        pub fn bytes(&mut self, data: &[u8]) -> &mut Self {
            self.u32(data.len() as u32);
            self.body.extend_from_slice(data);
            self
        }
        pub fn finish(&self) -> Vec<u8> {
            let mut out = vec![0, 1, 0, 0];
            out.extend_from_slice(&self.body);
            out
        }
    }

    pub fn cloud_cdr(sec: u32, frame: &str, points: &[[f32; 3]]) -> Vec<u8> {
        let mut writer = CdrWriter::default();
        writer.header(sec, 0, frame).u32(1).u32(points.len() as u32).u32(3);
        for (index, name) in ["x", "y", "z"].iter().enumerate() {
            writer.string(name).u32(index as u32 * 4).u8(7).u32(1);
        }
        let data: Vec<u8> = points.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        writer.u8(0).u32(12).u32(12 * points.len() as u32).bytes(&data).u8(1);
        writer.finish()
    }

    pub fn tf_cdr(sec: u32, parent: &str, child: &str, translation: [f64; 3]) -> Vec<u8> {
        let mut writer = CdrWriter::default();
        writer.u32(1).header(sec, 0, parent).string(child);
        for value in translation.iter().chain([0.0, 0.0, 0.0, 1.0].iter()) {
            writer.f64(*value);
        }
        writer.finish()
    }

    #[test]
    fn decodes_what_the_recorder_writes() {
        let cloud = decode_point_cloud2(&cloud_cdr(7, "lidar", &[[1.0, 2.0, 3.0]])).unwrap();
        assert_eq!(cloud.header.frame_id, "lidar");
        assert_eq!((cloud.width, cloud.point_step, cloud.data.len()), (1, 12, 12));
        assert_eq!(f32::from_le_bytes(cloud.data[8..12].try_into().unwrap()), 3.0);
        let tf = decode_tf(&tf_cdr(3, "odom", "base_link", [1.0, 2.0, 0.5])).unwrap();
        assert_eq!(tf[0].header.frame_id, "odom");
        assert_eq!(tf[0].child, "base_link");
        assert_eq!(tf[0].translation, [1.0, 2.0, 0.5]);
        assert_eq!(tf[0].rotation, [0.0, 0.0, 0.0, 1.0]);
    }
}
