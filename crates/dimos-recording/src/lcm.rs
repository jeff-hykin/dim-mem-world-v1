//! dimos's LCM wire format for the handful of types a map needs: decode (what recordings hold) and encode (what the
//! Map Editor saves back). Every message starts with its 8-byte fingerprint; integers are big-endian; strings are a
//! u32 length (including a trailing NUL) then the bytes; LCM hoists array lengths to the front of a struct.
//! Reader and the decoders come from dim-live-viewer's server (msgs.rs).
use anyhow::{bail, Result};

pub const TF_TYPE: &str = "tf2_msgs.TFMessage";
pub const POINT_CLOUD2_TYPE: &str = "sensor_msgs.PointCloud2";
pub const ODOMETRY_TYPE: &str = "nav_msgs.Odometry";
pub const POSE_STAMPED_TYPE: &str = "geometry_msgs.PoseStamped";
pub const PATH_TYPE: &str = "nav_msgs.Path";
pub const OCCUPANCY_GRID_TYPE: &str = "nav_msgs.OccupancyGrid";
pub const STRING_TYPE: &str = "std_msgs.String";

// From the generated dimos-lcm python types (`Type._get_packed_fingerprint()`).
pub const TF_FINGERPRINT: [u8; 8] = [0xc2, 0xb8, 0xa1, 0xc3, 0x3a, 0x89, 0x23, 0xec];
pub const POINT_CLOUD2_FINGERPRINT: [u8; 8] = [0xf5, 0xeb, 0x3d, 0xa1, 0xc2, 0x85, 0x31, 0x75];
pub const ODOMETRY_FINGERPRINT: [u8; 8] = [0x94, 0xe1, 0x7f, 0x94, 0x64, 0x8c, 0xf0, 0xe4];
pub const POSE_STAMPED_FINGERPRINT: [u8; 8] = [0x6a, 0x82, 0x69, 0x64, 0x58, 0xc2, 0x79, 0xa0];
pub const PATH_FINGERPRINT: [u8; 8] = [0xc3, 0xae, 0x62, 0xac, 0xb3, 0x57, 0x93, 0xe2];
pub const OCCUPANCY_GRID_FINGERPRINT: [u8; 8] = [0xe7, 0xdf, 0xd1, 0x79, 0xcd, 0xfc, 0x3b, 0x65];
pub const STRING_FINGERPRINT: [u8; 8] = [0x21, 0xbf, 0x37, 0x09, 0x9b, 0x9d, 0x5e, 0x15];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    pub stamp_sec: i32,
    pub stamp_nsec: i32,
    pub frame_id: String,
}

impl Header {
    pub fn at(ts: f64, frame_id: &str) -> Self {
        let sec = ts.floor();
        Header { stamp_sec: sec as i32, stamp_nsec: ((ts - sec) * 1e9).round().min(999_999_999.0) as i32, frame_id: frame_id.into() }
    }

    pub fn ts(&self) -> f64 {
        self.stamp_sec as f64 + self.stamp_nsec as f64 * 1e-9
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pose {
    pub position: [f64; 3],
    /// x, y, z, w
    pub orientation: [f64; 4],
}

impl Pose {
    pub const IDENTITY: Pose = Pose { position: [0.0; 3], orientation: [0.0, 0.0, 0.0, 1.0] };
}

#[derive(Debug, Clone, PartialEq)]
pub struct TfEdge {
    pub header: Header,
    pub child: String,
    pub translation: [f64; 3],
    pub rotation: [f64; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointField {
    pub name: String,
    pub offset: u32,
    pub datatype: u8,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud2 {
    pub header: Header,
    pub height: u32,
    pub width: u32,
    pub fields: Vec<PointField>,
    pub is_bigendian: bool,
    pub point_step: u32,
    pub row_step: u32,
    pub data: Vec<u8>,
    pub is_dense: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Odometry {
    pub header: Header,
    pub child_frame_id: String,
    pub pose: Pose,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseStamped {
    pub header: Header,
    pub pose: Pose,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyGrid {
    pub header: Header,
    pub resolution: f32,
    pub width: u32,
    pub height: u32,
    /// the grid's cell (0, 0) corner in `header.frame_id`
    pub origin: Pose,
    /// row-major from the origin, -1 unknown, 0 free, 100 occupied
    pub data: Vec<i8>,
}

pub struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, offset: 0 }
    }

    pub fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.offset + count > self.data.len() {
            bail!("truncated message");
        }
        let slice = &self.data[self.offset..self.offset + count];
        self.offset += count;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into()?))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into()?))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_be_bytes(self.take(4)?.try_into()?))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_be_bytes(self.take(8)?.try_into()?))
    }

    pub fn boolean(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    pub fn f64_array<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut values = [0.0; N];
        for value in values.iter_mut() {
            *value = self.f64()?;
        }
        Ok(values)
    }

    pub fn length(&mut self) -> Result<usize> {
        let length = self.i32()?;
        if length < 0 {
            bail!("negative lcm array length");
        }
        Ok(length as usize)
    }

    pub fn string(&mut self) -> Result<String> {
        let length = self.u32()? as usize;
        if length == 0 {
            bail!("zero-length lcm string");
        }
        let raw = self.take(length)?;
        Ok(String::from_utf8_lossy(&raw[..length - 1]).into_owned())
    }

    pub fn expect_fingerprint(&mut self, expected: &[u8; 8]) -> Result<()> {
        if self.take(8)? != expected {
            bail!("fingerprint mismatch");
        }
        Ok(())
    }
}

/// The leading `seq` is read and dropped.
pub fn read_header(reader: &mut Reader) -> Result<Header> {
    reader.i32()?;
    Ok(Header { stamp_sec: reader.i32()?, stamp_nsec: reader.i32()?, frame_id: reader.string()? })
}

fn read_pose(reader: &mut Reader) -> Result<Pose> {
    Ok(Pose { position: reader.f64_array()?, orientation: reader.f64_array()? })
}

pub fn fingerprint_of(payload: &[u8]) -> Option<[u8; 8]> {
    payload.get(..8).map(|bytes| bytes.try_into().unwrap())
}

/// Which of our types a payload is, from its fingerprint.
pub fn type_of(payload: &[u8]) -> Option<&'static str> {
    let fingerprint = fingerprint_of(payload)?;
    [
        (TF_FINGERPRINT, TF_TYPE),
        (POINT_CLOUD2_FINGERPRINT, POINT_CLOUD2_TYPE),
        (ODOMETRY_FINGERPRINT, ODOMETRY_TYPE),
        (POSE_STAMPED_FINGERPRINT, POSE_STAMPED_TYPE),
        (PATH_FINGERPRINT, PATH_TYPE),
        (OCCUPANCY_GRID_FINGERPRINT, OCCUPANCY_GRID_TYPE),
        (STRING_FINGERPRINT, STRING_TYPE),
    ]
    .into_iter()
    .find(|(print, _)| *print == fingerprint)
    .map(|(_, name)| name)
}

pub fn decode_tf(payload: &[u8]) -> Result<Vec<TfEdge>> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&TF_FINGERPRINT)?;
    let count = reader.i32()?;
    if !(0..=4096).contains(&count) {
        bail!("nonsense transform count");
    }
    let mut edges = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let header = read_header(&mut reader)?;
        let child = reader.string()?;
        edges.push(TfEdge { header, child, translation: reader.f64_array()?, rotation: reader.f64_array()? });
    }
    Ok(edges)
}

pub fn decode_point_cloud2(payload: &[u8]) -> Result<PointCloud2> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&POINT_CLOUD2_FINGERPRINT)?;
    let fields_length = reader.i32()?;
    let data_length = reader.i32()?;
    if !(0..=1024).contains(&fields_length) || data_length < 0 {
        bail!("nonsense point cloud lengths");
    }
    let header = read_header(&mut reader)?;
    let height = reader.i32()? as u32;
    let width = reader.i32()? as u32;
    let mut fields = Vec::with_capacity(fields_length as usize);
    for _ in 0..fields_length {
        fields.push(PointField { name: reader.string()?, offset: reader.i32()? as u32, datatype: reader.u8()?, count: reader.i32()? as u32 });
    }
    let is_bigendian = reader.boolean()?;
    let point_step = reader.i32()? as u32;
    let row_step = reader.i32()? as u32;
    let data = reader.take(data_length as usize)?.to_vec();
    let is_dense = reader.boolean()?;
    Ok(PointCloud2 { header, height, width, fields, is_bigendian, point_step, row_step, data, is_dense })
}

pub fn decode_odometry(payload: &[u8]) -> Result<Odometry> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&ODOMETRY_FINGERPRINT)?;
    Ok(Odometry { header: read_header(&mut reader)?, child_frame_id: reader.string()?, pose: read_pose(&mut reader)? })
}

pub fn decode_pose_stamped(payload: &[u8]) -> Result<PoseStamped> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&POSE_STAMPED_FINGERPRINT)?;
    Ok(PoseStamped { header: read_header(&mut reader)?, pose: read_pose(&mut reader)? })
}

pub fn decode_string(payload: &[u8]) -> Result<String> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&STRING_FINGERPRINT)?;
    reader.string()
}

pub fn decode_occupancy_grid(payload: &[u8]) -> Result<OccupancyGrid> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&OCCUPANCY_GRID_FINGERPRINT)?;
    let length = reader.length()?;
    let header = read_header(&mut reader)?;
    reader.i32()?; // map_load_time
    reader.i32()?;
    let resolution = reader.f32()?;
    let width = reader.i32()? as u32;
    let height = reader.i32()? as u32;
    let origin = read_pose(&mut reader)?;
    let data = reader.take(length)?.iter().map(|byte| *byte as i8).collect();
    Ok(OccupancyGrid { header, resolution, width, height, origin, data })
}

pub fn decode_path(payload: &[u8]) -> Result<Vec<PoseStamped>> {
    let mut reader = Reader::new(payload);
    reader.expect_fingerprint(&PATH_FINGERPRINT)?;
    let count = reader.length()?;
    read_header(&mut reader)?;
    (0..count).map(|_| Ok(PoseStamped { header: read_header(&mut reader)?, pose: read_pose(&mut reader)? })).collect()
}

/// Builds an LCM message.
#[derive(Default)]
pub struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn new(fingerprint: &[u8; 8]) -> Self {
        Writer { bytes: fingerprint.to_vec() }
    }

    pub fn i32(&mut self, value: i32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.bytes.push(value);
        self
    }

    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn string(&mut self, value: &str) -> &mut Self {
        self.bytes.extend_from_slice(&(value.len() as u32 + 1).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
        self.bytes.push(0);
        self
    }

    pub fn header(&mut self, header: &Header) -> &mut Self {
        self.i32(0).i32(header.stamp_sec).i32(header.stamp_nsec).string(&header.frame_id)
    }

    pub fn pose(&mut self, pose: &Pose) -> &mut Self {
        for value in pose.position.iter().chain(pose.orientation.iter()) {
            self.f64(*value);
        }
        self
    }
}

/// An x, y, z float32 cloud (plus an optional float32 `intensity`).
pub fn encode_xyz_cloud(header: &Header, points: &[[f32; 3]], intensity: Option<&[f32]>) -> Vec<u8> {
    let step: u32 = if intensity.is_some() { 16 } else { 12 };
    let mut fields = vec![("x", 0u32), ("y", 4), ("z", 8)];
    if intensity.is_some() {
        fields.push(("intensity", 12));
    }
    let mut data = Vec::with_capacity(points.len() * step as usize);
    for (index, point) in points.iter().enumerate() {
        for value in point {
            data.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(intensity) = intensity {
            data.extend_from_slice(&intensity[index].to_le_bytes());
        }
    }
    let mut writer = Writer::new(&POINT_CLOUD2_FINGERPRINT);
    writer.i32(fields.len() as i32).i32(data.len() as i32).header(header).i32(1).i32(points.len() as i32);
    for (name, offset) in fields {
        writer.string(name).i32(offset as i32).u8(7).i32(1); // 7 = FLOAT32
    }
    writer.u8(0).i32(step as i32).i32((step as usize * points.len()) as i32);
    writer.bytes.extend_from_slice(&data);
    writer.u8(1);
    writer.bytes
}

pub fn encode_string(value: &str) -> Vec<u8> {
    let mut writer = Writer::new(&STRING_FINGERPRINT);
    writer.string(value);
    writer.bytes
}

pub fn encode_occupancy_grid(grid: &OccupancyGrid) -> Vec<u8> {
    let mut writer = Writer::new(&OCCUPANCY_GRID_FINGERPRINT);
    writer.i32(grid.data.len() as i32).header(&grid.header).i32(grid.header.stamp_sec).i32(grid.header.stamp_nsec);
    writer.f32(grid.resolution).i32(grid.width as i32).i32(grid.height as i32).pose(&grid.origin);
    writer.bytes.extend(grid.data.iter().map(|cell| *cell as u8));
    writer.bytes
}

pub fn encode_path(header: &Header, poses: &[PoseStamped]) -> Vec<u8> {
    let mut writer = Writer::new(&PATH_FINGERPRINT);
    writer.i32(poses.len() as i32).header(header);
    for pose in poses {
        writer.header(&pose.header).pose(&pose.pose);
    }
    writer.bytes
}

pub fn encode_tf(edges: &[TfEdge]) -> Vec<u8> {
    let mut writer = Writer::new(&TF_FINGERPRINT);
    writer.i32(edges.len() as i32);
    for edge in edges {
        writer.header(&edge.header).string(&edge.child);
        for value in edge.translation.iter().chain(edge.rotation.iter()) {
            writer.f64(*value);
        }
    }
    writer.bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let header = Header::at(12.25, "world");
        assert!((header.ts() - 12.25).abs() < 1e-9);
        let points = [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]];
        let cloud = decode_point_cloud2(&encode_xyz_cloud(&header, &points, Some(&[0.5, 1.0]))).unwrap();
        assert_eq!(cloud.header, header);
        assert_eq!((cloud.width, cloud.point_step, cloud.fields.len()), (2, 16, 4));
        assert_eq!(decode_string(&encode_string("{\"a\":1}")).unwrap(), "{\"a\":1}");
        let grid = OccupancyGrid {
            header: header.clone(),
            resolution: 0.05,
            width: 2,
            height: 1,
            origin: Pose { position: [1.0, 2.0, 0.0], orientation: [0.0, 0.0, 0.0, 1.0] },
            data: vec![-1, 100],
        };
        assert_eq!(decode_occupancy_grid(&encode_occupancy_grid(&grid)).unwrap(), grid);
        let poses = vec![PoseStamped { header: header.clone(), pose: Pose::IDENTITY }];
        assert_eq!(decode_path(&encode_path(&header, &poses)).unwrap(), poses);
        let edges = vec![TfEdge { header, child: "base_link".into(), translation: [1.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0] }];
        assert_eq!(decode_tf(&encode_tf(&edges)).unwrap(), edges);
        assert_eq!(type_of(&encode_tf(&edges)), Some(TF_TYPE));
    }
}
