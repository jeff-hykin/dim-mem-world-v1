//! Reading and writing dimos recordings (`.db` memory2 SQLite and `.mcap`) without dimos: the message types a map
//! needs (clouds, tf, odometry), in both LCM (dimos's wire format) and ROS 2 CDR (what the Live Viewer's recorder
//! transcodes to). Shared by dim-map-builder's server; the Live Viewer's recorder writes the files it reads.
pub mod cdr;
pub mod db;
pub mod image;
pub mod lcm;
pub mod mcap_append;
pub mod mcap_io;
pub mod points;

use anyhow::{bail, Context, Result};
use lcm::{Odometry, PoseStamped, TfEdge};
use points::Cloud;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Cloud,
    Tf,
    Odometry,
    PoseStamped,
    Path,
    OccupancyGrid,
    Text,
    Image,
    CompressedImage,
    CameraInfo,
    Other,
}

impl Kind {
    /// From a dimos type (`sensor_msgs.PointCloud2`) or a ROS 2 schema name (`sensor_msgs/msg/PointCloud2`).
    pub fn of(type_name: &str) -> Kind {
        let normalized = type_name.replace("/msg/", ".").replace('/', ".");
        match normalized.as_str() {
            "sensor_msgs.PointCloud2" => Kind::Cloud,
            "tf2_msgs.TFMessage" => Kind::Tf,
            "nav_msgs.Odometry" => Kind::Odometry,
            "geometry_msgs.PoseStamped" => Kind::PoseStamped,
            "nav_msgs.Path" => Kind::Path,
            "nav_msgs.OccupancyGrid" => Kind::OccupancyGrid,
            "std_msgs.String" => Kind::Text,
            "sensor_msgs.Image" => Kind::Image,
            "sensor_msgs.CompressedImage" => Kind::CompressedImage,
            "sensor_msgs.CameraInfo" => Kind::CameraInfo,
            _ => Kind::Other,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub kind: Kind,
    pub count: u64,
    /// the stream this one was made from (a corrected copy of a lidar), from the mcap channel's `derived_from`
    /// metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<String>,
}

/// A decoded message, with the stored observation pose a `.db` row may carry (old datasets put world poses there).
pub enum Message {
    Cloud(Cloud),
    Tf(Vec<TfEdge>),
    Odometry(Odometry),
    PoseStamped(PoseStamped),
}

enum Backend {
    Db(rusqlite::Connection, Vec<db::DbStream>),
    Mcap(mcap_io::McapFile),
}

pub struct Recording {
    pub path: PathBuf,
    backend: Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Db,
    Mcap,
}

pub fn format_of(path: &Path) -> Result<Format> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("db") => Ok(Format::Db),
        Some("mcap") => Ok(Format::Mcap),
        _ => bail!("not a recording (.db or .mcap): {}", path.display()),
    }
}

fn decode(encoding: &str, kind: Kind, payload: &[u8]) -> Result<Option<Message>> {
    let cdr = encoding == "cdr";
    Ok(Some(match kind {
        Kind::Cloud => Message::Cloud(points::extract(&if cdr { cdr::decode_point_cloud2(payload)? } else { lcm::decode_point_cloud2(payload)? })?),
        Kind::Tf => Message::Tf(if cdr { cdr::decode_tf(payload)? } else { lcm::decode_tf(payload)? }),
        Kind::Odometry => Message::Odometry(if cdr { cdr::decode_odometry(payload)? } else { lcm::decode_odometry(payload)? }),
        Kind::PoseStamped => Message::PoseStamped(if cdr { cdr::decode_pose_stamped(payload)? } else { lcm::decode_pose_stamped(payload)? }),
        _ => return Ok(None),
    }))
}

impl Recording {
    pub fn open(path: &Path) -> Result<Recording> {
        let backend = match format_of(path)? {
            Format::Db => {
                let connection = db::open_read(path)?;
                let streams = db::streams(&connection)?;
                Backend::Db(connection, streams)
            }
            Format::Mcap => Backend::Mcap(mcap_io::McapFile::open(path)?),
        };
        Ok(Recording { path: path.to_path_buf(), backend })
    }

    pub fn format(&self) -> Format {
        match self.backend {
            Backend::Db(..) => Format::Db,
            Backend::Mcap(_) => Format::Mcap,
        }
    }

    pub fn streams(&self) -> Result<Vec<StreamInfo>> {
        Ok(match &self.backend {
            Backend::Db(_, streams) => streams
                .iter()
                .map(|s| StreamInfo { name: s.name.clone(), type_name: s.kind.clone(), kind: Kind::of(&s.kind), count: s.count, derived_from: None })
                .collect(),
            Backend::Mcap(file) => file
                .channels()?
                .into_iter()
                .map(|c| StreamInfo { kind: Kind::of(&c.kind), name: c.topic, type_name: c.kind, count: c.count, derived_from: c.derived_from })
                .collect(),
        })
    }

    /// Every message of `stream` decoded, with the row's stored pose (db only). `each` returns false to stop.
    pub fn for_each(&self, stream: &str, mut each: impl FnMut(f64, Message, Option<[f64; 7]>) -> Result<bool>) -> Result<()> {
        match &self.backend {
            Backend::Db(connection, streams) => {
                let info = streams.iter().find(|s| s.name == stream).with_context(|| format!("no stream {stream}"))?;
                let kind = Kind::of(&info.kind);
                db::for_each(connection, info, |row| match decode("lcm", kind, &row.payload) {
                    Ok(Some(message)) => each(row.ts, message, row.pose),
                    Ok(None) => Ok(true),
                    Err(_) => Ok(true), // one bad row doesn't end a recording
                })
            }
            Backend::Mcap(file) => file.for_each(stream, |ts, encoding, kind, payload| match decode(encoding, Kind::of(kind), payload) {
                Ok(Some(message)) => each(ts, message, None),
                _ => Ok(true),
            }),
        }
    }

    /// Every raw message of `stream`: `each(ts, encoding, type, payload)` with `encoding` "lcm" or "cdr" and `type` the
    /// stream's type name (dimos type or ROS 2 schema). Db rows come in id order, mcap messages through the chunk index
    /// (only chunks holding the topic are read). `each` returns false to stop.
    pub fn raw_each(&self, stream: &str, mut each: impl FnMut(f64, &str, &str, &[u8]) -> Result<bool>) -> Result<()> {
        match &self.backend {
            Backend::Db(connection, streams) => {
                let info = streams.iter().find(|s| s.name == stream).with_context(|| format!("no stream {stream}"))?;
                db::for_each(connection, info, |row| each(row.ts, "lcm", &info.kind, &row.payload))
            }
            Backend::Mcap(file) => file.for_each_indexed(stream, each),
        }
    }

    /// Message times of `stream` without reading payloads: db row stamps; mcap log times from the chunks' message
    /// indexes (no decompression). Sorted.
    pub fn timestamps(&self, stream: &str) -> Result<Vec<f64>> {
        let mut times = match &self.backend {
            Backend::Db(connection, _) => db::timestamps(connection, stream)?.into_iter().map(|(_, ts)| ts).collect(),
            Backend::Mcap(file) => file.log_times(stream)?,
        };
        times.sort_by(f64::total_cmp);
        Ok(times)
    }

    /// Like `raw_each`, but only messages whose time passes `keep` are read: a db fetches just those rows' blobs (one
    /// short query each, so no read stays open while the same file is written), an mcap skips the rest unread.
    pub fn raw_each_where(&self, stream: &str, mut keep: impl FnMut(f64) -> bool, mut each: impl FnMut(f64, &str, &str, &[u8]) -> Result<bool>) -> Result<()> {
        match &self.backend {
            Backend::Db(connection, streams) => {
                let info = streams.iter().find(|s| s.name == stream).with_context(|| format!("no stream {stream}"))?;
                for (id, ts) in db::timestamps(connection, stream)? {
                    if keep(ts) && !each(ts, "lcm", &info.kind, &db::payload(connection, info, id)?)? {
                        break;
                    }
                }
                Ok(())
            }
            Backend::Mcap(file) => file.for_each_indexed(stream, |ts, encoding, kind, payload| if keep(ts) { each(ts, encoding, kind, payload) } else { Ok(true) }),
        }
    }

    /// Every transform in every tf stream (`tf`, `/tf`, `tf_static`, ...).
    pub fn tf_edges(&self) -> Result<Vec<TfEdge>> {
        Ok(self.tf_edges_logged()?.into_iter().map(|(_, edge)| edge).collect())
    }

    /// Every transform with the log time of the message that carried it (to tell a header stamp's clock).
    pub fn tf_edges_logged(&self) -> Result<Vec<(f64, TfEdge)>> {
        let mut edges = Vec::new();
        for stream in self.streams()?.into_iter().filter(|s| s.kind == Kind::Tf) {
            self.for_each(&stream.name, |ts, message, _| {
                if let Message::Tf(list) = message {
                    edges.extend(list.into_iter().map(|edge| (ts, edge)));
                }
                Ok(true)
            })?;
        }
        Ok(edges)
    }

    /// The newest raw payload of a stream / topic (for state the Map Editor saved earlier).
    pub fn latest(&self, stream: &str) -> Result<Option<(f64, Vec<u8>)>> {
        match &self.backend {
            Backend::Db(connection, _) => db::latest(connection, stream),
            Backend::Mcap(file) => file.latest(stream),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcap_cdr_and_lcm_clouds_read_alike() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drive.mcap");
        let lcm_cloud = lcm::encode_xyz_cloud(&lcm::Header::at(2.0, "lidar"), &[[4.0, 5.0, 6.0]], None);
        mcap_io::write_fixture(
            &path,
            &[
                ("/lidar", "cdr", "sensor_msgs/msg/PointCloud2", 1.0, cdr::tests::cloud_cdr(1, "lidar", &[[1.0, 2.0, 3.0]])),
                ("/lidar", "cdr", "sensor_msgs/msg/PointCloud2", 2.0, lcm_cloud.clone()), // garbage as cdr: skipped
                ("/lcm_lidar", "lcm", "sensor_msgs.PointCloud2", 2.0, lcm_cloud),
                ("/tf", "cdr", "tf2_msgs/msg/TFMessage", 1.0, cdr::tests::tf_cdr(1, "odom", "base_link", [1.0, 0.0, 0.0])),
            ],
        );
        let recording = Recording::open(&path).unwrap();
        let kinds: Vec<_> = recording.streams().unwrap().iter().map(|s| (s.name.clone(), s.kind)).collect();
        assert_eq!(kinds, [("/lcm_lidar".to_string(), Kind::Cloud), ("/lidar".into(), Kind::Cloud), ("/tf".into(), Kind::Tf)]);
        let mut clouds = Vec::new();
        for topic in ["/lidar", "/lcm_lidar"] {
            recording
                .for_each(topic, |_, message, _| {
                    if let Message::Cloud(cloud) = message {
                        clouds.push(cloud.points);
                    }
                    Ok(true)
                })
                .unwrap();
        }
        assert_eq!(clouds, vec![vec![[1.0, 2.0, 3.0]], vec![[4.0, 5.0, 6.0]]]);
        assert_eq!(recording.tf_edges().unwrap()[0].child, "base_link");
    }
}
