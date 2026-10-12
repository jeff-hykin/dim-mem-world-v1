//! Memory World's streams live INSIDE the recording, next to the camera they were made from: new tables in a `.db`
//! (memory2 layout, rows pickled as Python `bytes`, so dimos lists and reads them) or new channels in an `.mcap`
//! (appended in place: the summary is cut off, the new chunks go where it was, a summary covering both is put back).
//! Nothing is written beside the recording.
//!
//! A write session holds an exclusive lock on the file. An mcap is without its summary only inside `flush`, so writes
//! go in batches: a crash loses at most the batch in hand, never the recording.
use anyhow::{bail, Context, Result};
use dimos_recording::{db, format_of, mcap_append::Appender, Format, Recording};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// The logical streams; `name()` gives each format's spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stream {
    /// one per generation (and per clear): what was embedded, how, with which models
    Meta,
    /// one per keyframe: camera pose + intrinsics + a JPEG of the frame
    Frames,
    /// one per keyframe: depth drawn into the color camera, at the JPEG's size
    Depth,
    /// one per keyframe per member: fp16 per-patch embeddings
    Patches(usize),
    /// one per keyframe: OWLv2's image half (fp16 class embeddings, logit shift/scale, boxes per patch)
    Owl,
    /// one per generation: the lidar's voxel map in the world frame (f32 x, y, z), when the recording has a lidar
    Map,
}

pub const PREFIX: &str = "memworld";
/// mcap message encoding of every Memory World channel (blob.rs: MWB1 + JSON header + bytes)
pub const MESSAGE_ENCODING: &str = "memworld.mwb1";

impl Stream {
    pub fn all() -> Vec<Stream> {
        let mut all = vec![Stream::Meta, Stream::Frames, Stream::Depth, Stream::Owl, Stream::Map];
        all.extend((0..crate::siglip::MEMBERS.len()).map(Stream::Patches));
        all
    }

    fn suffix(&self) -> String {
        match self {
            Stream::Meta => "meta".into(),
            Stream::Frames => "frames".into(),
            Stream::Depth => "depth".into(),
            Stream::Patches(i) => format!("patches_{}", crate::siglip::MEMBERS[*i]),
            Stream::Owl => "owl".into(),
            Stream::Map => "map".into(),
        }
    }

    /// `memworld_frames` in a `.db` (a table name), `/memworld/frames` in an `.mcap` (a topic).
    pub fn name(&self, format: Format) -> String {
        match format {
            Format::Db => format!("{PREFIX}_{}", self.suffix()),
            Format::Mcap => format!("/{PREFIX}/{}", self.suffix()),
        }
    }
}

/// Whether a stream / topic name is one of ours (left out of what the recording "has").
pub fn is_ours(name: &str) -> bool {
    let name = name.trim_start_matches('/');
    name.starts_with(&format!("{PREFIX}_")) || name.starts_with(&format!("{PREFIX}/"))
}

/// A write session on one recording.
pub struct Writer {
    format: Format,
    path: PathBuf,
    connection: Option<db::Connection>,
    queued: Vec<(Stream, f64, Vec<u8>)>,
    _lock: File,
    written: u64,
}

impl Writer {
    pub fn open(path: &Path) -> Result<Writer> {
        let format = format_of(path)?;
        let lock = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // an mcap is locked for the session; a .db relies on SQLite's own locks (on macOS a whole-file flock would
        // block SQLite's byte-range locks on the same file)
        if format == Format::Mcap && lock.try_lock().is_err() {
            bail!("{} is being written by another process", path.display());
        }
        let connection = match format {
            Format::Db => {
                let connection = db::Connection::open(path)?;
                connection.busy_timeout(std::time::Duration::from_secs(60))?;
                for stream in Stream::all() {
                    db::ensure_stream(&connection, &stream.name(format), "builtins.bytes", "pickle")?;
                }
                Some(connection)
            }
            Format::Mcap => None,
        };
        Ok(Writer { format, path: path.to_path_buf(), connection, queued: Vec::new(), _lock: lock, written: 0 })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// Queues one message; `flush` writes it.
    pub fn put(&mut self, stream: Stream, ts: f64, bytes: Vec<u8>) {
        self.queued.push((stream, ts, bytes));
    }

    pub fn queued_bytes(&self) -> usize {
        self.queued.iter().map(|m| m.2.len()).sum()
    }

    /// Bytes added to the recording so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Writes everything queued, in one transaction (db) or one append (mcap).
    pub fn flush(&mut self) -> Result<()> {
        if self.queued.is_empty() {
            return Ok(());
        }
        let mut queued = std::mem::take(&mut self.queued);
        queued.sort_by(|a, b| a.1.total_cmp(&b.1));
        let bytes: u64 = queued.iter().map(|m| m.2.len() as u64).sum();
        match self.format {
            Format::Db => {
                let connection = self.connection.as_mut().expect("db writer has a connection");
                let mut by_stream: BTreeMap<Stream, Vec<(f64, Vec<u8>)>> = BTreeMap::new();
                for (stream, ts, data) in queued {
                    by_stream.entry(stream).or_default().push((ts, db::pickle_bytes(&data)));
                }
                for (stream, rows) in by_stream {
                    db::append_rows(connection, &stream.name(Format::Db), &rows)?;
                }
            }
            Format::Mcap => {
                let mut appender = Appender::open(&self.path)?;
                let metadata = BTreeMap::from([("written_by".to_string(), "dim-mem-world".to_string())]);
                let mut channels: BTreeMap<Stream, u16> = BTreeMap::new();
                for (stream, ts, data) in queued {
                    let id = *channels.entry(stream).or_insert_with(|| appender.channel(&stream.name(Format::Mcap), 0, MESSAGE_ENCODING, &metadata));
                    appender.write_stream(id, (ts.max(0.0) * 1e9).round() as u64, data)?;
                }
                appender.finish()?;
            }
        }
        self.written += bytes;
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if let Err(error) = self.flush() {
            eprintln!("memworld: writing the last batch failed: {error:#}");
        }
    }
}

/// Every message of one of our streams, (ts, bytes) in ts order; empty when the recording has none.
pub fn read(recording: &Recording, stream: Stream) -> Result<Vec<(f64, Vec<u8>)>> {
    let name = stream.name(recording.format());
    if !recording.streams()?.iter().any(|s| s.name == name) {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    recording.raw_each(&name, |ts, _, _, payload| {
        out.push((ts, payload.to_vec()));
        Ok(true)
    })?;
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

/// Removes Memory World from a `.db` (drops its tables). An mcap can't shrink in place, so there the caller writes a
/// `cleared` meta message instead and readers skip everything before it.
pub fn drop_db_streams(path: &Path) -> Result<()> {
    let mut connection = db::Connection::open(path)?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    let names: Vec<String> = Stream::all().iter().map(|s| s.name(Format::Db)).collect();
    db::drop_streams(&mut connection, &names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob;
    use dimos_recording::mcap_io;

    fn message(n: u8) -> Vec<u8> {
        blob::encode(&serde_json::json!({ "n": n }), &[n; 100])
    }

    #[test]
    fn mcap_gains_channels_in_place_and_keeps_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drive.mcap");
        let edge = dimos_recording::lcm::TfEdge { header: dimos_recording::lcm::Header::at(1.0, "odom"), child: "base_link".into(), translation: [1.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0] };
        mcap_io::write_fixture(&path, &[("/tf", "lcm", "tf2_msgs.TFMessage", 1.0, dimos_recording::lcm::encode_tf(&[edge]))]);
        {
            let mut writer = Writer::open(&path).unwrap();
            // a second writer is refused while the first holds the mcap
            assert!(Writer::open(&path).is_err());
            writer.put(Stream::Frames, 2.0, message(2));
            writer.put(Stream::Frames, 1.5, message(1));
            writer.flush().unwrap();
            writer.put(Stream::Patches(2), 1.5, message(9));
        } // the drop flushes the second batch
        let recording = Recording::open(&path).unwrap();
        let names: Vec<String> = recording.streams().unwrap().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["/memworld/frames", "/memworld/patches_so400m_naflex", "/tf"]);
        let frames = read(&recording, Stream::Frames).unwrap();
        assert_eq!(frames.iter().map(|f| f.0).collect::<Vec<_>>(), [1.5, 2.0]);
        assert_eq!(blob::decode(&frames[1].1).unwrap().0["n"], 2);
        assert_eq!(read(&recording, Stream::Depth).unwrap().len(), 0);
        // the original channel is untouched and the file is still a valid, indexed mcap
        let mut tf = 0;
        recording.raw_each("/tf", |_, _, _, _| {
            tf += 1;
            Ok(true)
        }).unwrap();
        assert_eq!(tf, 1);
        assert!(mcap::Summary::read(&std::fs::read(&path).unwrap()).unwrap().is_some());
    }

    #[test]
    fn db_gains_tables_dimos_can_list_and_loses_them_on_clear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drive.db");
        {
            let mut connection = db::Connection::open(&path).unwrap();
            db::write_stream(&mut connection, "odom_note", "std_msgs.String", &[(1.0, dimos_recording::lcm::encode_string("x"))]).unwrap();
        }
        {
            let mut writer = Writer::open(&path).unwrap();
            writer.put(Stream::Frames, 1.0, message(1));
            writer.put(Stream::Patches(0), 1.0, message(3));
            writer.flush().unwrap();
            assert!(writer.written() > 200);
        }
        let recording = Recording::open(&path).unwrap();
        let frames = read(&recording, Stream::Frames).unwrap();
        assert_eq!(blob::decode(&frames[0].1).unwrap().1, &[1u8; 100][..]);
        assert_eq!(read(&recording, Stream::Patches(0)).unwrap().len(), 1);
        drop(recording);
        drop_db_streams(&path).unwrap();
        let recording = Recording::open(&path).unwrap();
        let names: Vec<String> = recording.streams().unwrap().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["odom_note"]);
    }
}
