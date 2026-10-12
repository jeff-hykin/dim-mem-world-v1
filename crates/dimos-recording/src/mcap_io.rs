//! `.mcap` recordings: channels are either ROS 2 `cdr` (what the Live Viewer's recorder transcodes to, so Foxglove
//! opens them) or `lcm` (raw dimos LCM bytes, type in the channel's `lcm_type` metadata). Saving appends in place
//! (`append`): the summary is cut off, the new chunks go where it was and a summary covering both is put back, so a
//! save costs the bytes saved, never a copy of the recording.
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

pub const LCM_TYPE_KEY: &str = "lcm_type";
/// channel metadata naming the topic a channel was made from (a corrected copy of a lidar names the raw one)
pub const DERIVED_FROM_KEY: &str = "derived_from";

#[derive(Debug, Clone, PartialEq)]
pub struct McapChannel {
    pub topic: String,
    pub encoding: String,
    /// the schema name (`sensor_msgs/msg/PointCloud2`) for cdr, the dimos type (`sensor_msgs.PointCloud2`) for lcm
    pub kind: String,
    pub count: u64,
    pub derived_from: Option<String>,
}

pub struct McapFile {
    map: memmap2::Mmap,
}

impl McapFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Ok(McapFile { map: unsafe { memmap2::Mmap::map(&file) }? })
    }

    /// The compression of the file's chunks: the first chunk's ("" = none). A file with no chunk index (cut short)
    /// is taken as uncompressed, the Live Viewer recorder's default.
    pub fn compression(&self) -> Option<mcap::Compression> {
        let summary = mcap::Summary::read(&self.map).ok().flatten()?;
        match summary.chunk_indexes.first()?.compression.as_str() {
            "zstd" => Some(mcap::Compression::Zstd),
            "lz4" => Some(mcap::Compression::Lz4),
            _ => None,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn channels(&self) -> Result<Vec<McapChannel>> {
        let mut by_topic: BTreeMap<String, McapChannel> = BTreeMap::new();
        let summary = mcap::Summary::read(&self.map).ok().flatten();
        let mut add = |channel: &mcap::Channel, count: u64| {
            let kind = match channel.message_encoding.as_str() {
                "lcm" => channel.metadata.get(LCM_TYPE_KEY).cloned().unwrap_or_default(),
                _ => channel.schema.as_ref().map(|schema| schema.name.clone()).unwrap_or_default(),
            };
            let entry = by_topic.entry(channel.topic.clone()).or_insert(McapChannel {
                topic: channel.topic.clone(),
                encoding: channel.message_encoding.clone(),
                kind,
                count: 0,
                derived_from: channel.metadata.get(DERIVED_FROM_KEY).filter(|v| !v.is_empty()).cloned(),
            });
            entry.count += count;
        };
        match summary {
            Some(summary) if summary.stats.is_some() => {
                let counts = summary.stats.as_ref().map(|stats| stats.channel_message_counts.clone()).unwrap_or_default();
                for channel in summary.channels.values() {
                    add(channel, counts.get(&channel.id).copied().unwrap_or(0));
                }
            }
            _ => {
                for message in mcap::MessageStream::new(&self.map)? {
                    let Ok(message) = message else { break };
                    add(&message.channel, 1);
                }
            }
        }
        Ok(by_topic.into_values().collect())
    }

    /// Calls `each(log_time_seconds, encoding, kind, payload)` for every message on `topic`, in file order (log time
    /// for the Live Viewer's recordings). `each` returns false to stop.
    pub fn for_each(&self, topic: &str, mut each: impl FnMut(f64, &str, &str, &[u8]) -> Result<bool>) -> Result<()> {
        for message in mcap::MessageStream::new(&self.map)? {
            let message = match message {
                Ok(message) => message,
                // a recording cut short (no footer) still yields what was written
                Err(_) => break,
            };
            if message.channel.topic != topic {
                continue;
            }
            let kind = match message.channel.message_encoding.as_str() {
                "lcm" => message.channel.metadata.get(LCM_TYPE_KEY).cloned().unwrap_or_default(),
                _ => message.channel.schema.as_ref().map(|schema| schema.name.clone()).unwrap_or_default(),
            };
            if !each(message.log_time as f64 / 1e9, &message.channel.message_encoding, &kind, &message.data)? {
                break;
            }
        }
        Ok(())
    }

    /// Like `for_each`, but through the summary's chunk index: only chunks that hold `topic` are decompressed (an
    /// 80 GB recording with one small derived topic reads megabytes, not the file). Chunks come in start-time order;
    /// a file with no summary (cut short) falls back to the linear walk.
    pub fn for_each_indexed(&self, topic: &str, mut each: impl FnMut(f64, &str, &str, &[u8]) -> Result<bool>) -> Result<()> {
        let Some(summary) = mcap::Summary::read(&self.map).ok().flatten() else {
            return self.for_each(topic, each);
        };
        if summary.chunk_indexes.is_empty() {
            return self.for_each(topic, each);
        }
        let ids: Vec<u16> = summary.channels.values().filter(|c| c.topic == topic).map(|c| c.id).collect();
        if ids.is_empty() {
            return Ok(());
        }
        let mut chunks: Vec<&mcap::records::ChunkIndex> = summary
            .chunk_indexes
            .iter()
            .filter(|chunk| chunk.message_index_offsets.is_empty() || ids.iter().any(|id| chunk.message_index_offsets.contains_key(id)))
            .collect();
        chunks.sort_by_key(|chunk| (chunk.message_start_time, chunk.chunk_start_offset));
        for chunk in chunks {
            for message in summary.stream_chunk(&self.map, chunk)? {
                let message = message?;
                if message.channel.topic != topic {
                    continue;
                }
                let kind = match message.channel.message_encoding.as_str() {
                    "lcm" => message.channel.metadata.get(LCM_TYPE_KEY).cloned().unwrap_or_default(),
                    _ => message.channel.schema.as_ref().map(|schema| schema.name.clone()).unwrap_or_default(),
                };
                if !each(message.log_time as f64 / 1e9, &message.channel.message_encoding, &kind, &message.data)? {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Log times of every message on `topic`, from the chunk message indexes when the file has them.
    pub fn log_times(&self, topic: &str) -> Result<Vec<f64>> {
        let summary = mcap::Summary::read(&self.map).ok().flatten();
        let indexed = summary.as_ref().is_some_and(|s| !s.chunk_indexes.is_empty() && s.chunk_indexes.iter().all(|c| !c.message_index_offsets.is_empty()));
        if let (Some(summary), true) = (summary, indexed) {
            let ids: Vec<u16> = summary.channels.values().filter(|c| c.topic == topic).map(|c| c.id).collect();
            let mut times = Vec::new();
            for chunk in summary.chunk_indexes.iter().filter(|c| ids.iter().any(|id| c.message_index_offsets.contains_key(id))) {
                for (channel, entries) in summary.read_message_indexes(&self.map, chunk)? {
                    if channel.topic == topic {
                        times.extend(entries.iter().map(|e| e.log_time as f64 / 1e9));
                    }
                }
            }
            return Ok(times);
        }
        let mut times = Vec::new();
        self.for_each(topic, |ts, _, _, _| {
            times.push(ts);
            Ok(true)
        })?;
        Ok(times)
    }

    /// The newest message on `topic`.
    pub fn latest(&self, topic: &str) -> Result<Option<(f64, Vec<u8>)>> {
        let mut found = None;
        self.for_each_indexed(topic, |ts, _, _, payload| {
            if found.as_ref().is_none_or(|(at, _): &(f64, Vec<u8>)| ts >= *at) {
                found = Some((ts, payload.to_vec()));
            }
            Ok(true)
        })?;
        Ok(found)
    }
}

/// One new LCM channel to add: topic, dimos type, (log time, payload) messages.
pub struct NewChannel {
    pub topic: String,
    pub kind: String,
    pub messages: Vec<(f64, Vec<u8>)>,
}

/// Appends `channels` to the finished recording at `path` in place (mcap_append): nothing already in it is rewritten
/// or removed, so a reader of an earlier save's channels takes the newest message (`McapFile::latest`). A recording
/// with no summary (cut short) can't be appended to; it's refused rather than copied.
pub fn append(path: &Path, channels: &[NewChannel]) -> Result<()> {
    let mut appender = crate::mcap_append::Appender::open(path)?;
    for channel in channels {
        let mut metadata = BTreeMap::new();
        metadata.insert(LCM_TYPE_KEY.to_string(), channel.kind.clone());
        let id = appender.channel(&channel.topic, 0, "lcm", &metadata);
        for (ts, payload) in &channel.messages {
            appender.write(id, (ts.max(0.0) * 1e9) as u64, payload.clone())?;
        }
    }
    appender.finish()?;
    Ok(())
}

/// Writes a small mcap (topic, encoding, kind, time, payload): fixtures for tests here and in dependents.
/// Uncompressed chunks, like the Live Viewer's recorder by default.
pub fn write_fixture(path: &Path, messages: &[(&str, &str, &str, f64, Vec<u8>)]) {
    let mut writer = mcap::WriteOptions::new().compression(None).create(std::io::BufWriter::new(std::fs::File::create(path).unwrap())).unwrap();
    let mut channels: HashMap<String, u16> = HashMap::new();
    for (index, (topic, encoding, kind, ts, payload)) in messages.iter().enumerate() {
        let id = *channels.entry(topic.to_string()).or_insert_with(|| {
            if *encoding == "lcm" {
                let metadata = BTreeMap::from([(LCM_TYPE_KEY.to_string(), kind.to_string())]);
                writer.add_channel(0, topic, "lcm", &metadata).unwrap()
            } else {
                let schema = writer.add_schema(kind, "ros2msg", b"").unwrap();
                writer.add_channel(schema, topic, encoding, &BTreeMap::new()).unwrap()
            }
        });
        let time = (ts * 1e9) as u64;
        writer
            .write_to_known_channel(&mcap::records::MessageHeader { channel_id: id, sequence: index as u32, log_time: time, publish_time: time }, payload)
            .unwrap();
    }
    writer.finish().unwrap();
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_save_appends_in_place_and_the_newest_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.mcap");
        write_fixture(&path, &[("/lidar", "cdr", "sensor_msgs/msg/PointCloud2", 1.0, vec![1, 2]), ("/odd", "lcm", "x.Y", 2.0, vec![3])]);
        let before = std::fs::metadata(&path).unwrap().len();
        let save = |ts: f64, text: &[u8]| {
            let channel = NewChannel { topic: "/map_builder/annotations".into(), kind: "std_msgs.String".into(), messages: vec![(ts, text.to_vec())] };
            append(&path, &[channel]).unwrap();
        };
        save(5.0, b"first");
        save(6.0, b"second");
        let file = McapFile::open(&path).unwrap();
        let channels = file.channels().unwrap();
        let topics: Vec<_> = channels.iter().map(|c| (c.topic.as_str(), c.encoding.as_str(), c.kind.as_str(), c.count)).collect();
        assert_eq!(
            topics,
            [("/lidar", "cdr", "sensor_msgs/msg/PointCloud2", 1), ("/map_builder/annotations", "lcm", "std_msgs.String", 2), ("/odd", "lcm", "x.Y", 1)]
        );
        assert_eq!(file.latest("/map_builder/annotations").unwrap(), Some((6.0, b"second".to_vec())));
        assert_eq!(file.latest("/lidar").unwrap(), Some((1.0, vec![1, 2])));
        // grew by the saves, not by a copy; nothing left beside it
        assert!(std::fs::metadata(&path).unwrap().len() < before + 4096);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_recording_cut_short_is_refused_not_copied() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.mcap");
        write_fixture(&path, &[("/lidar", "cdr", "sensor_msgs/msg/PointCloud2", 1.0, vec![1, 2])]);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 40]).unwrap();
        let channel = NewChannel { topic: "/a".into(), kind: "std_msgs.String".into(), messages: vec![(1.0, vec![1])] };
        assert!(append(&path, &[channel]).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes[..bytes.len() - 40]);
    }
}
