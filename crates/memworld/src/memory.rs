//! A recording's memory, read back from inside it and held in RAM for questions: every keyframe (pose, intrinsics,
//! JPEG, depth) and, per member, every patch embedding as one matrix (a question is one matrix-vector product per
//! member, a fraction of a second, no database round trips).
use crate::blob;
use crate::camera::Pinhole;
use crate::generate::key;
use crate::siglip::MEMBERS;
use crate::store::{self, Stream};
use crate::tf::{iso, Iso};
use anyhow::{Context, Result};
use dimos_recording::Recording;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

pub struct Frame {
    pub ts: f64,
    /// world <- camera (optical)
    pub pose: Iso,
    pub camera: Pinhole,
    pub jpeg: Vec<u8>,
    /// zstd of u16 LE millimeters at the camera's size; empty when the keyframe had no depth
    pub depth: Vec<u8>,
    /// OWLv2's image half for this keyframe (fp16, owl::Features::decode); empty for a memory made without it
    pub owl: Vec<u8>,
}

impl Frame {
    pub fn depth_mm(&self) -> Option<Vec<u16>> {
        if self.depth.is_empty() {
            return None;
        }
        let raw = zstd::decode_all(&self.depth[..]).ok()?;
        Some(raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    pub fn position(&self) -> [f64; 3] {
        let t = self.pose.translation.vector;
        [t.x, t.y, t.z]
    }
}

/// One member's patches across all keyframes: `data` is (patches, dim), `cells[i]` says whose patch row i is.
pub struct MemberIndex {
    pub key: String,
    pub dim: usize,
    pub data: Vec<f32>,
    /// (frame index, row, col) per patch
    pub cells: Vec<(u32, u16, u16)>,
    /// per frame: (first patch, rows, cols)
    pub grids: Vec<(usize, u16, u16)>,
}

impl MemberIndex {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Every patch's dot product with `query`.
    pub fn scores(&self, query: &[f32]) -> Vec<f32> {
        self.data.par_chunks(self.dim * 4096).flat_map_iter(|block| block.chunks_exact(self.dim).map(|row| dot(row, query)).collect::<Vec<_>>()).collect()
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        for lane in 0..8 {
            sums[lane] += a[i * 8 + lane] * b[i * 8 + lane];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for i in chunks * 8..a.len() {
        total += a[i] * b[i];
    }
    total
}

pub struct Memory {
    pub generation: u64,
    pub meta: Value,
    pub frames: Vec<Frame>,
    pub members: Vec<MemberIndex>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// none, partial (generation stopped part way), ready
    pub state: &'static str,
    pub generation: Option<u64>,
    pub frames: usize,
    pub planned: usize,
    pub options: Value,
    pub created_ms: Option<u64>,
}

/// The newest generation's meta, unless it's a clear.
fn current_meta(recording: &Recording) -> Result<Option<Value>> {
    let metas = store::read(recording, Stream::Meta)?;
    let Some((_, bytes)) = metas.last() else { return Ok(None) };
    let (meta, _) = blob::decode(bytes)?;
    Ok((!meta["cleared"].as_bool().unwrap_or(false)).then_some(meta))
}

/// What's in the recording, cheaply (frames are counted from the frame stream only).
pub fn status(recording: &Recording) -> Result<Status> {
    let Some(meta) = current_meta(recording)? else {
        return Ok(Status { state: "none", generation: None, frames: 0, planned: 0, options: Value::Null, created_ms: None });
    };
    let generation = meta["generation"].as_u64();
    let frames = store::read(recording, Stream::Frames)?.iter().filter(|(_, b)| blob::decode(b).map(|(h, _)| h["generation"].as_u64() == generation).unwrap_or(false)).count();
    let planned = meta["planned"].as_u64().unwrap_or(0) as usize;
    Ok(Status {
        state: if frames == 0 { "none" } else if frames < planned { "partial" } else { "ready" },
        generation,
        frames,
        planned,
        options: meta["options"].clone(),
        created_ms: meta["created"].as_u64(),
    })
}

fn pose_of(header: &Value) -> Option<Iso> {
    let t: Vec<f64> = header["pose"]["translation"].as_array()?.iter().filter_map(Value::as_f64).collect();
    let r: Vec<f64> = header["pose"]["rotation"].as_array()?.iter().filter_map(Value::as_f64).collect();
    (t.len() == 3 && r.len() == 4).then(|| iso([t[0], t[1], t[2]], [r[0], r[1], r[2], r[3]]))
}

/// Everything of the current generation; None when the recording has no memory.
pub fn load(recording: &Recording) -> Result<Option<Memory>> {
    let Some(meta) = current_meta(recording)? else { return Ok(None) };
    let generation = meta["generation"].as_u64().context("meta without generation")?;
    let mine = |header: &Value| header["generation"].as_u64() == Some(generation);
    let mut depth: HashMap<u64, Vec<u8>> = HashMap::new();
    for (ts, bytes) in store::read(recording, Stream::Depth)? {
        let (header, body) = blob::decode(&bytes)?;
        if mine(&header) {
            depth.insert(key(ts), body.to_vec());
        }
    }
    let mut owl: HashMap<u64, Vec<u8>> = HashMap::new();
    for (ts, bytes) in store::read(recording, Stream::Owl)? {
        let (header, body) = blob::decode(&bytes)?;
        if mine(&header) {
            owl.insert(key(ts), body.to_vec());
        }
    }
    let mut grids_by_member: Vec<HashMap<u64, (u16, u16, usize, Vec<f32>)>> = Vec::new();
    for i in 0..MEMBERS.len() {
        let mut grids = HashMap::new();
        for (ts, bytes) in store::read(recording, Stream::Patches(i))? {
            let (header, body) = blob::decode(&bytes)?;
            if mine(&header) {
                let rows = header["rows"].as_u64().unwrap_or(0) as u16;
                let cols = header["cols"].as_u64().unwrap_or(0) as u16;
                let dim = header["dim"].as_u64().unwrap_or(0) as usize;
                grids.insert(key(ts), (rows, cols, dim, blob::f16_values(body)));
            }
        }
        grids_by_member.push(grids);
    }
    let mut frames = Vec::new();
    for (ts, bytes) in store::read(recording, Stream::Frames)? {
        let (header, body) = blob::decode(&bytes)?;
        // a frame is usable once every member's grid is in
        if !mine(&header) || grids_by_member.iter().any(|g| !g.contains_key(&key(ts))) {
            continue;
        }
        let camera: Pinhole = serde_json::from_value(header["camera"].clone())?;
        let pose = pose_of(&header).context("frame without a pose")?;
        frames.push(Frame { ts, pose, camera, jpeg: body.to_vec(), depth: depth.remove(&key(ts)).unwrap_or_default(), owl: owl.remove(&key(ts)).unwrap_or_default() });
    }
    frames.sort_by(|a, b| a.ts.total_cmp(&b.ts));
    let mut members = Vec::new();
    for (i, mut grids) in grids_by_member.into_iter().enumerate() {
        let mut index = MemberIndex { key: MEMBERS[i].into(), dim: 0, data: Vec::new(), cells: Vec::new(), grids: Vec::new() };
        for (f, frame) in frames.iter().enumerate() {
            let (rows, cols, dim, values) = grids.remove(&key(frame.ts)).expect("checked above");
            index.dim = dim;
            index.grids.push((index.cells.len(), rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    index.cells.push((f as u32, r, c));
                }
            }
            index.data.extend(values);
        }
        members.push(index);
    }
    Ok(Some(Memory { generation, meta, frames, members }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_handles_tails() {
        let a: Vec<f32> = (0..13).map(|v| v as f32).collect();
        let b = vec![1.0f32; 13];
        assert_eq!(dot(&a, &b), 78.0);
    }
}
