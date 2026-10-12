//! Generation: every `every_seconds` of the color stream becomes a keyframe, and each keyframe is written INTO the
//! recording as a frame (camera pose from tf, intrinsics, a JPEG), its depth drawn into the color camera, and one
//! per-patch embedding grid per SigLIP2 member. Batches of keyframes are appended as they finish, so progress
//! survives a stop: running it again resumes the same generation where it left off.
use crate::blob;
use crate::camera::{register_depth, Pinhole};
use crate::compat::{self, Setup};
use crate::owl::Owl;
use crate::siglip::{Member, MEMBERS};
use crate::store::{self, Stream, Writer};
use crate::tf;
use anyhow::{bail, Context, Result};
use dimos_recording::image::{decode_image, Rgb};
use dimos_recording::Recording;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Options {
    /// one keyframe per this many seconds of the color stream
    pub every_seconds: f64,
    /// width of the stored frame JPEG (and its depth); height keeps the aspect ratio
    pub image_width: u32,
    /// depth readings farther than this are dropped
    pub max_depth_m: f32,
    /// keyframes per append
    pub batch: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { every_seconds: 1.0, image_width: 640, max_depth_m: 8.0, batch: 16 }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    /// planning, depth, embedding, done, cancelled, failed
    pub phase: String,
    pub done: u64,
    pub total: u64,
    /// keyframes with no camera pose at their time (no tf), left out
    pub skipped: u64,
    pub elapsed_s: f64,
    pub eta_s: Option<f64>,
    pub frames_per_s: Option<f64>,
    pub written_bytes: u64,
    pub generation: u64,
    pub message: String,
}

/// The keyframe times: the first color frame, then the first one at least `every` seconds after the last pick.
pub fn pick_keyframes(times: &[f64], every: f64) -> Vec<f64> {
    let mut picked: Vec<f64> = Vec::new();
    for &t in times {
        if picked.last().is_none_or(|last| t - last >= every - 1e-6) {
            picked.push(t);
        }
    }
    picked
}

/// For each wanted time, the nearest of `times` within `tolerance`.
pub fn nearest(times: &[f64], wanted: &[f64], tolerance: f64) -> Vec<Option<f64>> {
    wanted
        .iter()
        .map(|&w| {
            let i = times.partition_point(|&t| t < w);
            [i.checked_sub(1), Some(i)].into_iter().flatten().filter_map(|j| times.get(j).copied()).filter(|t| (t - w).abs() <= tolerance).min_by(|a, b| (a - w).abs().total_cmp(&(b - w).abs()))
        })
        .collect()
}

/// The newest generation's meta and the times it already holds, when its settings match `options` (to resume).
fn resumable(recording: &Recording, options: &Options, setup: &Setup) -> Result<Option<(u64, HashSet<u64>)>> {
    let metas = store::read(recording, Stream::Meta)?;
    let Some((_, last)) = metas.last() else { return Ok(None) };
    let (meta, _) = blob::decode(last)?;
    if meta["cleared"].as_bool().unwrap_or(false) || meta["color"] != json!(setup.color.name) {
        return Ok(None);
    }
    let theirs: Options = serde_json::from_value(meta["options"].clone()).unwrap_or_default();
    if theirs.every_seconds != options.every_seconds || theirs.image_width != options.image_width {
        return Ok(None);
    }
    let generation = meta["generation"].as_u64().context("meta without generation")?;
    let mut done = HashSet::new();
    let frames = store::read(recording, Stream::Frames)?;
    // a frame counts only once all its patch grids and its OWLv2 features are in (a batch is written whole, but be strict)
    let mut grids: HashMap<u64, usize> = HashMap::new();
    for i in 0..MEMBERS.len() {
        for (ts, bytes) in store::read(recording, Stream::Patches(i))? {
            if blob::decode(&bytes)?.0["generation"].as_u64() == Some(generation) {
                *grids.entry(key(ts)).or_default() += 1;
            }
        }
    }
    for (ts, bytes) in store::read(recording, Stream::Owl)? {
        if blob::decode(&bytes)?.0["generation"].as_u64() == Some(generation) {
            *grids.entry(key(ts)).or_default() += 1;
        }
    }
    for (ts, bytes) in frames {
        if blob::decode(&bytes)?.0["generation"].as_u64() == Some(generation) && grids.get(&key(ts)) == Some(&(MEMBERS.len() + 1)) {
            done.insert(key(ts));
        }
    }
    Ok(Some((generation, done)))
}

/// Times as integer microseconds, for exact set lookups.
pub fn key(ts: f64) -> u64 {
    (ts * 1e6).round() as u64
}

fn jpeg(rgb: &Rgb) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 88);
    encoder.encode(&rgb.data, rgb.width, rgb.height, image::ExtendedColorType::Rgb8)?;
    Ok(out)
}

pub fn generate(path: &Path, members: &[Member], owl: &Owl, options: &Options, report: &dyn Fn(&Progress), cancel: &AtomicBool) -> Result<Progress> {
    let started = Instant::now();
    let mut progress = Progress { phase: "planning".into(), message: "reading the recording".into(), ..Default::default() };
    report(&progress);
    if members.len() != MEMBERS.len() {
        bail!("generation needs all {} SigLIP2 members", MEMBERS.len());
    }
    let recording = Recording::open(path)?;
    let (check, setup) = compat::check(&recording)?;
    let setup = setup.with_context(|| format!("this recording can't be used: missing {}", check.missing.join("; ")))?;
    let times = recording.timestamps(&setup.color.name)?;
    let planned = pick_keyframes(&times, options.every_seconds);
    let (generation, done) = match resumable(&recording, options, &setup)? {
        Some(found) => found,
        None => (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis() as u64, HashSet::new()),
    };
    let todo: Vec<f64> = planned.iter().copied().filter(|t| !done.contains(&key(*t))).collect();
    progress.generation = generation;
    progress.total = planned.len() as u64;
    progress.done = (planned.len() - todo.len()) as u64;

    let height = (options.image_width as f64 * setup.color.height as f64 / setup.color.width as f64).round() as u32;
    let thumb_camera = setup.color_camera.scaled(options.image_width, height);
    let mut writer = Writer::open(path)?;
    if done.is_empty() {
        let meta = json!({
            "generation": generation,
            "created": generation,
            "options": options,
            "planned": planned.len(),
            "color": setup.color.name,
            "depth": setup.depth.name,
            "colorFrame": setup.color.frame_id,
            "worldFrame": setup.world_frame,
            "members": members.iter().map(|m| json!({ "key": m.key, "repo": m.spec.repo, "dim": m.spec.dim })).collect::<Vec<_>>(),
            "app": concat!("dim-mem-world ", env!("CARGO_PKG_VERSION")),
            "cleared": false,
        });
        let first = planned.first().copied().unwrap_or(0.0);
        writer.put(Stream::Meta, first, blob::encode(&meta, &[]));
        writer.flush()?;
    }
    // an mcap append moves its summary: read through a fresh map from here on (the passes below parse the summary once,
    // at their start, and then read only the data section, which appends never touch)
    let recording = Recording::open(path)?;

    // the lidar's voxel map (the scene answers are drawn in), once per generation
    if let Some(lidar) = setup.lidar.as_deref() {
        let has_map = store::read(&recording, Stream::Map)?.iter().any(|(_, b)| blob::decode(b).map(|(h, _)| h["generation"].as_u64() == Some(generation)).unwrap_or(false));
        if !has_map {
            progress.phase = "map".into();
            progress.message = format!("building the 3D map from {lidar}");
            report(&progress);
            let points = crate::lidar::build(&recording, lidar, &setup.tf, &setup.world_frame, &mut |done, total| {
                progress.message = format!("building the 3D map from {lidar}: {done} of {total} scans");
                report(&progress);
            }, cancel)?;
            let first = planned.first().copied().unwrap_or(0.0);
            let header = json!({ "generation": generation, "source": lidar, "voxel": crate::lidar::VOXEL_M, "count": points.len(), "frame": setup.world_frame });
            writer.put(Stream::Map, first, blob::encode(&header, &crate::lidar::encode(&points)));
            writer.flush()?;
        }
    }
    let recording = Recording::open(path)?;

    // depth first: each keyframe's nearest depth image, drawn into the color camera at the JPEG's size
    progress.phase = "depth".into();
    progress.message = format!("matching depth to {} keyframes", todo.len());
    report(&progress);
    let depth_times = recording.timestamps(&setup.depth.name)?;
    let matches = nearest(&depth_times, &todo, 0.1);
    let mut wanted: HashMap<u64, Vec<f64>> = HashMap::new();
    for (keyframe, depth_ts) in todo.iter().zip(&matches) {
        if let Some(depth_ts) = depth_ts {
            wanted.entry(key(*depth_ts)).or_default().push(*keyframe);
        }
    }
    let mut depths: HashMap<u64, Vec<u8>> = HashMap::new();
    let depth_camera: Pinhole = setup.depth_camera;
    recording.raw_each_where(
        &setup.depth.name,
        |ts| wanted.contains_key(&key(ts)),
        |ts, encoding, kind, payload| {
            let Ok(image) = decode_image(encoding, kind, payload) else { return Ok(true) };
            let Ok(depth) = image.to_depth() else { return Ok(true) };
            let registered = register_depth(&depth, &depth_camera, &thumb_camera, &setup.color_from_depth, options.max_depth_m);
            let raw: Vec<u8> = registered.iter().flat_map(|v| v.to_le_bytes()).collect();
            let packed = zstd::encode_all(&raw[..], 3)?;
            for keyframe in wanted.get(&key(ts)).into_iter().flatten() {
                depths.insert(key(*keyframe), packed.clone());
            }
            Ok(!cancel.load(Ordering::Relaxed))
        },
    )?;

    progress.phase = "embedding".into();
    progress.message = format!("embedding {} keyframes with {} models + OWLv2", todo.len(), members.len());
    report(&progress);
    let wanted: HashSet<u64> = todo.iter().map(|t| key(*t)).collect();
    let embed_started = Instant::now();
    let mut embedded = 0u64;
    let mut in_batch = 0usize;
    let mut failure: Option<anyhow::Error> = None;
    recording.raw_each_where(
        &setup.color.name,
        |ts| wanted.contains(&key(ts)),
        |ts, encoding, kind, payload| {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            let step = (|| -> Result<bool> {
                let Some(pose) = setup.tf.lookup(&setup.world_frame, &setup.color.frame_id, ts, 0.25) else {
                    return Ok(false);
                };
                let rgb = decode_image(encoding, kind, payload)?.to_rgb()?;
                let grids = members.iter().map(|m| m.embed_grid(&rgb)).collect::<Result<Vec<_>>>()?;
                let boxes = owl.features(&rgb)?;
                let thumb = crate::siglip::resize_rgb(&rgb, thumb_camera.width, thumb_camera.height);
                let (translation, rotation) = tf::parts(&pose);
                let depth = depths.remove(&key(ts));
                let shapes: BTreeMap<&str, [u32; 2]> = members.iter().zip(&grids).map(|(m, g)| (m.key.as_str(), [g.rows, g.cols])).collect();
                let frame = json!({
                    "generation": generation,
                    "pose": { "translation": translation, "rotation": rotation },
                    "camera": thumb_camera,
                    "grids": shapes,
                    "depth": depth.is_some(),
                });
                writer.put(Stream::Frames, ts, blob::encode(&frame, &jpeg(&thumb)?));
                if let Some(depth) = depth {
                    let header = json!({ "generation": generation, "width": thumb_camera.width, "height": thumb_camera.height, "unit": "mm", "codec": "zstd-u16le" });
                    writer.put(Stream::Depth, ts, blob::encode(&header, &depth));
                }
                writer.put(Stream::Owl, ts, blob::encode(&json!({ "generation": generation, "patches": crate::owl::PATCHES, "dim": crate::owl::DIM, "dtype": "f16" }), &boxes.encode()));
                for (i, grid) in grids.iter().enumerate() {
                    let header = json!({ "generation": generation, "rows": grid.rows, "cols": grid.cols, "dim": grid.dim, "dtype": "f16" });
                    writer.put(Stream::Patches(i), ts, blob::encode(&header, &blob::f16_bytes(&grid.data)));
                }
                Ok(true)
            })();
            match step {
                Ok(true) => {
                    embedded += 1;
                    progress.done += 1;
                    in_batch += 1;
                }
                Ok(false) => progress.skipped += 1,
                Err(error) => {
                    failure = Some(error);
                    return Ok(false);
                }
            }
            if in_batch >= options.batch {
                writer.flush()?;
                in_batch = 0;
            }
            let elapsed = embed_started.elapsed().as_secs_f64();
            progress.elapsed_s = started.elapsed().as_secs_f64();
            progress.written_bytes = writer.written();
            if embedded > 0 {
                let rate = embedded as f64 / elapsed;
                progress.frames_per_s = Some(rate);
                let remaining = progress.total.saturating_sub(progress.done + progress.skipped);
                progress.eta_s = Some(remaining as f64 / rate);
            }
            progress.message = format!("embedded {} of {} keyframes", progress.done, progress.total);
            report(&progress);
            Ok(true)
        },
    )?;
    writer.flush()?;
    progress.written_bytes = writer.written();
    progress.elapsed_s = started.elapsed().as_secs_f64();
    progress.eta_s = None;
    if let Some(error) = failure {
        progress.phase = "failed".into();
        progress.message = format!("{error:#}");
        report(&progress);
        return Err(error);
    }
    progress.phase = if cancel.load(Ordering::Relaxed) { "cancelled" } else { "done" }.into();
    progress.message = match progress.phase.as_str() {
        "done" => format!("{} keyframes in memory ({} without a camera pose left out)", progress.done, progress.skipped),
        _ => format!("stopped at {} of {}; Generate again to resume", progress.done, progress.total),
    };
    report(&progress);
    Ok(progress)
}

/// Forgets the memory: a `.db` drops its tables; an mcap gets a `cleared` meta message (it can't shrink in place).
pub fn clear(path: &Path) -> Result<()> {
    match dimos_recording::format_of(path)? {
        dimos_recording::Format::Db => store::drop_db_streams(path),
        dimos_recording::Format::Mcap => {
            let recording = Recording::open(path)?;
            let last = recording.timestamps(&Stream::Meta.name(dimos_recording::Format::Mcap)).unwrap_or_default().last().copied().unwrap_or(0.0);
            drop(recording);
            let mut writer = Writer::open(path)?;
            let generation = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis() as u64;
            writer.put(Stream::Meta, last, blob::encode(&json!({ "generation": generation, "cleared": true }), &[]));
            writer.flush()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframes_are_spaced_by_at_least_the_interval() {
        let times: Vec<f64> = (0..100).map(|i| i as f64 / 30.0).collect();
        let picked = pick_keyframes(&times, 1.0);
        assert_eq!(picked.len(), 4);
        assert!(picked.windows(2).all(|w| w[1] - w[0] >= 1.0 - 1e-6));
        assert_eq!(pick_keyframes(&[], 1.0), Vec::<f64>::new());
    }

    #[test]
    fn nearest_depth_is_within_tolerance_or_none() {
        let depth = [0.0, 0.033, 0.066, 0.5];
        assert_eq!(nearest(&depth, &[0.04, 0.3, 0.49, 2.0], 0.1), vec![Some(0.033), None, Some(0.5), None]);
    }
}
