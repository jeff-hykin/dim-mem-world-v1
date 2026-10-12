//! Can Memory World use this recording? It needs color images, the color camera's intrinsics (CameraInfo), a lidar
//! (PointCloud2), and poses: a tf path from a world frame (map / world / odom) to the camera and to the lidar. A raw
//! lidar + IMU recording with no poses yet (lite_record's) is usable too: generation runs lite_record's post_process
//! on it first, in place, which adds the odometry and the sensors' frame tree. `check` finds each input by what the
//! messages hold and says exactly what's missing.
use crate::camera::Pinhole;
use crate::store;
use crate::tf::{iso, TfTree};
use anyhow::Result;
use dimos_recording::image::{decode_camera_info, decode_image, CameraInfo};
use dimos_recording::{Format, Kind, Recording, StreamInfo};
use serde::Serialize;

pub const WORLD_FRAMES: [&str; 3] = ["map", "world", "odom"];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageStream {
    pub name: String,
    pub encoding: String,
    pub width: u32,
    pub height: u32,
    pub frame_id: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub path: String,
    pub format: &'static str,
    pub compatible: bool,
    /// one plain sentence per missing input
    pub missing: Vec<String>,
    pub color: Option<ImageStream>,
    pub color_info: Option<String>,
    /// the lidar the map is built from (its motion-corrected copy when post_process made one)
    pub lidar: Option<String>,
    pub lidar_frame: Option<String>,
    pub world_frame: Option<String>,
    /// set when generation must run lite_record's post_process first (why)
    pub post_process: Option<String>,
    pub start: f64,
    pub end: f64,
    pub duration_s: f64,
    pub notes: Vec<String>,
}

/// What generation reads, resolved.
#[derive(Clone)]
pub struct Setup {
    pub color: ImageStream,
    pub color_camera: Pinhole,
    pub lidar: String,
    pub world_frame: String,
    pub tf: TfTree,
}

fn first_message(recording: &Recording, stream: &str) -> Option<(String, String, Vec<u8>)> {
    let mut found = None;
    let _ = recording.raw_each(stream, |_, encoding, kind, payload| {
        found = Some((encoding.to_string(), kind.to_string(), payload.to_vec()));
        Ok(false)
    });
    found
}

fn rank_color(name: &str) -> i32 {
    let n = name.to_ascii_lowercase();
    if ["infra", "ir_", "_ir", "depth", "thumb", "mask"].iter().any(|bad| n.contains(bad)) {
        return -1;
    }
    if n.contains("color") {
        3
    } else if n.contains("rgb") {
        2
    } else {
        1
    }
}

pub fn tf_tree(recording: &Recording) -> Result<TfTree> {
    let mut tree = TfTree::default();
    for (_, edge) in recording.tf_edges_logged()? {
        tree.add(edge.header.ts(), &edge.header.frame_id, &edge.child, iso(edge.translation, edge.rotation));
    }
    tree.finish();
    Ok(tree)
}

/// lite_record's rule for a raw lidar recording: a PointCloud2 and an Imu under one topic prefix, and no odometry
/// anywhere to place its scans (the Map Editor runs the same check before its own post_process).
pub fn needs_post_process(streams: &[StreamInfo]) -> Option<String> {
    let prefix = |name: &str| name.rsplit_once('/').map_or(String::new(), |(prefix, _)| prefix.to_string());
    let is_imu = |s: &StreamInfo| s.type_name.replace("/msg/", ".").replace('/', ".") == "sensor_msgs.Imu";
    let lidar = streams
        .iter()
        .filter(|s| s.kind == Kind::Cloud && s.derived_from.is_none())
        .find(|cloud| streams.iter().any(|imu| is_imu(imu) && prefix(&imu.name) == prefix(&cloud.name)))?;
    if streams.iter().any(|s| s.kind == Kind::Odometry && s.count > 0) {
        return None;
    }
    Some(format!("{} has an IMU beside it but nothing places its scans yet: lite_record's post_process estimates the odometry and the sensors' frame tree first (in place)", lidar.name))
}

/// The lidar to map from: a motion-corrected copy (`derived_from` set, what post_process writes) over the raw scans,
/// then the busiest cloud.
fn pick_lidar(streams: &[StreamInfo]) -> Option<&StreamInfo> {
    streams
        .iter()
        .filter(|s| s.kind == Kind::Cloud && !store::is_ours(&s.name) && s.count > 0)
        .max_by_key(|s| (s.derived_from.is_some(), s.name.to_ascii_lowercase().contains("lidar"), s.count))
}

pub fn check(recording: &Recording) -> Result<(Check, Option<Setup>)> {
    let streams = recording.streams()?;
    let format = recording.format();
    let mut missing = Vec::new();
    let mut notes = Vec::new();

    let mut colors = Vec::new();
    for stream in streams.iter().filter(|s| matches!(s.kind, Kind::Image | Kind::CompressedImage) && !store::is_ours(&s.name) && s.count > 0 && rank_color(&s.name) >= 0) {
        let Some((encoding, kind, payload)) = first_message(recording, &stream.name) else { continue };
        let Ok(image) = decode_image(&encoding, &kind.replace('.', "/"), &payload) else {
            notes.push(format!("{}: couldn't read its first image", stream.name));
            continue;
        };
        if !image.is_depth() && image.to_rgb().is_ok() {
            colors.push(ImageStream { name: stream.name.clone(), encoding: image.encoding.clone(), width: image.width, height: image.height, frame_id: image.header.frame_id.trim_start_matches('/').into(), count: stream.count });
        }
    }
    colors.sort_by_key(|c| (-rank_color(&c.name), std::cmp::Reverse(c.count)));
    let color = colors.into_iter().next();
    if color.is_none() {
        missing.push("color images (a sensor_msgs Image or CompressedImage stream)".into());
    }

    let mut infos: Vec<(String, CameraInfo)> = Vec::new();
    for stream in streams.iter().filter(|s| s.kind == Kind::CameraInfo) {
        if let Some((encoding, _, payload)) = first_message(recording, &stream.name) {
            if let Ok(info) = decode_camera_info(&encoding, &payload) {
                infos.push((stream.name.clone(), info));
            }
        }
    }
    let color_info = color.as_ref().and_then(|c| {
        infos
            .iter()
            .find(|(_, info)| !c.frame_id.is_empty() && info.header.frame_id.trim_start_matches('/') == c.frame_id)
            .or_else(|| infos.iter().find(|(name, _)| name.to_ascii_lowercase().contains("color")))
            .or_else(|| infos.iter().find(|(name, _)| !name.to_ascii_lowercase().contains("depth")))
            .cloned()
    });
    if let (Some(c), None) = (&color, &color_info) {
        missing.push(format!("the camera's intrinsics (a CameraInfo stream for frame {})", c.frame_id));
    }

    let lidar = pick_lidar(&streams);
    let lidar_frame = lidar.and_then(|l| {
        let mut frame = None;
        let _ = recording.for_each(&l.name, |_, message, _| {
            if let dimos_recording::Message::Cloud(cloud) = message {
                frame = Some(cloud.frame_id.trim_start_matches('/').to_string());
            }
            Ok(false)
        });
        frame
    });
    if lidar.is_none() {
        missing.push("a lidar (a PointCloud2 stream): the map, and where things are, come from it".into());
    }

    let post_process = needs_post_process(&streams);
    let tf = tf_tree(recording)?;
    let times = color.as_ref().map(|c| recording.timestamps(&c.name)).transpose()?.unwrap_or_default();
    let (start, end) = (times.first().copied().unwrap_or(0.0), times.last().copied().unwrap_or(0.0));
    let middle = times.get(times.len() / 2).copied().unwrap_or(0.0);
    let mut world_frame = None;
    if let Some(reason) = &post_process {
        notes.push(reason.clone());
    } else if let (Some(c), Some(lf)) = (&color, &lidar_frame) {
        let reaches = |w: &&str| tf.lookup(w, &c.frame_id, middle, 0.5).is_some() && tf.lookup(w, lf, middle, 0.5).is_some();
        world_frame = WORLD_FRAMES.iter().find(|w| reaches(w)).map(|w| w.to_string());
        if tf.is_empty() {
            missing.push("poses: the recording has no tf (and no lidar + IMU for lite_record to estimate them from)".into());
        } else if world_frame.is_none() {
            let to_lidar = WORLD_FRAMES.iter().find(|w| tf.lookup(w, lf, middle, 0.5).is_some());
            missing.push(match to_lidar {
                // the lidar is placed but the camera hangs off a separate tree: the rig's camera <-> lidar mount isn't recorded
                Some(world) => format!("where the camera sits on the robot: the lidar ({lf}) has poses in {world}, but no tf joins the camera ({}) to it (the camera-lidar mount isn't in this recording's tf)", c.frame_id),
                None => format!("poses: no tf path from map, world or odom to the camera ({}) or the lidar ({lf})", c.frame_id),
            });
        }
    }
    let compatible = missing.is_empty();
    let setup = match (compatible, post_process.is_none(), &color, &color_info, lidar, &world_frame) {
        (true, true, Some(c), Some((_, info)), Some(l), Some(world)) => Some(Setup {
            color: c.clone(),
            color_camera: Pinhole::from_info(info).scaled(c.width, c.height),
            lidar: l.name.clone(),
            world_frame: world.clone(),
            tf,
        }),
        _ => None,
    };
    let check = Check {
        path: recording.path.display().to_string(),
        format: if format == Format::Db { "db" } else { "mcap" },
        compatible,
        missing,
        color,
        color_info: color_info.map(|c| c.0),
        lidar: lidar.map(|l| l.name.clone()),
        lidar_frame,
        world_frame,
        post_process,
        start,
        end,
        duration_s: end - start,
        notes,
    };
    Ok((check, setup))
}
