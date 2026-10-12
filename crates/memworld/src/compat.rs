//! Can Memory World use this recording? It needs color images, depth images, both cameras' intrinsics (CameraInfo)
//! and a tf path from a world frame (map / world / odom) to the color camera. `check` finds each one (by what the
//! messages hold, not only by name) and says exactly which are missing.
use crate::camera::Pinhole;
use crate::store;
use crate::tf::{iso, Iso, TfTree};
use anyhow::Result;
use dimos_recording::image::{decode_camera_info, decode_image, CameraInfo, StoredImage};
use dimos_recording::{Format, Kind, Recording};
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
    pub depth: Option<ImageStream>,
    pub color_info: Option<String>,
    pub depth_info: Option<String>,
    pub world_frame: Option<String>,
    /// a lidar (PointCloud2) stream: the scene answers are drawn in (optional)
    pub lidar: Option<String>,
    pub start: f64,
    pub end: f64,
    pub duration_s: f64,
    pub notes: Vec<String>,
}

/// What generation reads, resolved.
#[derive(Clone)]
pub struct Setup {
    pub color: ImageStream,
    pub depth: ImageStream,
    pub color_camera: Pinhole,
    pub depth_camera: Pinhole,
    pub world_frame: String,
    pub color_from_depth: Iso,
    pub tf: TfTree,
    pub lidar: Option<String>,
}

fn first_message(recording: &Recording, stream: &str) -> Option<(String, String, Vec<u8>)> {
    let mut found = None;
    let _ = recording.raw_each(stream, |_, encoding, kind, payload| {
        found = Some((encoding.to_string(), kind.to_string(), payload.to_vec()));
        Ok(false)
    });
    found
}

fn image_stream(recording: &Recording, name: &str, count: u64) -> Option<(ImageStream, StoredImage)> {
    let (encoding, kind, payload) = first_message(recording, name)?;
    let image = decode_image(&encoding, &kind.replace('.', "/"), &payload).ok()?;
    Some((ImageStream { name: name.into(), encoding: image.encoding.clone(), width: image.width, height: image.height, frame_id: image.header.frame_id.trim_start_matches('/').into(), count }, image))
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

fn rank_depth(name: &str) -> i32 {
    let n = name.to_ascii_lowercase();
    if n.contains("thumb") || n.contains("filled") || n.contains("colorized") {
        return -1;
    }
    if n.contains("depth") {
        2
    } else {
        1
    }
}

/// A stored image's frame size can differ from its CameraInfo's (a resized stream): intrinsics follow the image.
fn camera_for(info: &CameraInfo, stream: &ImageStream) -> Pinhole {
    Pinhole::from_info(info).scaled(stream.width, stream.height)
}

pub fn tf_tree(recording: &Recording) -> Result<TfTree> {
    let mut tree = TfTree::default();
    for (_, edge) in recording.tf_edges_logged()? {
        tree.add(edge.header.ts(), &edge.header.frame_id, &edge.child, iso(edge.translation, edge.rotation));
    }
    tree.finish();
    Ok(tree)
}

pub fn check(recording: &Recording) -> Result<(Check, Option<Setup>)> {
    let streams = recording.streams()?;
    let format = recording.format();
    let mut missing = Vec::new();
    let mut notes = Vec::new();
    let mut colors = Vec::new();
    let mut depths = Vec::new();
    for stream in streams.iter().filter(|s| matches!(s.kind, Kind::Image | Kind::CompressedImage) && !store::is_ours(&s.name) && s.count > 0) {
        let Some((info, image)) = image_stream(recording, &stream.name, stream.count) else {
            notes.push(format!("{}: couldn't read its first image", stream.name));
            continue;
        };
        if image.is_depth() {
            depths.push(info);
        } else if image.to_rgb().is_ok() {
            colors.push(info);
        } else {
            notes.push(format!("{}: {} images aren't color or depth", stream.name, image.encoding));
        }
    }
    colors.retain(|c| rank_color(&c.name) >= 0);
    colors.sort_by_key(|c| (-rank_color(&c.name), std::cmp::Reverse(c.count)));
    depths.retain(|d| rank_depth(&d.name) >= 0);
    depths.sort_by_key(|d| (-rank_depth(&d.name), std::cmp::Reverse(d.count)));
    let color = colors.into_iter().next();
    let depth = depths.into_iter().next();
    if color.is_none() {
        missing.push("color images (a sensor_msgs Image or CompressedImage stream)".into());
    }
    if depth.is_none() {
        missing.push("depth images (a 16UC1 / 32FC1 Image stream)".into());
    }

    let mut infos: Vec<(String, CameraInfo)> = Vec::new();
    for stream in streams.iter().filter(|s| s.kind == Kind::CameraInfo) {
        if let Some((encoding, _, payload)) = first_message(recording, &stream.name) {
            if let Ok(info) = decode_camera_info(&encoding, &payload) {
                infos.push((stream.name.clone(), info));
            }
        }
    }
    let info_for = |image: &ImageStream, hint: &str| -> Option<(String, CameraInfo)> {
        infos
            .iter()
            .find(|(_, info)| !image.frame_id.is_empty() && info.header.frame_id.trim_start_matches('/') == image.frame_id)
            .or_else(|| infos.iter().find(|(name, _)| name.to_ascii_lowercase().contains(hint)))
            .cloned()
    };
    let color_info = color.as_ref().and_then(|c| info_for(c, "color").or_else(|| infos.iter().find(|(n, _)| !n.contains("depth")).cloned()));
    let depth_info = depth.as_ref().and_then(|d| info_for(d, "depth"));
    if let (Some(c), None) = (&color, &color_info) {
        missing.push(format!("the color camera's intrinsics (a CameraInfo stream for frame {})", c.frame_id));
    }
    if let (Some(d), None) = (&depth, &depth_info) {
        missing.push(format!("the depth camera's intrinsics (a CameraInfo stream for frame {})", d.frame_id));
    }

    let tf = tf_tree(recording)?;
    let times = color.as_ref().map(|c| recording.timestamps(&c.name)).transpose()?.unwrap_or_default();
    let (start, end) = (times.first().copied().unwrap_or(0.0), times.last().copied().unwrap_or(0.0));
    let middle = times.get(times.len() / 2).copied().unwrap_or(0.0);
    let mut world_frame = None;
    if let Some(c) = &color {
        world_frame = WORLD_FRAMES.iter().find(|w| tf.lookup(w, &c.frame_id, middle, 0.5).is_some()).map(|w| w.to_string());
        if tf.is_empty() {
            missing.push("a camera pose: the recording has no tf".into());
        } else if world_frame.is_none() {
            missing.push(format!("a camera pose: no tf path from map, world or odom to {}", c.frame_id));
        }
    }
    let color_from_depth = match (&color, &depth) {
        (Some(c), Some(d)) if c.frame_id == d.frame_id => Some(Iso::identity()),
        (Some(c), Some(d)) => {
            let found = tf.lookup(&c.frame_id, &d.frame_id, middle, 0.5);
            if found.is_none() {
                missing.push(format!("where the depth camera sits: no tf between {} and {}", d.frame_id, c.frame_id));
            }
            found
        }
        _ => None,
    };
    let lidar = streams.iter().filter(|s| s.kind == Kind::Cloud && !store::is_ours(&s.name) && s.count > 0).max_by_key(|s| (s.name.to_ascii_lowercase().contains("lidar"), s.count)).map(|s| s.name.clone());
    if lidar.is_none() {
        notes.push("no lidar: the 3D scene is drawn from the keyframes' depth".into());
    }
    let compatible = missing.is_empty();
    let setup = match (compatible, &color, &depth, &color_info, &depth_info, &world_frame, color_from_depth) {
        (true, Some(c), Some(d), Some((_, ci)), Some((_, di)), Some(world), Some(extrinsic)) => Some(Setup {
            color: c.clone(),
            depth: d.clone(),
            color_camera: camera_for(ci, c),
            depth_camera: camera_for(di, d),
            world_frame: world.clone(),
            color_from_depth: extrinsic,
            tf,
            lidar: lidar.clone(),
        }),
        _ => None,
    };
    let check = Check {
        path: recording.path.display().to_string(),
        format: if format == Format::Db { "db" } else { "mcap" },
        compatible,
        missing,
        color,
        depth,
        color_info: color_info.map(|c| c.0),
        depth_info: depth_info.map(|d| d.0),
        world_frame,
        lidar,
        start,
        end,
        duration_s: end - start,
        notes,
    };
    Ok((check, setup))
}
