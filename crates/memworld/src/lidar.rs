//! The scene answers are drawn in: when the recording has a lidar (a PointCloud2 stream with tf to the world frame),
//! every scan accumulated into a 5 cm voxel map, keeping voxels hit at least twice (one-off returns are noise). Built
//! once at generation and stored in the recording; recordings without a lidar fall back to the keyframes' depth.
use crate::tf::TfTree;
use anyhow::Result;
use dimos_recording::{Message, Recording};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

pub const VOXEL_M: f64 = 0.05;
const MIN_HITS: u32 = 2;
const MIN_RANGE_M: f32 = 0.4;
const MAX_RANGE_M: f32 = 30.0;
pub const MAX_POINTS: usize = 3_000_000;

/// `stream`'s scans in `world`, voxelized. `report(done, total)` is called as scans go by.
pub fn build(recording: &Recording, stream: &str, tf: &TfTree, world: &str, report: &mut dyn FnMut(u64, u64), cancel: &AtomicBool) -> Result<Vec<[f32; 3]>> {
    let total = recording.timestamps(stream)?.len() as u64;
    let mut hits: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut done = 0u64;
    recording.for_each(stream, |ts, message, _| {
        done += 1;
        if done % 50 == 0 {
            report(done, total);
        }
        let Message::Cloud(cloud) = message else { return Ok(true) };
        let Some(pose) = tf.lookup(world, &cloud.frame_id, ts, 0.25) else { return Ok(true) };
        for p in &cloud.points {
            let range = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if !(MIN_RANGE_M..=MAX_RANGE_M).contains(&range) {
                continue;
            }
            let w = pose * nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            *hits.entry(((w.x / VOXEL_M).floor() as i32, (w.y / VOXEL_M).floor() as i32, (w.z / VOXEL_M).floor() as i32)).or_default() += 1;
        }
        Ok(!cancel.load(Ordering::Relaxed))
    })?;
    report(done, total);
    let mut points: Vec<[f32; 3]> = hits
        .into_iter()
        .filter(|(_, n)| *n >= MIN_HITS)
        .map(|((x, y, z), _)| [((x as f64 + 0.5) * VOXEL_M) as f32, ((y as f64 + 0.5) * VOXEL_M) as f32, ((z as f64 + 0.5) * VOXEL_M) as f32])
        .collect();
    points.sort_by(|a, b| a[2].total_cmp(&b[2]));
    points.truncate(MAX_POINTS);
    Ok(points)
}

pub fn encode(points: &[[f32; 3]]) -> Vec<u8> {
    points.iter().flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes())).collect()
}

pub fn decode(bytes: &[u8]) -> Vec<[f32; 3]> {
    bytes.chunks_exact(12).map(|b| std::array::from_fn(|i| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))).collect()
}

/// The map without its ceiling, so it can be seen into from above: everything more than `above_floor_m` over the floor
/// (taken as the 5th percentile of height; most of a map's points are floor) is left out.
pub fn without_ceiling(points: &[[f32; 3]], above_floor_m: f32) -> Vec<[f32; 3]> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut z: Vec<f32> = points.iter().map(|p| p[2]).collect();
    z.sort_by(f32::total_cmp);
    let floor = z[z.len() / 20];
    points.iter().copied().filter(|p| p[2] <= floor + above_floor_m).collect()
}

/// The Map Editor's height palette (purple, blue, cyan, green), stretched over the map's own height range (its 2nd to
/// 98th percentile), so the floor and the ceiling read apart in any building.
pub const PALETTE: [[u8; 3]; 4] = [[0xa0, 0x5c, 0xf0], [0x5a, 0x86, 0xf2], [0x3e, 0xd4, 0xe0], [0x6e, 0xe6, 0x8c]];

pub fn height_colors(points: &[[f32; 3]]) -> Vec<[u8; 3]> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut z: Vec<f32> = points.iter().map(|p| p[2]).collect();
    z.sort_by(f32::total_cmp);
    let (low, high) = (z[z.len() / 50], z[z.len() * 49 / 50]);
    let span = (high - low).max(0.5);
    points
        .iter()
        .map(|p| {
            let t = ((p[2] - low) / span).clamp(0.0, 1.0) * (PALETTE.len() - 1) as f32;
            let i = (t.floor() as usize).min(PALETTE.len() - 2);
            let f = t - i as f32;
            std::array::from_fn(|c| (PALETTE[i][c] as f32 * (1.0 - f) + PALETTE[i + 1][c] as f32 * f).round() as u8)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_round_trip_and_heights_span_the_palette() {
        let points: Vec<[f32; 3]> = (0..100).map(|i| [i as f32, 0.0, i as f32 / 33.0]).collect();
        assert_eq!(decode(&encode(&points)), points);
        let colors = height_colors(&points);
        assert_eq!(colors[0], PALETTE[0]);
        assert_eq!(colors[99], PALETTE[3]);
        assert!(height_colors(&[]).is_empty());
    }

    #[test]
    fn the_ceiling_is_cut_above_the_floor() {
        let mut points: Vec<[f32; 3]> = (0..100).map(|i| [i as f32, 0.0, 0.0]).collect();
        points.extend((0..20).map(|i| [i as f32, 0.0, 1.0]));
        points.extend((0..50).map(|i| [i as f32, 0.0, 4.0]));
        assert_eq!(without_ceiling(&points, 2.2).len(), 120);
        assert!(without_ceiling(&[], 2.2).is_empty());
    }

    #[test]
    fn a_wall_seen_twice_stays_and_a_single_return_goes() {
        use crate::tf::iso;
        use dimos_recording::lcm::{encode_xyz_cloud, encode_tf, Header, TfEdge};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drive.mcap");
        let wall: Vec<[f32; 3]> = (0..20).map(|i| [2.0, (i as f32 + 0.5) * 0.05, 0.0]).collect();
        let mut stray = wall.clone();
        stray.push([5.0, 5.0, 5.0]);
        let edge = |t: f64| TfEdge { header: Header::at(t, "odom"), child: "lidar".into(), translation: [1.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0] };
        dimos_recording::mcap_io::write_fixture(
            &path,
            &[
                ("/tf", "lcm", "tf2_msgs.TFMessage", 1.0, encode_tf(&[edge(1.0)])),
                ("/tf", "lcm", "tf2_msgs.TFMessage", 2.0, encode_tf(&[edge(2.0)])),
                ("/lidar", "lcm", "sensor_msgs.PointCloud2", 1.0, encode_xyz_cloud(&Header::at(1.0, "lidar"), &wall, None)),
                ("/lidar", "lcm", "sensor_msgs.PointCloud2", 2.0, encode_xyz_cloud(&Header::at(2.0, "lidar"), &stray, None)),
            ],
        );
        let recording = Recording::open(&path).unwrap();
        let mut tree = TfTree::default();
        tree.add(1.0, "odom", "lidar", iso([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]));
        tree.finish();
        let points = build(&recording, "/lidar", &tree, "odom", &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!(points.len(), 20);
        // moved by the tf: the wall at x = 2 in the lidar is x = 3 in odom (voxel centers)
        assert!(points.iter().all(|p| (p[0] - 3.0).abs() < 0.05), "{points:?}");
    }
}
