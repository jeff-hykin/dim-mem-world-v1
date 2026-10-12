//! Pinhole geometry: the lidar map drawn into a keyframe's camera (so a box in the image has the lidar's depth behind
//! it), and a box's depth turned into world points. Camera frames are optical (+z forward, +x right, +y down).
use crate::tf::Iso;
use dimos_recording::image::CameraInfo;
use nalgebra::Point3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pinhole {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl Pinhole {
    pub fn from_info(info: &CameraInfo) -> Pinhole {
        Pinhole { fx: info.fx(), fy: info.fy(), cx: info.cx(), cy: info.cy(), width: info.width, height: info.height }
    }

    /// The same camera at another resolution.
    pub fn scaled(&self, width: u32, height: u32) -> Pinhole {
        let sx = width as f64 / self.width.max(1) as f64;
        let sy = height as f64 / self.height.max(1) as f64;
        Pinhole { fx: self.fx * sx, fy: self.fy * sy, cx: self.cx * sx, cy: self.cy * sy, width, height }
    }

    pub fn project(&self, p: &Point3<f64>) -> Option<(f64, f64)> {
        if p.z <= 1e-6 {
            return None;
        }
        Some((self.fx * p.x / p.z + self.cx, self.fy * p.y / p.z + self.cy))
    }

    /// The camera-frame point at pixel (u, v) (pixel centers at +0.5) and depth z.
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> Point3<f64> {
        Point3::new((u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z)
    }
}

/// The lidar map as this camera would see it: every map voxel (world frame) moved into the camera by
/// `world_from_camera`'s inverse and projected, drawn as a square of its own projected size so near surfaces cover what
/// is behind them (a z-buffer: nearest wins). Millimeters, 0 = nothing there. Points nearer than `min_m` or farther
/// than `max_m` are left out.
pub fn render_depth(points: &[[f32; 3]], voxel_m: f64, camera: &Pinhole, world_from_camera: &Iso, min_m: f64, max_m: f64) -> Vec<u16> {
    let (w, h) = (camera.width as i64, camera.height as i64);
    let mut out = vec![u16::MAX; (w * h) as usize];
    let camera_from_world = world_from_camera.inverse();
    for p in points {
        let q = camera_from_world * Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        if q.z < min_m || q.z > max_m {
            continue;
        }
        let Some((u, v)) = camera.project(&q) else { continue };
        let half = ((camera.fx * voxel_m / q.z) / 2.0).ceil().max(1.0) as i64;
        let (cu, cv) = (u.floor() as i64, v.floor() as i64);
        if cu + half < 0 || cv + half < 0 || cu - half >= w || cv - half >= h {
            continue;
        }
        let mm = (q.z * 1000.0).round().clamp(1.0, 65534.0) as u16;
        for y in (cv - half).max(0)..(cv + half).min(h) {
            let row = (y * w) as usize;
            for x in (cu - half).max(0)..(cu + half).min(w) {
                let at = row + x as usize;
                out[at] = out[at].min(mm);
            }
        }
    }
    out.iter_mut().for_each(|v| {
        if *v == u16::MAX {
            *v = 0
        }
    });
    out
}

/// What a 2D box sees in 3D: the world points of the readings inside it that belong to the thing in front.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Placed {
    pub center: [f64; 3],
    pub size: [f64; 3],
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// readings that went into it
    pub support: usize,
    /// the median distance from the camera
    pub range: f64,
}

/// The thing in `bbox` (pixels of `camera`): the readings in the box's central half pick the object's distance (their
/// median), every reading in the box within a band around it is lifted to the world by `world_from_camera`, and the
/// 2nd-98th percentiles bound it. None when the box has too few readings.
pub fn place_box(depth_mm: &[u16], camera: &Pinhole, bbox: [f32; 4], world_from_camera: &Iso) -> Option<Placed> {
    let (w, h) = (camera.width as i64, camera.height as i64);
    let clamp = |v: f32, hi: i64| (v.floor() as i64).clamp(0, hi);
    let (x0, y0, x1, y1) = (clamp(bbox[0], w), clamp(bbox[1], h), clamp(bbox[2], w), clamp(bbox[3], h));
    if x1 - x0 < 2 || y1 - y0 < 2 {
        return None;
    }
    let (qx, qy) = ((x1 - x0) / 4, (y1 - y0) / 4);
    let mut central: Vec<u16> = Vec::new();
    for y in (y0 + qy)..(y1 - qy) {
        for x in (x0 + qx)..(x1 - qx) {
            let mm = depth_mm[(y * w + x) as usize];
            if mm > 0 {
                central.push(mm);
            }
        }
    }
    if central.len() < 5 {
        return None;
    }
    central.sort_unstable();
    let range = central[central.len() / 2] as f64 / 1000.0;
    let band = (0.15 * range).clamp(0.25, 0.8);
    let mut points: Vec<Point3<f64>> = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let z = depth_mm[(y * w + x) as usize] as f64 / 1000.0;
            if z > 0.0 && (z - range).abs() <= band {
                points.push(world_from_camera * camera.unproject(x as f64 + 0.5, y as f64 + 0.5, z));
            }
        }
    }
    if points.len() < 10 {
        return None;
    }
    let mut placed = Placed { support: points.len(), range, ..Default::default() };
    for axis in 0..3 {
        let mut values: Vec<f64> = points.iter().map(|p| p[axis]).collect();
        values.sort_by(f64::total_cmp);
        let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
        let (low, high) = (at(0.02), at(0.98));
        placed.min[axis] = low;
        placed.max[axis] = high;
        placed.center[axis] = (low + high) / 2.0;
        placed.size[axis] = high - low;
    }
    Some(placed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tf::iso;

    fn camera() -> Pinhole {
        Pinhole { fx: 100.0, fy: 100.0, cx: 50.0, cy: 50.0, width: 100, height: 100 }
    }

    #[test]
    fn project_and_unproject_invert() {
        let cam = camera();
        let p = cam.unproject(70.0, 20.0, 2.5);
        let (u, v) = cam.project(&p).unwrap();
        assert!((u - 70.0).abs() < 1e-9 && (v - 20.0).abs() < 1e-9);
        assert!(cam.project(&Point3::new(0.0, 0.0, -1.0)).is_none());
    }

    #[test]
    fn the_map_renders_nearest_first_into_the_camera() {
        // camera at the origin looking along +z (world = optical): a wall at 4 m behind a post at 2 m
        let cam = camera();
        let mut points = Vec::new();
        for y in -20..20 {
            for x in -20..20 {
                points.push([x as f32 * 0.05, y as f32 * 0.05, 4.0]);
            }
        }
        for y in -5..5 {
            for x in -1..1 {
                points.push([x as f32 * 0.05, y as f32 * 0.05, 2.0]);
            }
        }
        let depth = render_depth(&points, 0.05, &cam, &Iso::identity(), 0.3, 10.0);
        // the image center sees the post, not the wall behind it; off to the side, the wall
        assert_eq!(depth[50 * 100 + 50], 2000);
        assert_eq!(depth[50 * 100 + 60], 4000);
        // nothing beyond the wall's edge (0.95 m at 4 m = 24 px from center)
        assert_eq!(depth[50 * 100 + 90], 0);
        // too far is dropped
        assert!(render_depth(&points, 0.05, &cam, &Iso::identity(), 0.3, 1.0).iter().all(|&v| v == 0));
    }

    #[test]
    fn a_box_on_a_post_measures_its_own_height_not_the_wall_behind() {
        // a 0.5 m wide post 2 m away in front of a wall 5 m away; the camera looks along world +x (optical z),
        // world z is up (optical -y)
        let cam = camera();
        let mut depth = vec![5000u16; 100 * 100];
        for y in 25..75 {
            for x in 40..60 {
                depth[y * 100 + x] = 2000;
            }
        }
        // world <- optical: x_w = z_c, y_w = -x_c, z_w = -y_c, camera 1 m above the floor
        let world_from_camera = Iso::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::from_matrix(&nalgebra::Matrix3::new(0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0)),
        );
        let placed = place_box(&depth, &cam, [35.0, 20.0, 65.0, 80.0], &world_from_camera).unwrap();
        assert!((placed.range - 2.0).abs() < 1e-9);
        assert!((placed.center[0] - 2.0).abs() < 0.01, "{:?}", placed);
        // 50 px tall at 2 m = 1.0 m (the 2-98th percentiles trim a pixel row)
        assert!((placed.size[2] - 1.0).abs() < 0.06, "{:?}", placed);
        assert!((placed.size[1] - 0.4).abs() < 0.06, "{:?}", placed);
        assert!(place_box(&vec![0u16; 100 * 100], &cam, [35.0, 20.0, 65.0, 80.0], &world_from_camera).is_none());
    }
}
