//! A recorded tf tree: every (parent, child) edge's samples over time; `lookup(target, source, t)` chains edges
//! through the tree (up and down), interpolating each edge at `t` (lerp + slerp). An edge with one sample is static
//! and holds at any time; a dynamic edge answers only within `tolerance` of its sampled range.
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};

pub type Iso = Isometry3<f64>;

pub fn iso(translation: [f64; 3], rotation: [f64; 4]) -> Iso {
    let [x, y, z, w] = rotation;
    let q = Quaternion::new(w, x, y, z);
    let q = if q.norm() < 1e-9 { UnitQuaternion::identity() } else { UnitQuaternion::from_quaternion(q) };
    Isometry3::from_parts(Translation3::new(translation[0], translation[1], translation[2]), q)
}

pub fn parts(value: &Iso) -> ([f64; 3], [f64; 4]) {
    let t = value.translation.vector;
    let q = value.rotation;
    ([t.x, t.y, t.z], [q.i, q.j, q.k, q.w])
}

pub fn interpolate(a: &Iso, b: &Iso, alpha: f64) -> Iso {
    let translation = a.translation.vector.lerp(&b.translation.vector, alpha);
    let rotation = a.rotation.try_slerp(&b.rotation, alpha, 1e-9).unwrap_or(if alpha < 0.5 { a.rotation } else { b.rotation });
    Isometry3::from_parts(Translation3::from(translation), rotation)
}

#[derive(Default, Clone)]
struct Edge {
    /// (time, parent <- child), sorted by time
    samples: Vec<(f64, Iso)>,
    /// every sample is the same transform: a static edge (tf_static, or one republished at a rate), true at any time,
    /// whatever clock its stamps are on
    fixed: bool,
}

impl Edge {
    fn at(&self, t: f64, tolerance: f64) -> Option<Iso> {
        let samples = &self.samples;
        if samples.len() == 1 || self.fixed {
            return samples.first().map(|sample| sample.1);
        }
        let first = samples.first()?.0;
        let last = samples.last()?.0;
        if t < first - tolerance || t > last + tolerance {
            return None;
        }
        let index = samples.partition_point(|(time, _)| *time < t);
        if index == 0 {
            return Some(samples[0].1);
        }
        if index >= samples.len() {
            return Some(samples[samples.len() - 1].1);
        }
        let (t0, a) = &samples[index - 1];
        let (t1, b) = &samples[index];
        let alpha = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        Some(interpolate(a, b, alpha))
    }
}

#[derive(Default, Clone)]
pub struct TfTree {
    edges: HashMap<(String, String), Edge>,
    /// frame -> neighbours (both directions)
    links: HashMap<String, HashSet<String>>,
}

impl TfTree {
    pub fn add(&mut self, t: f64, parent: &str, child: &str, value: Iso) {
        let parent = parent.trim_start_matches('/');
        let child = child.trim_start_matches('/');
        self.edges.entry((parent.to_string(), child.to_string())).or_default().samples.push((t, value));
        self.links.entry(parent.to_string()).or_default().insert(child.to_string());
        self.links.entry(child.to_string()).or_default().insert(parent.to_string());
    }

    pub fn finish(&mut self) {
        for edge in self.edges.values_mut() {
            edge.samples.sort_by(|a, b| a.0.total_cmp(&b.0));
            edge.samples.dedup_by(|a, b| a.0 == b.0);
            let first = edge.samples.first().map(|sample| sample.1);
            edge.fixed = first.is_some_and(|first| {
                edge.samples.iter().all(|(_, value)| (value.translation.vector - first.translation.vector).norm() < 1e-6 && value.rotation.angle_to(&first.rotation) < 1e-6)
            });
        }
    }

    pub fn frames(&self) -> Vec<String> {
        let mut frames: Vec<String> = self.links.keys().cloned().collect();
        frames.sort();
        frames
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    fn path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let mut previous: HashMap<String, String> = HashMap::new();
        let mut queue = VecDeque::from([from.to_string()]);
        let mut seen = HashSet::from([from.to_string()]);
        while let Some(frame) = queue.pop_front() {
            if frame == to {
                let mut path = vec![frame.clone()];
                let mut at = frame;
                while let Some(before) = previous.get(&at) {
                    path.push(before.clone());
                    at = before.clone();
                }
                path.reverse();
                return Some(path);
            }
            for next in self.links.get(&frame).into_iter().flatten() {
                if seen.insert(next.clone()) {
                    previous.insert(next.clone(), frame.clone());
                    queue.push_back(next.clone());
                }
            }
        }
        None
    }

    /// `target <- source` at `t`: maps points in `source` into `target`.
    pub fn lookup(&self, target: &str, source: &str, t: f64, tolerance: f64) -> Option<Iso> {
        let target = target.trim_start_matches('/');
        let source = source.trim_start_matches('/');
        if target == source {
            return Some(Iso::identity());
        }
        let path = self.path(target, source)?;
        let mut result = Iso::identity();
        for pair in path.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            // result is target <- a; extend to target <- b
            let step = if let Some(edge) = self.edges.get(&(a.clone(), b.clone())) {
                edge.at(t, tolerance)?
            } else {
                self.edges.get(&(b.clone(), a.clone()))?.at(t, tolerance)?.inverse()
            };
            result *= step;
        }
        Some(result)
    }
}

pub fn transform_point(value: &Iso, p: [f32; 3]) -> [f32; 3] {
    let v = value * nalgebra::Point3::new(p[0] as f64, p[1] as f64, p[2] as f64);
    [v.x as f32, v.y as f32, v.z as f32]
}

pub fn yaw_pitch_roll_iso(position: [f64; 3], yaw: f64) -> Iso {
    Isometry3::from_parts(Translation3::new(position[0], position[1], position[2]), UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_up_and_down_and_interpolates() {
        let mut tree = TfTree::default();
        tree.add(0.0, "world", "odom", iso([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]));
        tree.add(0.0, "odom", "base_link", iso([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]));
        tree.add(10.0, "odom", "base_link", iso([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]));
        tree.add(0.0, "base_link", "lidar", iso([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]));
        tree.finish();
        let at5 = tree.lookup("world", "lidar", 5.0, 0.1).unwrap();
        let p = transform_point(&at5, [0.0, 0.0, 0.0]);
        assert!((p[0] - 6.0).abs() < 1e-6 && (p[2] - 0.5).abs() < 1e-6, "{p:?}");
        // the reverse direction is the inverse
        let back = tree.lookup("lidar", "world", 5.0, 0.1).unwrap();
        assert!((back * at5).translation.vector.norm() < 1e-9);
        // outside the dynamic edge's range (beyond tolerance): no answer
        assert!(tree.lookup("world", "lidar", 20.0, 0.1).is_none());
        assert!(tree.lookup("world", "nowhere", 5.0, 0.1).is_none());
    }

    /// a sensor mount republished at 5 Hz on one clock, the odometry on another 143 s behind
    #[test]
    fn a_constant_edge_holds_on_any_clock() {
        let mut tree = TfTree::default();
        for i in 0..50 {
            tree.add(1000.0 + i as f64 * 0.2, "sensor_link", "lidar_frame", iso([0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0]));
        }
        tree.add(857.0, "odom", "sensor_link", iso([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]));
        tree.add(867.0, "odom", "sensor_link", iso([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]));
        tree.finish();
        let p = transform_point(&tree.lookup("odom", "lidar_frame", 862.0, 0.1).unwrap(), [0.0, 0.0, 0.0]);
        assert!((p[0] - 5.0).abs() < 1e-6 && (p[2] - 0.1).abs() < 1e-6, "{p:?}");
    }
}
