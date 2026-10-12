//! A text question becomes places in 3D:
//!
//! 1. per member, every patch's heat = its score for the question minus its best score among background prompts
//!    ("floor", "a wall", ...): positive where the question wins;
//! 2. per keyframe, each cell of the finest grid is AGREED when it is hot in all three members (each member's grid
//!    covers the frame its own way; a fine cell reads the member cell under its center). Requiring all three is what
//!    keeps the floor beside a cone out: each model's false heat lands in different cells;
//! 3. the frames with the most agreed heat go to OWLv2 with the same question; a box counts when agreed cells fall
//!    inside it (with no OWLv2, the agreed regions themselves are the boxes);
//! 4. the lidar map is drawn into the keyframe's camera (nearest surface wins), and what it shows inside each box is
//!    lifted to the world with the frame's pose: a center, a size, a height;
//! 5. sightings closer than a place radius merge into one place. The count of places answers "how many".
use crate::camera::{place_box, render_depth, Placed};
use crate::memory::{Memory, MemberIndex};
use crate::owl::{self, Owl};
use crate::siglip::Member;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// the lidar map is drawn into a keyframe between these distances (nearer is the robot itself)
const MIN_RANGE_M: f64 = 0.3;
const MAX_RANGE_M: f64 = 20.0;

pub const BACKGROUND: [&str; 7] = ["floor", "a wall", "the ceiling", "a room", "an object", "a photo", "a door"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Options {
    /// at most this many keyframes go to OWLv2
    pub frames: usize,
    pub owl_threshold: f32,
    /// sightings whose centers are closer than this (horizontally) are one place
    pub place_radius_m: f64,
    /// skip OWLv2: agreed regions are the boxes
    pub agreement_only: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { frames: 200, owl_threshold: 0.2, place_radius_m: 0.75, agreement_only: false }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub frame: usize,
    pub ts: f64,
    /// agreed heat summed over the frame
    pub heat: f32,
    /// the finest grid (rows, cols) and its agreed cells as [row, col]
    pub grid: [u16; 2],
    pub cells: Vec<[u16; 2]>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Observation {
    pub frame: usize,
    pub ts: f64,
    /// [x0, y0, x1, y1] in the stored frame's pixels
    pub bbox: [f32; 4],
    /// "owl" or "agreement"
    pub source: &'static str,
    pub score: f32,
    /// agreed heat inside the box
    pub agreement: f32,
    pub placed: Option<Placed>,
    /// which place it joined
    pub place: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Place {
    pub id: usize,
    pub center: [f64; 3],
    pub size: [f64; 3],
    /// top minus bottom, meters (median over sightings)
    pub height_m: f64,
    pub bottom_z: f64,
    pub top_z: f64,
    pub observations: Vec<usize>,
    /// the sighting with the best score: the evidence photo
    pub best: usize,
    pub score: f32,
    pub first_seen: f64,
    pub last_seen: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Timings {
    pub text_ms: f64,
    pub search_ms: f64,
    pub owl_ms: f64,
    pub place_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Answer {
    pub query: String,
    pub count: usize,
    pub places: Vec<Place>,
    pub observations: Vec<Observation>,
    pub candidates: Vec<Candidate>,
    /// boxes found but with no lidar map behind them (not placed, not counted)
    pub unplaced: usize,
    pub world_frame: String,
    pub timings: Timings,
    pub summary: String,
}

/// Per member, per patch: the best score among the background prompts. Question-independent: worked out once per
/// memory and kept.
pub fn background(memory: &Memory, members: &[Member]) -> Result<Vec<Vec<f32>>> {
    if members.len() != memory.members.len() {
        bail!("the memory has {} members, {} models are loaded", memory.members.len(), members.len());
    }
    let mut background = Vec::new();
    for (member, index) in members.iter().zip(&memory.members) {
        let prompts = member.embed_text(&BACKGROUND)?;
        let mut best = vec![f32::MIN; index.len()];
        for prompt in &prompts {
            for (b, s) in best.iter_mut().zip(index.scores(prompt)) {
                *b = b.max(s);
            }
        }
        background.push(best);
    }
    Ok(background)
}

/// What a question needs: the memory, the models, and the memory's background scores.
pub struct Engine<'a> {
    pub memory: &'a Memory,
    pub members: &'a [Member],
    pub owl: Option<&'a Owl>,
    pub background: &'a [Vec<f32>],
}

impl<'a> Engine<'a> {

    /// Each member's heat for `text`, per patch.
    pub fn heat(&self, text: &str) -> Result<Vec<Vec<f32>>> {
        let mut heats = Vec::new();
        for ((member, index), background) in self.members.iter().zip(&self.memory.members).zip(self.background.iter()) {
            let query = member.embed_text(&[text])?.remove(0);
            heats.push(index.scores(&query).into_iter().zip(background).map(|(s, b)| s - b).collect());
        }
        Ok(heats)
    }

    pub fn ask(&self, text: &str, options: &Options) -> Result<Answer> {
        let started = Instant::now();
        let mut timings = Timings::default();
        let heats = self.heat(text)?;
        timings.text_ms = ms(started);
        let search_started = Instant::now();
        let mut candidates = agreed_frames(&self.memory.members, &heats, self.memory.frames.len());
        for candidate in &mut candidates {
            candidate.ts = self.memory.frames[candidate.frame].ts;
        }
        candidates.truncate(options.frames);
        timings.search_ms = ms(search_started);

        let owl_started = Instant::now();
        let owl_query = match (self.owl, options.agreement_only) {
            (Some(owl), false) => Some(owl.embed_text(text)?),
            _ => None,
        };
        let mut observations = Vec::new();
        for candidate in &candidates {
            let frame = &self.memory.frames[candidate.frame];
            let (width, height) = (frame.camera.width as f32, frame.camera.height as f32);
            let (rows, cols) = (candidate.grid[0] as f32, candidate.grid[1] as f32);
            let cell_heat = cell_heats(&self.memory.members, &heats, candidate.frame, candidate.grid);
            let agreement_in = |bbox: &[f32; 4]| -> f32 {
                candidate
                    .cells
                    .iter()
                    .filter(|[r, c]| {
                        let (x, y) = ((*c as f32 + 0.5) / cols * width, (*r as f32 + 0.5) / rows * height);
                        x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3]
                    })
                    .map(|[r, c]| cell_heat[*r as usize * candidate.grid[1] as usize + *c as usize])
                    .sum()
            };
            match (self.owl, &owl_query) {
                (Some(owl), Some(query)) => {
                    // the stored image half when the memory has it, else run it now on the stored JPEG
                    let features = if frame.owl.is_empty() { owl.features(&owl::decode_jpeg(&frame.jpeg)?)? } else { owl::Features::decode(&frame.owl)? };
                    for detection in features.detect(query, frame.camera.width, frame.camera.height, options.owl_threshold) {
                        let agreement = agreement_in(&detection.bbox);
                        if agreement > 0.0 {
                            observations.push(Observation { frame: candidate.frame, ts: frame.ts, bbox: detection.bbox, source: "owl", score: detection.score, agreement, placed: None, place: None });
                        }
                    }
                }
                _ => {
                    for (bbox, agreement) in regions(candidate, width, height, &cell_heat) {
                        observations.push(Observation { frame: candidate.frame, ts: frame.ts, bbox, source: "agreement", score: agreement, agreement, placed: None, place: None });
                    }
                }
            }
        }
        timings.owl_ms = ms(owl_started);

        let place_started = Instant::now();
        let mut unplaced = 0;
        let mut rendered: std::collections::HashMap<usize, Vec<u16>> = std::collections::HashMap::new();
        for observation in &mut observations {
            let frame = &self.memory.frames[observation.frame];
            observation.placed = self.memory.map.as_ref().and_then(|map| {
                let depth = rendered.entry(observation.frame).or_insert_with(|| render_depth(map, crate::lidar::VOXEL_M, &frame.camera, &frame.pose, MIN_RANGE_M, MAX_RANGE_M));
                place_box(depth, &frame.camera, observation.bbox, &frame.pose)
            });
            if observation.placed.is_none() {
                unplaced += 1;
            }
        }
        let places = merge(&mut observations, options.place_radius_m);
        timings.place_ms = ms(place_started);
        timings.total_ms = ms(started);
        let world_frame = self.memory.meta["worldFrame"].as_str().unwrap_or("world").to_string();
        let summary = summarize(text, &places, &observations, &world_frame);
        Ok(Answer { query: text.into(), count: places.len(), places, observations, candidates, unplaced, world_frame, timings, summary })
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// The finest grid among the members for one frame (most cells).
fn finest(members: &[MemberIndex], frame: usize) -> [u16; 2] {
    members.iter().map(|m| (m.grids[frame].1, m.grids[frame].2)).max_by_key(|(r, c)| *r as u32 * *c as u32).map(|(r, c)| [r, c]).unwrap_or([0, 0])
}

/// The member cell under a fine cell's center, as a patch index.
fn member_patch(member: &MemberIndex, frame: usize, grid: [u16; 2], r: u16, c: u16) -> usize {
    let (start, rows, cols) = member.grids[frame];
    let v = (r as f32 + 0.5) / grid[0] as f32;
    let u = (c as f32 + 0.5) / grid[1] as f32;
    let mr = ((v * rows as f32) as usize).min(rows as usize - 1);
    let mc = ((u * cols as f32) as usize).min(cols as usize - 1);
    start + mr * cols as usize + mc
}

/// Per fine cell of one frame: the summed heat when all members are hot there, else 0.
pub fn cell_heats(members: &[MemberIndex], heats: &[Vec<f32>], frame: usize, grid: [u16; 2]) -> Vec<f32> {
    let mut out = vec![0.0f32; grid[0] as usize * grid[1] as usize];
    for r in 0..grid[0] {
        for c in 0..grid[1] {
            let mut sum = 0.0;
            let mut agreed = true;
            for (member, heat) in members.iter().zip(heats) {
                let h = heat[member_patch(member, frame, grid, r, c)];
                if h <= 0.0 {
                    agreed = false;
                    break;
                }
                sum += h;
            }
            if agreed {
                out[r as usize * grid[1] as usize + c as usize] = sum;
            }
        }
    }
    out
}

/// Frames with agreed cells, most agreed heat first.
pub fn agreed_frames(members: &[MemberIndex], heats: &[Vec<f32>], frames: usize) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = (0..frames)
        .filter_map(|frame| {
            let grid = finest(members, frame);
            let heat = cell_heats(members, heats, frame, grid);
            let cells: Vec<[u16; 2]> = heat.iter().enumerate().filter(|(_, h)| **h > 0.0).map(|(i, _)| [(i / grid[1] as usize) as u16, (i % grid[1] as usize) as u16]).collect();
            (!cells.is_empty()).then(|| Candidate { frame, ts: 0.0, heat: heat.iter().sum(), grid, cells })
        })
        .collect();
    out.sort_by(|a, b| b.heat.total_cmp(&a.heat));
    out
}

/// The agreed cells' connected regions (4-neighbours) as pixel boxes with their heat: the boxes when OWLv2 isn't used.
pub fn regions(candidate: &Candidate, width: f32, height: f32, cell_heat: &[f32]) -> Vec<([f32; 4], f32)> {
    let (rows, cols) = (candidate.grid[0] as i32, candidate.grid[1] as i32);
    let mut seen = vec![false; (rows * cols) as usize];
    let mut out = Vec::new();
    for [r, c] in &candidate.cells {
        let start = (*r as i32 * cols + *c as i32) as usize;
        if seen[start] {
            continue;
        }
        let (mut r0, mut c0, mut r1, mut c1, mut heat) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN, 0.0f32);
        let mut stack = vec![(*r as i32, *c as i32)];
        seen[start] = true;
        while let Some((y, x)) = stack.pop() {
            heat += cell_heat[(y * cols + x) as usize];
            (r0, c0, r1, c1) = (r0.min(y), c0.min(x), r1.max(y), c1.max(x));
            for (dy, dx) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                let (ny, nx) = (y + dy, x + dx);
                if ny >= 0 && nx >= 0 && ny < rows && nx < cols {
                    let at = (ny * cols + nx) as usize;
                    if !seen[at] && cell_heat[at] > 0.0 {
                        seen[at] = true;
                        stack.push((ny, nx));
                    }
                }
            }
        }
        let bbox = [c0 as f32 / cols as f32 * width, r0 as f32 / rows as f32 * height, (c1 + 1) as f32 / cols as f32 * width, (r1 + 1) as f32 / rows as f32 * height];
        out.push((bbox, heat));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Greedy, best sighting first: a placed sighting joins the place whose center is within `radius` horizontally (and
/// 1 m vertically), else starts one. A place's center, size and height are medians over its sightings.
pub fn merge(observations: &mut [Observation], radius: f64) -> Vec<Place> {
    let mut order: Vec<usize> = (0..observations.len()).filter(|&i| observations[i].placed.is_some()).collect();
    order.sort_by(|&a, &b| observations[b].score.total_cmp(&observations[a].score));
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut centers: Vec<[f64; 3]> = Vec::new();
    for i in order {
        let c = observations[i].placed.as_ref().unwrap().center;
        let found = centers.iter().position(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt() <= radius && (p[2] - c[2]).abs() <= 1.0);
        match found {
            Some(place) => members[place].push(i),
            None => {
                centers.push(c);
                members.push(vec![i]);
            }
        }
    }
    let mut places: Vec<Place> = members
        .into_iter()
        .enumerate()
        .map(|(id, list)| {
            let placed: Vec<&Placed> = list.iter().map(|&i| observations[i].placed.as_ref().unwrap()).collect();
            let axis = |f: &dyn Fn(&Placed) -> f64| median(placed.iter().map(|p| f(p)).collect());
            // the evidence photo: a confident sighting that shows the thing large (score x the box's diagonal)
            let best = *list.iter().max_by(|&&a, &&b| evidence_weight(&observations[a]).total_cmp(&evidence_weight(&observations[b]))).unwrap();
            Place {
                id,
                center: [axis(&|p| p.center[0]), axis(&|p| p.center[1]), axis(&|p| p.center[2])],
                size: [axis(&|p| p.size[0]), axis(&|p| p.size[1]), axis(&|p| p.size[2])],
                height_m: axis(&|p| p.size[2]),
                bottom_z: axis(&|p| p.min[2]),
                top_z: axis(&|p| p.max[2]),
                score: list.iter().map(|&i| observations[i].score).fold(0.0, f32::max),
                first_seen: list.iter().map(|&i| observations[i].ts).fold(f64::MAX, f64::min),
                last_seen: list.iter().map(|&i| observations[i].ts).fold(f64::MIN, f64::max),
                best,
                observations: list,
            }
        })
        .collect();
    places.sort_by(|a, b| b.score.total_cmp(&a.score));
    for (id, place) in places.iter_mut().enumerate() {
        place.id = id;
        for &i in &place.observations {
            observations[i].place = Some(id);
        }
    }
    places
}

fn evidence_weight(observation: &Observation) -> f32 {
    let [x0, y0, x1, y1] = observation.bbox;
    observation.score * (1.0 + ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt())
}

fn summarize(text: &str, places: &[Place], observations: &[Observation], world: &str) -> String {
    if places.is_empty() {
        return format!("No \"{text}\" was found in 3D ({} sightings, none with the lidar map behind them).", observations.len());
    }
    let list: Vec<String> = places
        .iter()
        .map(|p| format!("#{} at ({:.2}, {:.2}, {:.2}) m in {world}, about {:.2} m tall, seen in {} frame(s)", p.id + 1, p.center[0], p.center[1], p.center[2], p.height_m, p.observations.len()))
        .collect();
    format!("{} place(s) with \"{text}\": {}", places.len(), list.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(rows: u16, cols: u16, frames: usize) -> MemberIndex {
        let mut index = MemberIndex { key: "m".into(), dim: 1, data: Vec::new(), cells: Vec::new(), grids: Vec::new() };
        for f in 0..frames {
            index.grids.push((index.cells.len(), rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    index.cells.push((f as u32, r, c));
                    index.data.push(0.0);
                }
            }
        }
        index
    }

    /// heat of +1 in the given cells (row-major per frame), -1 elsewhere
    fn heat(index: &MemberIndex, hot: &[(usize, u16, u16)]) -> Vec<f32> {
        index.cells.iter().map(|(f, r, c)| if hot.contains(&(*f as usize, *r, *c)) { 1.0 } else { -1.0 }).collect()
    }

    #[test]
    fn only_cells_hot_in_every_member_agree() {
        // a coarse 2x2 member and a fine 4x4 member, one frame: the fine cell (0,0) sits in coarse (0,0)
        let coarse = index(2, 2, 1);
        let fine = index(4, 4, 1);
        let heats = vec![heat(&coarse, &[(0, 0, 0)]), heat(&fine, &[(0, 0, 0), (0, 0, 3), (0, 1, 1)])];
        let members = [coarse, fine];
        let found = agreed_frames(&members, &heats, 1);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].grid, [4, 4]);
        // (0,3) is hot in the fine member only (coarse (0,1) is cold): not agreed
        assert_eq!(found[0].cells, vec![[0, 0], [1, 1]]);
        assert_eq!(found[0].heat, 4.0);
    }

    #[test]
    fn frames_rank_by_agreed_heat_and_cold_frames_drop() {
        let m = index(2, 2, 3);
        let heats = vec![heat(&m, &[(1, 0, 0), (2, 0, 0), (2, 1, 1)]); 3];
        let members = [index(2, 2, 3), index(2, 2, 3), index(2, 2, 3)];
        let found = agreed_frames(&members, &heats, 3);
        assert_eq!(found.iter().map(|c| c.frame).collect::<Vec<_>>(), vec![2, 1]);
    }

    #[test]
    fn regions_split_on_gaps() {
        let candidate = Candidate { frame: 0, ts: 0.0, heat: 3.0, grid: [2, 4], cells: vec![[0, 0], [1, 0], [0, 3]] };
        let mut cell_heat = vec![0.0; 8];
        cell_heat[0] = 1.0;
        cell_heat[4] = 1.0;
        cell_heat[3] = 1.0;
        let found = regions(&candidate, 400.0, 200.0, &cell_heat);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], ([0.0, 0.0, 100.0, 200.0], 2.0));
        assert_eq!(found[1].0, [300.0, 0.0, 400.0, 100.0]);
    }

    fn sighting(score: f32, center: [f64; 3], height: f64) -> Observation {
        let placed = Placed { center, size: [0.3, 0.3, height], min: [0.0, 0.0, center[2] - height / 2.0], max: [0.0, 0.0, center[2] + height / 2.0], support: 100, range: 2.0 };
        Observation { frame: 0, ts: score as f64, bbox: [0.0; 4], source: "owl", score, agreement: 1.0, placed: Some(placed), place: None }
    }

    #[test]
    fn nearby_sightings_are_one_place_far_ones_two() {
        let mut observations = vec![sighting(0.9, [0.0, 0.0, 0.35], 0.7), sighting(0.8, [0.3, 0.1, 0.36], 0.72), sighting(0.7, [5.0, 0.0, 0.2], 0.4), sighting(0.6, [0.1, 0.0, 0.35], 0.68)];
        observations.push(Observation { placed: None, ..sighting(0.95, [0.0; 3], 0.0) });
        let places = merge(&mut observations, 0.75);
        assert_eq!(places.len(), 2);
        assert_eq!(places[0].observations, vec![0, 1, 3]);
        assert!((places[0].height_m - 0.7).abs() < 1e-9);
        assert_eq!(observations[2].place, Some(1));
        // equal boxes: the best-scoring sighting is the evidence
        assert_eq!(places[0].best, 0);
        assert_eq!(observations[4].place, None);
    }
}
