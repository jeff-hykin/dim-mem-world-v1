//! Memory World: a robot recording's visual memory, stored inside the recording, and questions about it answered as
//! places in 3D. `compat` checks a recording, `generate` embeds its frames (three SigLIP2 members, per patch), `memory`
//! loads that back, `query` answers a text question (agreement of all three, OWLv2 boxes, placed in 3D with the lidar map).
pub mod blob;
pub mod camera;
pub mod compat;
#[cfg(target_os = "macos")]
pub mod coreml;
pub mod generate;
pub mod lidar;
pub mod lite_record;
pub mod memory;
pub mod models;
pub mod onnx;
pub mod owl;
pub mod query;
pub mod siglip;
pub mod store;
pub mod tf;

pub use dimos_recording as recording;
