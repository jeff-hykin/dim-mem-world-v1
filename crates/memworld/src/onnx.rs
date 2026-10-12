//! ONNX Runtime, loaded at run time from the library the nix build points `ORT_DYLIB_PATH` at (no Python, no model
//! code compiled in). One environment per process; sessions are built per model file.
use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;
use std::sync::OnceLock;

static READY: OnceLock<std::result::Result<(), String>> = OnceLock::new();

pub fn init() -> Result<()> {
    READY
        .get_or_init(|| {
            let builder = match std::env::var("ORT_DYLIB_PATH") {
                Ok(path) if !path.is_empty() => ort::init_from(path),
                _ => ort::init(),
            };
            builder.with_name("memworld").commit().map(|_| ()).map_err(|error| format!("ONNX Runtime didn't load (set ORT_DYLIB_PATH): {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

pub fn session(path: &Path) -> Result<Session> {
    init()?;
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads())?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}
