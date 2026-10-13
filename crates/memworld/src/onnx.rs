//! ONNX Runtime, loaded at run time from the library the nix build points `ORT_DYLIB_PATH` at (no Python, no model
//! code compiled in). One environment per process; sessions are built per model file, on the GPU when there is one
//! (CoreML on a Mac, CUDA on Nvidia), and rebuilt on the CPU when the GPU can't load or run a model.
use anyhow::{Context, Result};
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch};
use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLModelFormat};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

static READY: OnceLock<std::result::Result<(), String>> = OnceLock::new();

pub fn init() -> Result<()> {
    READY
        .get_or_init(|| {
            preload_cuda_driver();
            let builder = match std::env::var("ORT_DYLIB_PATH") {
                Ok(path) if !path.is_empty() => ort::init_from(path),
                _ => ort::init(),
            };
            builder.with_name("memworld").commit().map(|_| ()).map_err(|error| format!("ONNX Runtime didn't load (set ORT_DYLIB_PATH): {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// A nix-built process doesn't search the system's library folders, so the Nvidia driver's libcuda is loaded here by
/// its full path; CUDA's own lookup by name then finds it already loaded.
fn preload_cuda_driver() {
    if !cfg!(target_os = "linux") {
        return;
    }
    let folders = ["/run/opengl-driver/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/lib/wsl/lib"];
    for folder in folders {
        let path = Path::new(folder).join("libcuda.so.1");
        // SAFETY: the driver library has no load-time side effects beyond its own initializers
        if path.exists() && unsafe { libloading::os::unix::Library::open(Some(&path), libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL) }.map(std::mem::forget).is_ok() {
            return;
        }
    }
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Where models run: `MEMWORLD_DEVICE` = auto (default: the GPU if this machine has one), cpu, coreml or cuda.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Coreml,
    Cuda,
}

pub fn wanted_device() -> Device {
    match std::env::var("MEMWORLD_DEVICE").unwrap_or_default().to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "coreml" => Device::Coreml,
        "cuda" => Device::Cuda,
        _ if cfg!(target_os = "macos") => Device::Coreml,
        _ if cfg!(target_os = "linux") && Path::new("/proc/driver/nvidia/version").exists() => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn provider(device: Device, path: &Path, fixed: &[(&str, i64)]) -> Option<ExecutionProviderDispatch> {
    match device {
        Device::Cpu => None,
        // compiled CoreML models are cached beside the onnx file, one per shape (a compile takes minutes; a cached load, seconds)
        Device::Coreml => Some(
            CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(CoreMLComputeUnits::CPUAndGPU)
                .with_model_cache_dir(path.with_extension(fixed.iter().fold("coreml".to_string(), |name, (dim, size)| format!("{name}.{dim}{size}"))).display())
                .build()
                .error_on_failure(),
        ),
        Device::Cuda => Some(CUDAExecutionProvider::default().build().error_on_failure()),
    }
}

/// `fixed` pins a model's named dynamic dimensions (CoreML compiles only fixed shapes).
fn build(path: &Path, device: Device, fixed: &[(&str, i64)]) -> Result<Session> {
    init()?;
    let mut builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_intra_threads(threads())?;
    for (name, size) in fixed {
        builder = builder.with_dimension_override(*name, *size)?;
    }
    if let Some(provider) = provider(device, path, fixed) {
        builder = builder.with_execution_providers([provider])?;
    }
    builder.commit_from_file(path).with_context(|| format!("loading {} on {device:?}", path.display()))
}

/// A model's session. Built on the wanted device; if that fails to load, or a run fails on the GPU (out of memory, an
/// op the GPU can't do), it is rebuilt on the CPU and the run is retried there, once, for good.
pub struct Model {
    path: PathBuf,
    fixed: Vec<(String, i64)>,
    state: Mutex<(Session, Device)>,
}

impl Model {
    pub fn load(path: &Path, fixed: &[(&str, i64)]) -> Result<Model> {
        Model::load_on(path, fixed, wanted_device())
    }

    pub fn load_on(path: &Path, fixed: &[(&str, i64)], wanted: Device) -> Result<Model> {
        let (session, device) = match build(path, wanted, fixed) {
            Ok(session) => (session, wanted),
            Err(error) if wanted != Device::Cpu => {
                eprintln!("[memworld] {} won't load on {wanted:?}, using the CPU: {error:#}", path.display());
                (build(path, Device::Cpu, fixed)?, Device::Cpu)
            }
            Err(error) => return Err(error),
        };
        Ok(Model { path: path.into(), fixed: fixed.iter().map(|(name, size)| (name.to_string(), *size)).collect(), state: Mutex::new((session, device)) })
    }

    pub fn device(&self) -> Device {
        self.state.lock().unwrap().1
    }

    /// Runs `f` on the session (`f` builds its inputs and copies out what it needs, so it can be run again).
    pub fn run<T>(&self, f: impl Fn(&mut Session) -> Result<T>) -> Result<T> {
        let mut state = self.state.lock().unwrap();
        match f(&mut state.0) {
            Ok(value) => Ok(value),
            Err(error) if state.1 != Device::Cpu => {
                eprintln!("[memworld] {} failed on {:?}, moving it to the CPU: {error:#}", self.path.display(), state.1);
                let fixed: Vec<(&str, i64)> = self.fixed.iter().map(|(name, size)| (name.as_str(), *size)).collect();
                *state = (build(&self.path, Device::Cpu, &fixed)?, Device::Cpu);
                f(&mut state.0)
            }
            Err(error) => Err(error),
        }
    }
}
