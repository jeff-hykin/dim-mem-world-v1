//! Native CoreML on a Mac: the image models as fp16 ML Programs (tools/export_coreml.py, release `models-v3`). A
//! package is compiled for this machine once (`<name>.mlmodelc` beside it) and then loads in about a second; ONNX
//! Runtime's CoreML provider recompiled the ONNX graphs on every start instead (minutes).
use anyhow::{bail, Context, Result};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration, MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

pub struct Native {
    name: String,
    model: Mutex<Retained<MLModel>>,
}

// SAFETY: an MLModel can predict from any thread; the Mutex keeps it to one prediction at a time anyway
unsafe impl Send for Native {}
unsafe impl Sync for Native {}

/// `<dir>/<name>.mlpackage.tar` (downloaded) -> `<name>.mlpackage` (unpacked) -> `<name>.mlmodelc` (compiled here).
fn package_tar(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.mlpackage.tar"))
}

fn compiled(dir: &Path, name: &str) -> Result<PathBuf> {
    let target = dir.join(format!("{name}.mlmodelc"));
    if target.exists() {
        return Ok(target);
    }
    let package = dir.join(format!("{name}.mlpackage"));
    if !package.exists() {
        let tar = package_tar(dir, name);
        let status = std::process::Command::new("/usr/bin/tar").arg("-xf").arg(&tar).arg("-C").arg(dir).status().context("running tar")?;
        if !status.success() {
            bail!("unpacking {}", tar.display());
        }
    }
    let url = NSURL::from_file_path(&package).context("package path")?;
    #[allow(deprecated)]
    let made = unsafe { MLModel::compileModelAtURL_error(&url) }.map_err(|error| anyhow::anyhow!("compiling {name}: {error}"))?;
    let made = made.to_file_path().context("compiled model path")?;
    // the compiled model lands in a temporary folder, maybe on another volume
    let staging = dir.join(format!("{name}.mlmodelc.part"));
    let _ = std::fs::remove_dir_all(&staging);
    let status = std::process::Command::new("/bin/cp").arg("-R").arg(&made).arg(&staging).status().context("copying the compiled model")?;
    if !status.success() {
        bail!("copying {}", made.display());
    }
    let _ = std::fs::remove_dir_all(&made);
    std::fs::rename(&staging, &target)?;
    let _ = std::fs::remove_dir_all(&package);
    Ok(target)
}

/// The native model `name` when models run on CoreML here and its package was downloaded; None (ONNX Runtime on the
/// CPU) otherwise, or if it won't load.
pub fn load_if_wanted(dir: &Path, name: &str) -> Option<Arc<Native>> {
    // ONNX Runtime's CoreML caches from v0.2.0 (`<model>.vision.coreml*`, gigabytes each) are never read again
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().contains(".vision.coreml") {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    if crate::onnx::wanted_device() != crate::onnx::Device::Coreml || !Native::available(dir, name) {
        return None;
    }
    match Native::load(dir, name) {
        Ok(native) => Some(Arc::new(native)),
        Err(error) => {
            eprintln!("[memworld] {name} won't load on CoreML, using the CPU: {error:#}");
            None
        }
    }
}

impl Native {
    pub fn available(dir: &Path, name: &str) -> bool {
        dir.join(format!("{name}.mlmodelc")).exists() || dir.join(format!("{name}.mlpackage")).exists() || package_tar(dir, name).exists()
    }

    pub fn load(dir: &Path, name: &str) -> Result<Native> {
        let path = compiled(dir, name)?;
        autoreleasepool(|_| {
            let url = NSURL::from_file_path(&path).context("model path")?;
            let configuration = unsafe { MLModelConfiguration::new() };
            unsafe { configuration.setComputeUnits(MLComputeUnits::CPUAndGPU) };
            let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration) }.map_err(|error| anyhow::anyhow!("loading {name}: {error}"))?;
            Ok(Native { name: name.into(), model: Mutex::new(model) })
        })
    }

    /// Runs the model on f32 inputs (name, shape, values); returns each named output as f32, row-major.
    pub fn run(&self, inputs: &[(&str, &[usize], &[f32])], outputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.lock().unwrap();
        autoreleasepool(|_| {
            let mut keys: Vec<Retained<NSString>> = Vec::new();
            let mut values: Vec<Retained<MLFeatureValue>> = Vec::new();
            for (name, shape, data) in inputs {
                if data.len() != shape.iter().product::<usize>() {
                    bail!("{}: {name} has {} values for shape {shape:?}", self.name, data.len());
                }
                let mut strides = vec![1usize; shape.len()];
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                let numbers = |list: &[usize]| NSArray::from_retained_slice(&list.iter().map(|&v| NSNumber::new_usize(v)).collect::<Vec<_>>());
                // CoreML only reads the inputs, and `data` outlives the prediction below
                let pointer = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).context("empty input")?;
                let array = unsafe { MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(MLMultiArray::alloc(), pointer, &numbers(shape), MLMultiArrayDataType::Float32, &numbers(&strides), None) }
                    .map_err(|error| anyhow::anyhow!("{}: input {name}: {error}", self.name))?;
                keys.push(NSString::from_str(name));
                values.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&array) });
            }
            let key_refs: Vec<&NSString> = keys.iter().map(|key| &**key).collect();
            let value_refs: Vec<&AnyObject> = values.iter().map(|value| -> &AnyObject { value }).collect();
            let dictionary = NSDictionary::from_slices(&key_refs, &value_refs);
            let provider = unsafe { MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dictionary) }.map_err(|error| anyhow::anyhow!("{}: inputs: {error}", self.name))?;
            let result = unsafe { model.predictionFromFeatures_error(ProtocolObject::from_ref(&*provider)) }.map_err(|error| anyhow::anyhow!("{}: {error}", self.name))?;
            outputs
                .iter()
                .map(|name| {
                    let value = unsafe { result.featureValueForName(&NSString::from_str(name)) }.with_context(|| format!("{}: no output {name}", self.name))?;
                    let array = unsafe { value.multiArrayValue() }.with_context(|| format!("{}: {name} isn't an array", self.name))?;
                    read(&array).with_context(|| format!("{}: reading {name}", self.name))
                })
                .collect()
        })
    }
}

/// An MLMultiArray's values in row-major order, whatever its strides and element type.
fn read(array: &MLMultiArray) -> Result<Vec<f32>> {
    let shape: Vec<usize> = unsafe { array.shape() }.iter().map(|n| n.as_usize()).collect();
    let strides: Vec<usize> = unsafe { array.strides() }.iter().map(|n| n.as_usize()).collect();
    let count: usize = shape.iter().product();
    let kind = unsafe { array.dataType() };
    #[allow(deprecated)]
    let base = unsafe { array.dataPointer() }.as_ptr();
    let contiguous = (0..shape.len()).all(|axis| strides[axis] == shape[axis + 1..].iter().product::<usize>());
    if contiguous && kind == MLMultiArrayDataType::Float32 {
        return Ok(unsafe { std::slice::from_raw_parts(base as *const f32, count) }.to_vec());
    }
    let mut out = Vec::with_capacity(count);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..count {
        let offset: usize = index.iter().zip(&strides).map(|(i, s)| i * s).sum();
        let value = unsafe {
            if kind == MLMultiArrayDataType::Float32 {
                *(base as *const f32).add(offset)
            } else if kind == MLMultiArrayDataType::Float16 {
                half::f16::from_bits(*(base as *const u16).add(offset)).to_f32()
            } else if kind == MLMultiArrayDataType::Double {
                *(base as *const f64).add(offset) as f32
            } else {
                bail!("unexpected element type {kind:?}");
            }
        };
        out.push(value);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(out)
}
