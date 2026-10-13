//! The server's state: the open recording, its memory held in RAM, the models (downloaded on demand, loaded once),
//! and the one background job of each kind (download, generation, loading).
use crate::downloads::Downloads;
use anyhow::{bail, Context, Result};
use memworld::compat::{self, Check};
use memworld::generate::{self, Progress};
use memworld::memory::{self, Memory, Status};
use memworld::owl::Owl;
use memworld::query::{self, Answer};
use memworld::recording::Recording;
use memworld::siglip::{Member, MEMBERS};
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// The models, loaded from the models folder (vision + text towers lazily inside each member).
pub struct Runtime {
    pub members: Vec<Member>,
    pub owl: Owl,
}

/// A memory in RAM, with what questions reuse.
pub struct Loaded {
    pub path: PathBuf,
    pub memory: Memory,
    pub load_ms: f64,
    /// the background scores (needs the text towers): worked out on the first question
    pub background: Mutex<Option<Arc<Vec<Vec<f32>>>>>,
    /// the point cloud for the 3D view, built on first request
    pub cloud: Mutex<Option<Arc<Vec<u8>>>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Open {
    pub path: String,
    pub name: String,
    /// when it was opened (ms since the epoch): a page opened before it switches to it
    pub opened_ms: u64,
}

pub struct App {
    /// the app's own folder (Desktop's data dir): models, and which generation to resume after a restart
    pub data_dir: PathBuf,
    pub recordings_dir: PathBuf,
    pub downloads: Arc<Downloads>,
    pub open: RwLock<Option<Open>>,
    pub check: RwLock<Option<Check>>,
    pub status: RwLock<Option<Status>>,
    pub runtime: tokio::sync::Mutex<Option<Arc<Runtime>>>,
    pub loaded: RwLock<Option<Arc<Loaded>>>,
    pub loading: AtomicBool,
    pub generation: Mutex<Option<Progress>>,
    pub generating: AtomicBool,
    pub cancel: Arc<AtomicBool>,
    pub last_answer: RwLock<Option<Answer>>,
    /// questions run one at a time (they use every core)
    pub asking: tokio::sync::Mutex<()>,
}

impl App {
    pub fn new(data_dir: PathBuf, recordings_dir: PathBuf) -> Arc<App> {
        let models_dir = memworld::models::dir(&data_dir);
        Arc::new(App {
            downloads: Downloads::new(models_dir),
            data_dir,
            recordings_dir,
            open: RwLock::new(None),
            check: RwLock::new(None),
            status: RwLock::new(None),
            runtime: tokio::sync::Mutex::new(None),
            loaded: RwLock::new(None),
            loading: AtomicBool::new(false),
            generation: Mutex::new(None),
            generating: AtomicBool::new(false),
            cancel: Arc::new(AtomicBool::new(false)),
            last_answer: RwLock::new(None),
            asking: tokio::sync::Mutex::new(()),
        })
    }

    /// A recording path: absolute, or a name in the recordings folder (symlinks are followed).
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        let candidate = if Path::new(path).is_absolute() { PathBuf::from(path) } else { self.recordings_dir.join(path) };
        if !candidate.exists() {
            bail!("no such recording: {}", candidate.display());
        }
        memworld::recording::format_of(&candidate)?;
        Ok(candidate)
    }

    pub fn open_path(&self) -> Result<PathBuf> {
        let open = self.open.read().unwrap();
        Ok(PathBuf::from(&open.as_ref().context("no recording is open: POST api/open first")?.path))
    }

    pub fn status_json(&self) -> Value {
        let loaded = self.loaded.read().unwrap().as_ref().map(|l| json!({ "frames": l.memory.frames.len(), "patches": l.memory.members.iter().map(|m| m.len()).collect::<Vec<_>>(), "loadMs": l.load_ms, "path": l.path }));
        json!({
            "recording": *self.open.read().unwrap(),
            "check": *self.check.read().unwrap(),
            "memory": *self.status.read().unwrap(),
            "models": self.downloads.snapshot(),
            // where the image models run (cpu, coreml, cuda); a model the GPU can't run falls back to the CPU
            "device": memworld::onnx::wanted_device(),
            "generation": *self.generation.lock().unwrap(),
            "loaded": loaded,
            "loading": self.loading.load(Ordering::Relaxed),
        })
    }

    /// Opens a recording: checks it, reads its memory status, and loads its memory in the background when it has one.
    pub async fn open(self: &Arc<Self>, path: PathBuf) -> Result<()> {
        if self.generating.load(Ordering::Relaxed) {
            bail!("a generation is running: cancel it (POST api/generate/cancel) before opening another recording");
        }
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let probe = path.clone();
        let (check, status) = tokio::task::spawn_blocking(move || -> Result<(Check, Status)> {
            let recording = Recording::open(&probe)?;
            let (check, _) = compat::check(&recording)?;
            Ok((check, memory::status(&recording)?))
        })
        .await??;
        let opened_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
        *self.open.write().unwrap() = Some(Open { path: path.display().to_string(), name, opened_ms });
        *self.check.write().unwrap() = Some(check);
        let has_memory = status.state != "none";
        *self.status.write().unwrap() = Some(status);
        let same = self.loaded.read().unwrap().as_ref().is_some_and(|l| l.path == path);
        if !same {
            *self.loaded.write().unwrap() = None;
            *self.last_answer.write().unwrap() = None;
            if has_memory {
                self.load_memory(path);
            }
        }
        Ok(())
    }

    /// Reads the open recording's memory into RAM, in the background.
    pub fn load_memory(self: &Arc<Self>, path: PathBuf) {
        if self.loading.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let result = Recording::open(&path).and_then(|recording| memory::load(&recording));
            match result {
                Ok(Some(memory)) => {
                    let loaded = Loaded { path: path.clone(), memory, load_ms: started.elapsed().as_secs_f64() * 1000.0, background: Mutex::new(None), cloud: Mutex::new(None) };
                    eprintln!("memworld: loaded {} keyframes of {} in {:.0} ms", loaded.memory.frames.len(), path.display(), loaded.load_ms);
                    if this.open.read().unwrap().as_ref().is_some_and(|o| Path::new(&o.path) == path) {
                        *this.loaded.write().unwrap() = Some(Arc::new(loaded));
                    }
                }
                Ok(None) => {}
                Err(error) => eprintln!("memworld: loading the memory of {} failed: {error:#}", path.display()),
            }
            this.loading.store(false, Ordering::SeqCst);
        });
    }

    pub fn refresh_status(&self, path: &Path) -> Result<()> {
        let recording = Recording::open(path)?;
        *self.status.write().unwrap() = Some(memory::status(&recording)?);
        Ok(())
    }

    /// The models, loaded once (downloads must be complete).
    pub async fn runtime(&self) -> Result<Arc<Runtime>> {
        let mut guard = self.runtime.lock().await;
        if let Some(runtime) = guard.as_ref() {
            return Ok(runtime.clone());
        }
        if !self.downloads.ready() {
            bail!("the models aren't downloaded yet: POST api/models/download (GET api/models shows progress)");
        }
        let dir = self.downloads.dir.clone();
        let runtime = tokio::task::spawn_blocking(move || -> Result<Runtime> {
            let started = Instant::now();
            let members = MEMBERS.iter().map(|key| Member::load(&dir, key)).collect::<Result<Vec<_>>>()?;
            let owl = Owl::load(&dir)?;
            eprintln!("memworld: models loaded in {:.1} s", started.elapsed().as_secs_f64());
            Ok(Runtime { members, owl })
        })
        .await??;
        let runtime = Arc::new(runtime);
        *guard = Some(runtime.clone());
        Ok(runtime)
    }

    /// Generation of the open recording, in the background: models first (downloaded if missing), then the embedding.
    pub fn start_generation(self: &Arc<Self>, options: generate::Options) -> Result<()> {
        let path = self.open_path()?;
        if !self.check.read().unwrap().as_ref().is_some_and(|c| c.compatible) {
            bail!("the open recording can't be used: GET api/status shows what it's missing");
        }
        if self.generating.swap(true, Ordering::SeqCst) {
            bail!("a generation is already running (GET api/generate)");
        }
        self.cancel.store(false, Ordering::SeqCst);
        // remembered until it ends, so a restart of this server (Desktop restarts app servers when it updates) picks it up again
        let _ = std::fs::create_dir_all(&self.data_dir);
        let _ = std::fs::write(self.resume_file(), json!({ "path": path, "options": options }).to_string());
        *self.generation.lock().unwrap() = Some(Progress { phase: "models".into(), message: "getting the models ready".into(), ..Default::default() });
        let this = self.clone();
        tokio::spawn(async move {
            let result = this.generate(path.clone(), options).await;
            if let Err(error) = &result {
                let mut progress = this.generation.lock().unwrap();
                let progress = progress.get_or_insert_with(Progress::default);
                if progress.phase != "cancelled" {
                    progress.phase = "failed".into();
                    progress.message = format!("{error:#}");
                }
                eprintln!("memworld: generation failed: {error:#}");
            }
            let _ = this.refresh_status(&path);
            *this.loaded.write().unwrap() = None;
            let _ = std::fs::remove_file(this.resume_file());
            this.generating.store(false, Ordering::SeqCst);
            this.load_memory(path);
        });
        Ok(())
    }

    async fn generate(self: &Arc<Self>, path: PathBuf, options: generate::Options) -> Result<()> {
        if !self.downloads.ready() {
            let job = self.downloads.start();
            // while it downloads, the generation's progress mirrors the download's
            let this = self.clone();
            let mirror = tokio::spawn(async move {
                loop {
                    let models = this.downloads.snapshot();
                    if let Some(progress) = this.generation.lock().unwrap().as_mut() {
                        progress.phase = "models".into();
                        progress.done = models.done_bytes;
                        progress.total = models.total_bytes;
                        progress.eta_s = models.eta_s;
                        progress.message = format!("downloading the models: {:.2} of {:.2} GB", models.done_bytes as f64 / 1e9, models.total_bytes as f64 / 1e9);
                    }
                    if !models.downloading {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                }
            });
            job.await??;
            let _ = mirror.await;
            if self.cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
        }
        let runtime = self.runtime().await?;
        let this = self.clone();
        let cancel = self.cancel.clone();
        tokio::task::spawn_blocking(move || {
            let report = |progress: &Progress| *this.generation.lock().unwrap() = Some(progress.clone());
            generate::generate(&path, &runtime.members, &runtime.owl, &options, &report, &cancel).map(|_| ())
        })
        .await?
    }

    fn resume_file(&self) -> PathBuf {
        self.data_dir.join("generating.json")
    }

    /// At start: a generation this server was running when it stopped (a Desktop update, a crash) continues where its
    /// last batch ended (generation resumes from what's in the recording).
    pub async fn resume_interrupted(self: &Arc<Self>) {
        let Ok(text) = std::fs::read_to_string(self.resume_file()) else { return };
        let Ok(saved) = serde_json::from_str::<Value>(&text) else { return };
        let (Some(path), Ok(options)) = (saved["path"].as_str(), serde_json::from_value::<generate::Options>(saved["options"].clone())) else { return };
        eprintln!("memworld: resuming the generation of {path} (this server stopped while it ran)");
        let result = match self.open(PathBuf::from(path)).await {
            Ok(()) => self.start_generation(options),
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            eprintln!("memworld: couldn't resume: {error:#}");
            let _ = std::fs::remove_file(self.resume_file());
        }
    }

    /// The loaded memory of the open recording, waiting for a load in flight.
    pub async fn memory(&self) -> Result<Arc<Loaded>> {
        for _ in 0..600 {
            if let Some(loaded) = self.loaded.read().unwrap().clone() {
                return Ok(loaded);
            }
            if !self.loading.load(Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let state = self.status.read().unwrap().as_ref().map(|s| s.state).unwrap_or("none");
        if self.open.read().unwrap().is_none() {
            bail!("no recording is open: POST api/open first");
        }
        bail!("the open recording has no memory yet (memory: {state}): POST api/generate builds it")
    }

    pub async fn ask(&self, text: &str, options: query::Options) -> Result<Answer> {
        let text = text.trim();
        if text.is_empty() {
            bail!("an empty question");
        }
        let _one_at_a_time = self.asking.lock().await;
        let loaded = self.memory().await?;
        let runtime = self.runtime().await?;
        let text = text.to_string();
        let answer = tokio::task::spawn_blocking(move || -> Result<Answer> {
            let background = {
                let mut cached = loaded.background.lock().unwrap();
                if cached.is_none() {
                    *cached = Some(Arc::new(query::background(&loaded.memory, &runtime.members)?));
                }
                cached.clone().unwrap()
            };
            let engine = query::Engine { memory: &loaded.memory, members: &runtime.members, owl: Some(&runtime.owl), background: &background };
            engine.ask(&text, &options)
        })
        .await??;
        *self.last_answer.write().unwrap() = Some(answer.clone());
        Ok(answer)
    }
}
