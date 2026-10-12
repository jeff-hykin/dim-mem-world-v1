//! Model downloads, at run time: each missing file is streamed to `<name>.part` (resumed with a Range request when a
//! partial one is there), its sha256 checked, then renamed into place. One download job at a time; its state is what
//! `GET api/models` shows (and the page's progress bar).
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use memworld::models::{self, ModelFile};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileState {
    pub name: &'static str,
    pub model: &'static str,
    pub bytes: u64,
    pub done_bytes: u64,
    /// missing, downloading, verifying, ready, failed
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Models {
    pub dir: String,
    pub ready: bool,
    pub total_bytes: u64,
    pub done_bytes: u64,
    pub bytes_per_second: Option<f64>,
    pub eta_s: Option<f64>,
    pub downloading: bool,
    pub error: Option<String>,
    pub files: Vec<FileState>,
}

pub struct Downloads {
    pub dir: PathBuf,
    state: Mutex<Models>,
    /// (when the running job started, bytes it had then) for the speed
    started: Mutex<Option<(Instant, u64)>>,
}

impl Downloads {
    pub fn new(dir: PathBuf) -> Arc<Downloads> {
        let downloads = Arc::new(Downloads {
            state: Mutex::new(Models { dir: dir.display().to_string(), ready: false, total_bytes: 0, done_bytes: 0, bytes_per_second: None, eta_s: None, downloading: false, error: None, files: Vec::new() }),
            started: Mutex::new(None),
            dir,
        });
        downloads.refresh();
        downloads
    }

    /// Re-reads the folder (unless a download is running, which keeps its own counts).
    pub fn refresh(&self) {
        let mut state = self.state.lock().unwrap();
        if state.downloading {
            return;
        }
        state.files = models::files()
            .into_iter()
            .map(|file| {
                let ready = models::present(&self.dir, &file);
                let partial = std::fs::metadata(part(&self.dir, &file)).map(|m| m.len()).unwrap_or(0);
                FileState { name: file.name, model: file.model, bytes: file.bytes, done_bytes: if ready { file.bytes } else { partial }, state: if ready { "ready" } else { "missing" }, error: None }
            })
            .collect();
        state.total_bytes = state.files.iter().map(|f| f.bytes).sum();
        state.done_bytes = state.files.iter().map(|f| f.done_bytes).sum();
        state.ready = state.files.iter().all(|f| f.state == "ready");
    }

    pub fn snapshot(&self) -> Models {
        self.refresh();
        self.state.lock().unwrap().clone()
    }

    pub fn ready(&self) -> bool {
        self.snapshot().ready
    }

    fn update(&self, name: &str, change: impl FnOnce(&mut FileState)) {
        let mut state = self.state.lock().unwrap();
        if let Some(file) = state.files.iter_mut().find(|f| f.name == name) {
            change(file);
        }
        state.done_bytes = state.files.iter().map(|f| f.done_bytes).sum();
        if let Some((since, at_start)) = *self.started.lock().unwrap() {
            let seconds = since.elapsed().as_secs_f64();
            if seconds > 1.0 {
                let rate = state.done_bytes.saturating_sub(at_start) as f64 / seconds;
                state.bytes_per_second = Some(rate);
                state.eta_s = (rate > 0.0).then(|| state.total_bytes.saturating_sub(state.done_bytes) as f64 / rate);
            }
        }
    }

    /// Starts downloading what's missing (no-op if it's running or nothing is missing). Resolves when it finishes.
    pub fn start(self: &Arc<Self>) -> tokio::task::JoinHandle<Result<()>> {
        let this = self.clone();
        {
            self.refresh();
            let mut state = self.state.lock().unwrap();
            if state.downloading || state.ready {
                return tokio::spawn(async { Ok(()) });
            }
            state.downloading = true;
            state.error = None;
            *self.started.lock().unwrap() = Some((Instant::now(), state.done_bytes));
        }
        tokio::spawn(async move {
            let result = this.run().await;
            let mut state = this.state.lock().unwrap();
            state.downloading = false;
            state.bytes_per_second = None;
            state.eta_s = None;
            if let Err(error) = &result {
                state.error = Some(format!("{error:#}"));
            }
            drop(state);
            this.refresh();
            result
        })
    }

    async fn run(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.dir).await.with_context(|| format!("creating {}", self.dir.display()))?;
        let client = reqwest::Client::builder().user_agent("dim-mem-world").build()?;
        for file in models::missing(&self.dir) {
            if let Err(error) = self.fetch(&client, &file).await {
                self.update(file.name, |f| {
                    f.state = "failed";
                    f.error = Some(format!("{error:#}"));
                });
                return Err(error.context(format!("downloading {}", file.name)));
            }
        }
        Ok(())
    }

    async fn fetch(&self, client: &reqwest::Client, file: &ModelFile) -> Result<()> {
        let partial = part(&self.dir, file);
        let mut have = tokio::fs::metadata(&partial).await.map(|m| m.len()).unwrap_or(0);
        if have > file.bytes {
            tokio::fs::remove_file(&partial).await?;
            have = 0;
        }
        self.update(file.name, |f| {
            f.state = "downloading";
            f.done_bytes = have;
        });
        if have < file.bytes {
            let mut request = client.get(&file.url);
            if have > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
            }
            let response = request.send().await?.error_for_status()?;
            // a server that ignores Range sends the whole file: start over
            if have > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                have = 0;
            }
            let mut out = tokio::fs::OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(&partial).await?;
            let mut stream = response.bytes_stream();
            let mut done = have;
            let mut last_report = Instant::now();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                out.write_all(&chunk).await?;
                done += chunk.len() as u64;
                if last_report.elapsed().as_millis() > 200 {
                    last_report = Instant::now();
                    self.update(file.name, |f| f.done_bytes = done);
                }
            }
            out.flush().await?;
            self.update(file.name, |f| f.done_bytes = done);
            if done != file.bytes {
                bail!("{}: got {} bytes, expected {}", file.name, done, file.bytes);
            }
        }
        self.update(file.name, |f| f.state = "verifying");
        let path = partial.clone();
        let digest = tokio::task::spawn_blocking(move || sha256(&path)).await??;
        if digest != file.sha256 {
            tokio::fs::remove_file(&partial).await?;
            bail!("{}: checksum mismatch (got {digest}), deleted; try again", file.name);
        }
        tokio::fs::rename(&partial, self.dir.join(file.name)).await?;
        self.update(file.name, |f| {
            f.state = "ready";
            f.done_bytes = file.bytes;
        });
        Ok(())
    }
}

fn part(dir: &Path, file: &ModelFile) -> PathBuf {
    dir.join(format!("{}.part", file.name))
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
