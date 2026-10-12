//! Memory World's `dimos-app-server` (dimOS Desktop app contract: docs/apps.md in dimos-desktop): serves the built page
//! and every action as an HTTP endpoint (api.rs), listed in agent.json for Desktop's agent. One compiled binary; the
//! models are downloaded at run time and run on ONNX Runtime (no Python, no API key).
//!
//! The same binary answers from a terminal: `--check <rec>`, `--generate <rec>`, `--ask <rec> <text>`, `--forget <rec>`.
mod api;
mod app;
mod dimos_app;
mod downloads;
mod render;
mod routes;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

#[derive(Parser, Debug, Clone)]
#[command(about = "Memory World backend: a robot recording's visual memory, and where things are in 3D")]
pub struct Args {
    /// serve HTTP on this TCP port instead of DIMOS_APP's socket (development, tests)
    #[arg(long)]
    port: Option<u16>,
    /// the built page (vite's dist); the nix wrapper sets it
    #[arg(long, env = "MEMWORLD_FRONTEND")]
    frontend: Option<PathBuf>,
    /// print the endpoints (agent.json) and exit
    #[arg(long)]
    agent_json: bool,
    /// print a recording's compatibility check and memory state, then exit
    #[arg(long, value_name = "RECORDING")]
    check: Option<PathBuf>,
    /// download the models if missing, build a recording's memory in place, then exit
    #[arg(long, value_name = "RECORDING")]
    generate: Option<PathBuf>,
    /// with --generate: one keyframe per this many seconds
    #[arg(long, default_value_t = 1.0)]
    every: f64,
    /// forget a recording's memory (drop its tables / clear its mcap memory), then exit
    #[arg(long, value_name = "RECORDING")]
    forget: Option<PathBuf>,
    /// ask a recording's memory (`--ask <recording> <text>`), print the answer as JSON, then exit
    #[arg(long, num_args = 2, value_names = ["RECORDING", "TEXT"])]
    ask: Option<Vec<String>>,
}

fn dimos_home() -> PathBuf {
    std::env::var("DIMOS_HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".dimos"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let desktop = dimos_app::get().cloned().unwrap_or_default();
    if args.agent_json {
        println!("{}", serde_json::to_string_pretty(&api::routes().manifest(api::DESCRIPTION))?);
        return Ok(());
    }
    let data = desktop.data_dir.map(PathBuf::from).unwrap_or_else(|| dimos_home().join("data").join("apps").join("dim-mem-world"));
    let recordings = desktop.recordings_dir.map(PathBuf::from).unwrap_or_else(|| dimos_home().join("recordings"));
    let state = app::App::new(data.clone(), recordings.clone());

    if let Some(path) = args.check {
        let recording = memworld::recording::Recording::open(&path)?;
        let (check, _) = memworld::compat::check(&recording)?;
        let mut value = serde_json::to_value(check)?;
        value["memory"] = serde_json::to_value(memworld::memory::status(&recording)?)?;
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    if let Some(path) = args.forget {
        memworld::generate::clear(&path)?;
        eprintln!("forgot the memory of {}", path.display());
        return Ok(());
    }
    if let Some(path) = args.generate {
        if !state.downloads.ready() {
            eprintln!("downloading the models into {}", state.downloads.dir.display());
            state.downloads.start().await??;
        }
        let runtime = state.runtime().await?;
        let options = memworld::generate::Options { every_seconds: args.every, ..Default::default() };
        let cancel = AtomicBool::new(false);
        let progress = tokio::task::spawn_blocking(move || {
            let report = |p: &memworld::generate::Progress| eprintln!("{:>10} {:>5}/{:<5} eta {:>6.0}s  {}", p.phase, p.done, p.total, p.eta_s.unwrap_or(0.0), p.message);
            memworld::generate::generate(&path, &runtime.members, &runtime.owl, &options, &report, &cancel)
        })
        .await??;
        println!("{}", serde_json::to_string_pretty(&progress)?);
        return Ok(());
    }
    if let Some(ask) = args.ask {
        let path = state.resolve(&ask[0])?;
        state.open(path.clone()).await?;
        let answer = state.ask(&ask[1], memworld::query::Options::default()).await?;
        println!("{}", serde_json::to_string_pretty(&answer)?);
        return Ok(());
    }

    let frontend = args.frontend.clone().unwrap_or_else(|| PathBuf::from("frontend/dist"));
    eprintln!("memory world: page {}, data {}, recordings {}", frontend.display(), data.display(), recordings.display());
    let router = api::routes()
        .router
        .with_state(state.clone())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(16 << 20))
        .fallback_service(tower_http::services::ServeDir::new(&frontend).fallback(tower_http::services::ServeFile::new(frontend.join("index.html"))));
    let shutdown = async {
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    };
    if let Some(port) = args.port {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.with_context(|| format!("bind :{port}"))?;
        eprintln!("memory world: http://127.0.0.1:{port}/");
        axum::serve(listener, router).with_graceful_shutdown(shutdown).await?;
    } else {
        let socket = desktop.socket.map(PathBuf::from).context("DIMOS_APP's socket (or --port) is required")?;
        let _ = std::fs::remove_file(&socket);
        let listener = tokio::net::UnixListener::bind(&socket).with_context(|| format!("bind {}", socket.display()))?;
        axum::serve(listener, router).with_graceful_shutdown(shutdown).await?;
    }
    Ok(())
}
