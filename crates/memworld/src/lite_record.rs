//! A raw lidar + IMU recording (lite_record's) has no poses yet: `lite_record post_process <file>` estimates the
//! odometry and writes it, the sensors' frame tree (/tf, cameras included), a motion-compensated copy of the lidar and
//! its voxel map INTO the recording, in place. Generation runs it first, the same way the Map Editor's Generate does.
//! The binary is part of the app (the nix build bakes lite_record in and sets LITE_RECORD_BIN).
use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub fn binary() -> Result<PathBuf> {
    std::env::var_os("LITE_RECORD_BIN")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .context("this recording needs lite_record's post_process first, and LITE_RECORD_BIN isn't set (the nix build sets it)")
}

/// What post_process says it's doing: `[Step i of N] name` starts a stage, `  42%  ...` is how far through it is.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stage {
    pub name: String,
    pub index: usize,
    pub count: usize,
    pub percent: f64,
}

pub fn parse_line(line: &str, current: &mut Stage) -> bool {
    let line = line.trim();
    if let Some(rest) = line.strip_prefix("[Step ") {
        let Some((numbers, name)) = rest.split_once("] ") else { return false };
        let Some((index, count)) = numbers.split_once(" of ") else { return false };
        let (Ok(index), Ok(count)) = (index.trim().parse::<usize>(), count.trim().parse::<usize>()) else { return false };
        *current = Stage { name: name.to_string(), index: index.saturating_sub(1), count: count.max(1), percent: 0.0 };
        return true;
    }
    if let Some((percent, _)) = line.split_once('%') {
        if let Ok(percent) = percent.trim().parse::<f64>() {
            current.percent = percent.clamp(0.0, 100.0);
            return true;
        }
    }
    if line.starts_with("done in") {
        current.percent = 100.0;
        return true;
    }
    false
}

/// The stages a kill would leave half-written: everything but reading and estimating.
fn writing(stage: &str) -> bool {
    !(stage.is_empty() || stage.contains("reading the recording") || stage.contains("estimating odometry"))
}

/// Runs post_process over `recording`, reporting each stage. Cancel stops it while it only reads; once it writes into
/// the file it finishes first (a kill there would leave the file without its index).
pub fn post_process(binary: &Path, recording: &Path, report: &mut dyn FnMut(&Stage), cancel: &AtomicBool) -> Result<String> {
    let mut child = Command::new(binary)
        .arg("post_process")
        .arg(recording)
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("couldn't run {}", binary.display()))?;
    let stdout = child.stdout.take().context("stdout")?;
    let stderr = child.stderr.take().context("stderr")?;
    let errors = std::thread::spawn(move || std::io::read_to_string(stderr).unwrap_or_default());
    let (lines, receiver) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            if lines.send(line).is_err() {
                break;
            }
        }
    });
    let mut output = String::new();
    let mut stage = Stage { name: "starting".into(), count: 1, ..Default::default() };
    report(&stage);
    let mut killed = false;
    loop {
        match receiver.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                output.push_str(&line);
                output.push('\n');
                if parse_line(&line, &mut stage) {
                    report(&stage);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if cancel.load(Ordering::Relaxed) && !killed && !writing(&stage.name) {
            let _ = child.kill();
            killed = true;
        }
    }
    let status = child.wait()?;
    let errors = errors.join().unwrap_or_default();
    remove_beside(recording);
    if killed {
        bail!("cancelled");
    }
    if !status.success() {
        let tail: Vec<&str> = errors.lines().chain(output.lines()).filter(|l| !l.trim().is_empty()).collect();
        bail!("lite_record post_process failed ({status}): {}", tail[tail.len().saturating_sub(6)..].join(" / "));
    }
    Ok(output)
}

/// What post_process leaves next to the recording (its map again as a .pc2.lcm, an interrupted run's temporaries):
/// nothing is kept beside a recording.
fn remove_beside(recording: &Path) {
    let mut found = vec![recording.with_extension("pc2.lcm")];
    for suffix in [".converting", ".deskew-spool"] {
        let mut name = recording.as_os_str().to_os_string();
        name.push(suffix);
        found.push(PathBuf::from(name));
    }
    for path in found.into_iter().filter(|path| path.exists()) {
        if let Err(error) = std::fs::remove_file(&path) {
            eprintln!("memworld: couldn't remove {}: {error}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_output_becomes_progress() {
        let mut stage = Stage::default();
        assert!(parse_line("[Step 3 of 5] estimating odometry from /rig/lidar + /rig/imu", &mut stage));
        assert_eq!((stage.index, stage.count), (2, 5));
        assert!(!writing(&stage.name));
        assert!(parse_line("   42%  1m10s  eta 1m35s  1.4x rt  2.1 GB written", &mut stage));
        assert_eq!(stage.percent, 42.0);
        assert!(parse_line("  done in 2s (6.7x rt)", &mut stage));
        assert_eq!(stage.percent, 100.0);
        assert!(writing("appending transforms, odometry and corrected clouds"));
        assert!(!parse_line("", &mut stage));
    }
}
