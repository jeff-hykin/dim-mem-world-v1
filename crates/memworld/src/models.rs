//! The model files, downloaded at run time (never shipped with the app), from this repo's `models-v1` release (the SigLIP2
//! vision graphs from `models-v2`: the same math, rewritten so CoreML runs them whole, tools/coreml_friendly.py)
//! (made by tools/export_models.py from google/siglip2-* and google/owlv2-base-patch16-ensemble): the three SigLIP2
//! members and OWLv2 split into its image and text halves. Each file's size and sha256 are fixed here, so a download is
//! checked before it's used.
use serde::Serialize;
use std::path::{Path, PathBuf};

pub const RELEASES: &str = "https://github.com/jeff-hykin/dim-mem-world-v1/releases/download";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    pub name: &'static str,
    /// what it's part of: a SigLIP2 member's key, or "owlv2"
    pub model: &'static str,
    pub url: String,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// (release, name, model, bytes, sha256)
const FILES: [(&str, &str, &str, u64, &str); 17] = [
    ("models-v1", "base224.json", "base224", 137, "6f86bc65c6e02ca8ec0f154ef7423b9839cdd59202b368e947aeefd43c59a1d0"),
    ("models-v1", "base224.text.onnx", "base224", 565164929, "46deb3780ef14d584bd04d632ee608a4f97d8478ac72850136b315a0ecca1136"),
    ("models-v1", "base224.tokenizer.json", "base224", 34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
    ("models-v2", "base224.vision.onnx", "base224", 371759317, "496030179bd008ca2f97814c2eea6ce6eb69f2992f70f5e13fc861fbfc476d54"),
    ("models-v1", "base_naflex.json", "base_naflex", 164, "b77d3a0d8fdce325533d79344730560257223ba488222ae6ec7f84ca7a1fa59a"),
    ("models-v1", "base_naflex.pos_grid.bin", "base_naflex", 786432, "3ba1ba032ad8d97e0a1afebf4513615fbfedb56f646c14dcdb83d3c228c12860"),
    ("models-v1", "base_naflex.text.onnx", "base_naflex", 565164929, "1806dfca25871868621cf78a5b5c6071f5a12f6dde63de7f2ef71724422d73f2"),
    ("models-v1", "base_naflex.tokenizer.json", "base_naflex", 34356304, "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0"),
    ("models-v2", "base_naflex.vision.onnx", "base_naflex", 371133372, "e8e8b4d0d58448554300d3aff4cb5bb53ee5aeb961dbbd6cf434855abf507149"),
    ("models-v1", "so400m_naflex.json", "so400m_naflex", 167, "fa2a6872ec99cbdbd6eca29953508c55b43e8fc3a1eb3c96d1c13d5476de320b"),
    ("models-v1", "so400m_naflex.pos_grid.bin", "so400m_naflex", 1179648, "2596cba5a6a2fa5e523a707a562a5754a4c1d6001f562f025062ad545f0a4df3"),
    ("models-v1", "so400m_naflex.text.onnx", "so400m_naflex", 1416880080, "38b97a14d0b4206d223f3539bd29356ef80441b7c86097cd5725b581892a140b"),
    ("models-v1", "so400m_naflex.tokenizer.json", "so400m_naflex", 34356304, "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0"),
    ("models-v2", "so400m_naflex.vision.onnx", "so400m_naflex", 1710791531, "d2f41c30462c00aba50c017aa680957a872f2092d625f3546c7bdb3709e9195a"),
    ("models-v1", "owlv2.vision.onnx", "owlv2", 360215906, "1dbfa2e70360b36e64ccb68915ed08287b3b8ada4b6b34417a33fd2879f5bc3b"),
    ("models-v1", "owlv2.text.onnx", "owlv2", 253937349, "980e20df55f2b65d3b8cd9f97f0f704fb589cf680b63a4f54111cdf07e656dc2"),
    ("models-v1", "owlv2.tokenizer.json", "owlv2", 3642370, "fbf857dd4099afc6da394d17cba2f22c5d01eb6d87bb74967c90634f3bcf4e82"),
];

/// A Mac also gets the image models as native CoreML packages (tools/export_coreml.py): its GPU runs those, and the
/// ONNX files above stay as the CPU fallback.
const MAC_FILES: [(&str, &str, &str, u64, &str); 4] = [
    ("models-v3", "base224.vision.mlpackage.tar", "base224", 183570432, "3d4e78471f25b0c33718fd79c58399e91ff045bbfcbedc2cb6a22f226269f7ca"),
    ("models-v3", "base_naflex.vision.mlpackage.tar", "base_naflex", 183271424, "fc765398e7f156b16819aa22bb893022bedff54609d8b3c558ab9e3cb5b162ef"),
    ("models-v3", "so400m_naflex.vision.mlpackage.tar", "so400m_naflex", 850214912, "fe5ca4dd15af40e2eedfe596a4c0eef260b5105427febeb052d8b2f92a958a31"),
    ("models-v3", "owlv2.vision.mlpackage.tar", "owlv2", 180380672, "09294c4e9462a0cebfc3ac2c95fd013ff7826a9a221533fd8a8692580517eac9"),
];

pub fn files() -> Vec<ModelFile> {
    let mac: &[(&str, &str, &str, u64, &str)] = if cfg!(target_os = "macos") { &MAC_FILES } else { &[] };
    FILES.iter().chain(mac).map(|&(release, name, model, bytes, sha256)| ModelFile { name, model, url: format!("{RELEASES}/{release}/{name}"), bytes, sha256 }).collect()
}

/// Where the models live: `$MEMWORLD_MODELS`, else `<app data>/models`.
pub fn dir(data_dir: &Path) -> PathBuf {
    std::env::var("MEMWORLD_MODELS").map(PathBuf::from).unwrap_or_else(|_| data_dir.join("models"))
}

/// A file counts as present when it has its full size (the sha256 is checked once, right after the download).
pub fn present(dir: &Path, file: &ModelFile) -> bool {
    std::fs::metadata(dir.join(file.name)).map(|m| m.len() == file.bytes).unwrap_or(false)
}

pub fn missing(dir: &Path) -> Vec<ModelFile> {
    files().into_iter().filter(|f| !present(dir, f)).collect()
}

pub fn total_bytes() -> u64 {
    files().iter().map(|f| f.bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_member_has_its_files_and_owl_is_pinned() {
        for member in crate::siglip::MEMBERS {
            for part in ["json", "vision.onnx", "text.onnx", "tokenizer.json"] {
                assert!(files().iter().any(|f| f.name == format!("{member}.{part}")), "{member}.{part}");
            }
        }
        for part in ["vision.onnx", "text.onnx", "tokenizer.json"] {
            assert!(files().iter().any(|f| f.name == format!("owlv2.{part}")), "owlv2.{part}");
        }
        assert!(files().iter().all(|f| f.sha256.len() == 64 && f.bytes > 0 && f.url.starts_with(RELEASES)));
        // about 5.7 GB in all: the progress bar's denominator
        assert!((5_000_000_000..8_000_000_000).contains(&total_bytes()));
    }
}
