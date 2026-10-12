//! The model files, downloaded at run time (never shipped with the app): the three SigLIP2 members from this repo's
//! `models-v1` release (made by tools/export_models.py) and OWLv2 from onnx-community on Hugging Face, pinned to a
//! commit. Each file's size and sha256 are fixed here, so a download is checked before it's used.
use serde::Serialize;
use std::path::{Path, PathBuf};

pub const RELEASE: &str = "https://github.com/jeff-hykin/dim-mem-world-v1/releases/download/models-v1";
pub const OWL_REPO: &str = "https://huggingface.co/onnx-community/owlv2-base-patch16-ensemble-ONNX/resolve/180d6ef7599bd69bb48db5c72bd49ad03ddfab80";

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

const SIGLIP: [(&str, &str, u64, &str); 14] = [
    ("base224.json", "base224", 137, "6f86bc65c6e02ca8ec0f154ef7423b9839cdd59202b368e947aeefd43c59a1d0"),
    ("base224.text.onnx", "base224", 565164929, "46deb3780ef14d584bd04d632ee608a4f97d8478ac72850136b315a0ecca1136"),
    ("base224.tokenizer.json", "base224", 34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
    ("base224.vision.onnx", "base224", 371759162, "abd4df0c111da75cdc7a719aad09840338f3db7e0261424a8961e81a174dbadc"),
    ("base_naflex.json", "base_naflex", 164, "b77d3a0d8fdce325533d79344730560257223ba488222ae6ec7f84ca7a1fa59a"),
    ("base_naflex.pos_grid.bin", "base_naflex", 786432, "3ba1ba032ad8d97e0a1afebf4513615fbfedb56f646c14dcdb83d3c228c12860"),
    ("base_naflex.text.onnx", "base_naflex", 565164929, "1806dfca25871868621cf78a5b5c6071f5a12f6dde63de7f2ef71724422d73f2"),
    ("base_naflex.tokenizer.json", "base_naflex", 34356304, "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0"),
    ("base_naflex.vision.onnx", "base_naflex", 371133214, "bfc9e5176ad5ee99be6d314b9ce1960a6b2bfc90956077fe3a7ebd54ed0051d2"),
    ("so400m_naflex.json", "so400m_naflex", 167, "fa2a6872ec99cbdbd6eca29953508c55b43e8fc3a1eb3c96d1c13d5476de320b"),
    ("so400m_naflex.pos_grid.bin", "so400m_naflex", 1179648, "2596cba5a6a2fa5e523a707a562a5754a4c1d6001f562f025062ad545f0a4df3"),
    ("so400m_naflex.text.onnx", "so400m_naflex", 1416880080, "38b97a14d0b4206d223f3539bd29356ef80441b7c86097cd5725b581892a140b"),
    ("so400m_naflex.tokenizer.json", "so400m_naflex", 34356304, "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0"),
    ("so400m_naflex.vision.onnx", "so400m_naflex", 1710791373, "4cded3f8921adb0dd9c2746b3d2c753613b4595174cf87c49e1f97ade8df155e"),
];

pub fn files() -> Vec<ModelFile> {
    let mut files: Vec<ModelFile> = SIGLIP.iter().map(|&(name, model, bytes, sha256)| ModelFile { name, model, url: format!("{RELEASE}/{name}"), bytes, sha256 }).collect();
    files.push(ModelFile { name: "owlv2.onnx", model: "owlv2", url: format!("{OWL_REPO}/onnx/model.onnx"), bytes: 614219112, sha256: "ca3ea10811d55cd8763058c6eaddec514bbb91cc07ab774a0db0775346aef732" });
    files.push(ModelFile { name: "owlv2.tokenizer.json", model: "owlv2", url: format!("{OWL_REPO}/tokenizer.json"), bytes: 3642208, sha256: "e277946093d72c7748281a6a344d6c79a5226c48954d6797dc36984aea23ac60" });
    files
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
        assert!(files().iter().all(|f| f.sha256.len() == 64 && f.bytes > 0));
        assert!(OWL_REPO.contains("/resolve/") && !OWL_REPO.ends_with("/main"));
        // about 5.6 GB in all: the progress bar's denominator
        assert!((5_000_000_000..6_000_000_000).contains(&total_bytes()));
    }
}
