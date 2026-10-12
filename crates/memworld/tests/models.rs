//! The real models against transformers: a RealSense frame from sf_office embedded in Rust (our preprocessing, the
//! exported ONNX graphs) must match what transformers' processor + full vision tower give for the same JPEG
//! (tests/fixtures/reference.json, made by `tools/export_models.py --fixture`).
//!
//! Needs the downloaded models (`MEMWORLD_MODELS`) and ONNX Runtime (`ORT_DYLIB_PATH`); `nix develop` sets the
//! runtime, and the app's own download fills the models folder. Without them the tests say so and pass vacuously.
use memworld::recording::image::Rgb;
use memworld::siglip::{Member, MEMBERS};
use serde_json::Value;
use std::path::PathBuf;

fn models() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("MEMWORLD_MODELS").ok()?);
    let complete = memworld::models::missing(&dir).is_empty() && std::env::var("ORT_DYLIB_PATH").is_ok();
    if !complete {
        eprintln!("SKIPPED: set MEMWORLD_MODELS (a complete models folder) and ORT_DYLIB_PATH to run the model tests");
    }
    complete.then_some(dir)
}

fn fixture() -> (Rgb, Value) {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let image = image::open(here.join("frame_sf_office.jpg")).unwrap().to_rgb8();
    let rgb = Rgb { width: image.width(), height: image.height(), data: image.into_raw() };
    let reference = serde_json::from_slice(&std::fs::read(here.join("reference.json")).unwrap()).unwrap();
    (rgb, reference)
}

fn floats(value: &Value) -> Vec<f32> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt())
}

#[test]
fn per_patch_embeddings_match_transformers() {
    let Some(dir) = models() else { return };
    let (rgb, reference) = fixture();
    for key in MEMBERS {
        let member = Member::load(&dir, key).unwrap();
        let grid = member.embed_grid(&rgb).unwrap();
        let expected = &reference[key];
        assert_eq!((grid.rows as u64, grid.cols as u64), (expected["rows"].as_u64().unwrap(), expected["cols"].as_u64().unwrap()), "{key} grid");
        let mean: Vec<f32> = (0..grid.dim).map(|d| grid.data.chunks_exact(grid.dim).map(|row| row[d]).sum::<f32>() / (grid.rows * grid.cols) as f32).collect();
        let mean_cosine = cosine(&mean, &floats(&expected["mean"]));
        let mut worst = 1.0f32;
        for (index, values) in expected["patches"].as_object().unwrap() {
            let i: usize = index.parse().unwrap();
            worst = worst.min(cosine(&grid.data[i * grid.dim..(i + 1) * grid.dim], &floats(values)));
        }
        eprintln!("{key}: grid {}x{}, mean cosine {mean_cosine:.4}, worst sampled patch cosine {worst:.4}", grid.rows, grid.cols);
        // the resize differs slightly from PIL's; the embeddings must still be the same embeddings
        assert!(mean_cosine > 0.99, "{key}: mean cosine {mean_cosine}");
        assert!(worst > 0.95, "{key}: a patch's cosine is {worst}");
    }
}

#[test]
fn text_tokens_and_embeddings_match_transformers() {
    let Some(dir) = models() else { return };
    let (_, reference) = fixture();
    for key in MEMBERS {
        let member = Member::load(&dir, key).unwrap();
        let ids = member.tokens(&["a traffic cone"]).unwrap();
        let expected_ids: Vec<i64> = reference[key]["text_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert_eq!(ids, expected_ids, "{key} tokens");
        let text = member.embed_text(&["a traffic cone"]).unwrap().remove(0);
        let similarity = cosine(&text, &floats(&reference[key]["text"]));
        eprintln!("{key}: text cosine {similarity:.4}");
        // the shipped text towers are the small ones (fp16 table, int8 matmuls)
        assert!(similarity > 0.995, "{key}: text cosine {similarity}");
    }
}

#[test]
fn owl_finds_the_traffic_cone_in_the_frame() {
    let Some(dir) = models() else { return };
    let (rgb, _) = fixture();
    let owl = memworld::owl::Owl::load(&dir).unwrap();
    let found = owl.detect(&rgb, "a traffic cone", 0.2).unwrap();
    eprintln!("owl: {found:?}");
    // the frame (sf_office, 188 s) has the orange cone left of center, by the crates
    let cone = found.first().expect("no cone found");
    let center = ((cone.bbox[0] + cone.bbox[2]) / 2.0, (cone.bbox[1] + cone.bbox[3]) / 2.0);
    assert!(cone.score > 0.3, "{cone:?}");
    assert!(center.0 > 100.0 && center.0 < 500.0 && center.1 > 100.0 && center.1 < 400.0, "{cone:?}");
}
