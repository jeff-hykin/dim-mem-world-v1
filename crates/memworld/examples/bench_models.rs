//! Per-keyframe cost of each image model on this machine: `MEMWORLD_DEVICE=cpu|coreml|cuda cargo run --release
//! --example bench_models -- <models dir> <frame.jpg>`.
use memworld::{onnx, owl, siglip};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let rgb = owl::decode_jpeg(&std::fs::read(&args[2])?)?;
    let time = |name: &str, device: onnx::Device, f: &dyn Fn() -> anyhow::Result<()>| -> anyhow::Result<()> {
        let first = Instant::now();
        f()?;
        println!("{name:>14} first run (compile or cached load) {:.1} s", first.elapsed().as_secs_f64());
        let start = Instant::now();
        for _ in 0..5 {
            f()?;
        }
        println!("{name:>14} on {device:?}: {:.0} ms", start.elapsed().as_secs_f64() * 1000.0 / 5.0);
        Ok(())
    };
    for key in ["base224", "base_naflex", "so400m_naflex"] {
        let load = Instant::now();
        let member = siglip::Member::load(dir, key)?;
        println!("{key:>14} load {:.1} s", load.elapsed().as_secs_f64());
        time(key, member.device(), &|| member.embed_grid(&rgb).map(|_| ()))?;
    }
    let load = Instant::now();
    let owl = owl::Owl::load(dir)?;
    println!("{:>14} load {:.1} s", "owlv2", load.elapsed().as_secs_f64());
    time("owlv2", owl.device(), &|| owl.features(&rgb).map(|_| ()))?;
    // one keyframe the way generation does it: in turn, then all four at once
    let members: Vec<siglip::Member> = ["base224", "base_naflex", "so400m_naflex"].iter().map(|key| siglip::Member::load(dir, key)).collect::<anyhow::Result<_>>()?;
    time("in turn", owl.device(), &|| {
        for member in &members {
            member.embed_grid(&rgb)?;
        }
        owl.features(&rgb).map(|_| ())
    })?;
    time("at once", owl.device(), &|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = members.iter().map(|member| scope.spawn(|| member.embed_grid(&rgb).map(|_| ()))).collect();
            let owl_result = owl.features(&rgb).map(|_| ());
            handles.into_iter().try_for_each(|handle| handle.join().unwrap())?;
            owl_result
        })
    })?;
    Ok(())
}
