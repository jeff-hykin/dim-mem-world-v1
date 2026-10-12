//! Lists a recording's streams (.db) or channels (.mcap): name, message type, count.
//!   cargo run -p dimos-recording --example streams -- <file>
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: streams <recording.db|.mcap>"))?;
    let recording = dimos_recording::Recording::open(std::path::Path::new(&path))?;
    for stream in recording.streams()? {
        println!("{:<36} {:<32} {}", stream.name, stream.type_name, stream.count);
    }
    Ok(())
}
