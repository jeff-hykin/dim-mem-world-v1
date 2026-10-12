//! The one message shape Memory World stores in a recording: `MWB1`, a u32 (LE) length, a JSON header, then raw
//! bytes (a JPEG, fp16 patch embeddings, zstd depth). Self-describing, so a reader needs no schema registry, and the
//! same bytes go into a `.db` row (pickled as Python `bytes`) or an mcap message.
use anyhow::{bail, Context, Result};
use serde_json::Value;

const MAGIC: &[u8; 4] = b"MWB1";

pub fn encode(header: &Value, body: &[u8]) -> Vec<u8> {
    let json = header.to_string();
    let mut out = Vec::with_capacity(8 + json.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(body);
    out
}

pub fn decode(bytes: &[u8]) -> Result<(Value, &[u8])> {
    if bytes.len() < 8 || &bytes[..4] != MAGIC {
        bail!("not a Memory World message");
    }
    let length = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let json = bytes.get(8..8 + length).context("truncated Memory World message")?;
    Ok((serde_json::from_slice(json)?, &bytes[8 + length..]))
}

pub fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
}

pub fn f16_values(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_body_round_trip() {
        let bytes = encode(&serde_json::json!({ "rows": 2 }), &[1, 2, 3]);
        let (header, body) = decode(&bytes).unwrap();
        assert_eq!((header["rows"].as_u64(), body), (Some(2), &[1u8, 2, 3][..]));
        assert!(decode(b"nope").is_err());
        assert!(decode(&bytes[..10]).is_err());
    }

    #[test]
    fn fp16_keeps_unit_vectors_to_three_decimals() {
        let values = [0.123456f32, -0.5, 0.0, 0.999];
        let back = f16_values(&f16_bytes(&values));
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
