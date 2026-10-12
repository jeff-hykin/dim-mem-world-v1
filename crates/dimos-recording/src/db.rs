//! dimos memory2 SQLite recordings (`.db`): `_streams(name, config)` lists the streams; each stream is a table
//! `(id, ts, value, pose_x..pose_qw, tags)` plus `<name>_blob(id, data)` holding the encoded payload, `<name>_rtree`
//! a spatial index. Payload codecs: `lcm` (the raw LCM message) and `lz4+lcm` (an LZ4 frame around it).
//! Writing creates new streams the way dimos's SqliteStore does, so dimos reads them back.
use anyhow::{bail, Context, Result};
pub use rusqlite::Connection;
use rusqlite::{OpenFlags, OptionalExtension};
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct DbStream {
    pub name: String,
    /// dimos type, e.g. `sensor_msgs.PointCloud2`
    pub kind: String,
    pub codec: String,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub struct DbRow {
    pub ts: f64,
    /// the stored observation pose (x, y, z, qx, qy, qz, qw), when the recorder set one
    pub pose: Option<[f64; 7]>,
    /// the decoded (decompressed) LCM payload
    pub payload: Vec<u8>,
}

pub fn open_read(path: &Path) -> Result<Connection> {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
        .with_context(|| format!("opening {}", path.display()))
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// `dimos.msgs.sensor_msgs.PointCloud2.PointCloud2` → `sensor_msgs.PointCloud2`
pub fn dimos_type(module: &str) -> String {
    let parts: Vec<&str> = module.split('.').collect();
    match parts.as_slice() {
        ["dimos", "msgs", package, _, class] => format!("{package}.{class}"),
        _ => module.to_string(),
    }
}

pub fn streams(db: &Connection) -> Result<Vec<DbStream>> {
    let mut statement = db.prepare("SELECT name, config FROM _streams ORDER BY name").context("not a dimos recording (no _streams)")?;
    let rows: Vec<(String, String)> = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
    let mut streams = Vec::new();
    for (name, config) in rows {
        let config: serde_json::Value = serde_json::from_str(&config).unwrap_or_default();
        let kind = config["payload_module"].as_str().map(dimos_type).unwrap_or_default();
        let codec = config["codec_id"].as_str().unwrap_or("lcm").to_string();
        let count = db.query_row(&format!("SELECT count(*) FROM {}", quote(&name)), [], |row| row.get(0)).unwrap_or(0);
        streams.push(DbStream { name, kind, codec, count });
    }
    Ok(streams)
}

pub fn decode_blob(codec: &str, data: &[u8]) -> Result<Vec<u8>> {
    match codec {
        "lcm" => Ok(data.to_vec()),
        "lz4+lcm" => {
            let mut out = Vec::new();
            lz4_flex::frame::FrameDecoder::new(data).read_to_end(&mut out).context("lz4 frame")?;
            Ok(out)
        }
        "pickle" => unpickle_bytes(data).map(<[u8]>::to_vec).context("only pickled `bytes` rows are readable here"),
        other => bail!("unsupported codec {other}"),
    }
}

/// A Python `bytes` object as protocol-4 pickle (PROTO 4, BINBYTES8, STOP): what a `pickle`-codec stream of
/// `builtins.bytes` holds, so dimos's SqliteStore reads the rows this crate appends.
pub fn pickle_bytes(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 12);
    out.extend_from_slice(&[0x80, 4, 0x8e]);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
    out.push(b'.');
    out
}

/// The payload of `pickle_bytes` (also with a FRAME opcode, as Python's own pickler writes it), or None.
pub fn unpickle_bytes(data: &[u8]) -> Option<&[u8]> {
    let mut rest = data.strip_prefix(&[0x80, 4])?;
    if rest.first() == Some(&0x95) {
        rest = rest.get(9..)?;
    }
    let (length, body) = match rest.first()? {
        0x8e => (u64::from_le_bytes(rest.get(1..9)?.try_into().ok()?) as usize, rest.get(9..)?),
        b'B' => (u32::from_le_bytes(rest.get(1..5)?.try_into().ok()?) as usize, rest.get(5..)?),
        b'C' => (*rest.get(1)? as usize, rest.get(2..)?),
        _ => return None,
    };
    let payload = body.get(..length)?;
    (body.get(length) == Some(&b'.')).then_some(payload)
}

/// Calls `each` for every row of `stream` in timestamp order. `each` returns false to stop early.
pub fn for_each(db: &Connection, stream: &DbStream, mut each: impl FnMut(DbRow) -> Result<bool>) -> Result<()> {
    let table = quote(&stream.name);
    let blob = quote(&format!("{}_blob", stream.name));
    let mut statement = db.prepare(&format!(
        "SELECT m.ts, m.pose_x, m.pose_y, m.pose_z, m.pose_qx, m.pose_qy, m.pose_qz, m.pose_qw, b.data \
         FROM {table} AS m JOIN {blob} AS b ON b.id = m.id ORDER BY m.ts"
    ))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let ts: f64 = row.get(0)?;
        let pose: Vec<Option<f64>> = (1..8).map(|i| row.get(i)).collect::<rusqlite::Result<_>>()?;
        let pose = if pose.iter().all(Option::is_some) { Some(std::array::from_fn(|i| pose[i].unwrap())) } else { None };
        let data: Vec<u8> = row.get(8)?;
        let payload = decode_blob(&stream.codec, &data)?;
        if !each(DbRow { ts, pose, payload })? {
            break;
        }
    }
    Ok(())
}

/// What a new stream holds, for `_streams.config` (the module dimos imports to decode it).
pub fn payload_module(kind: &str) -> String {
    let (package, class) = kind.split_once('.').unwrap_or(("std_msgs", kind));
    format!("dimos.msgs.{package}.{class}.{class}")
}

/// Replaces stream `name` (dropping any previous one of that name) with `rows` (ts, LCM payload) of `kind`.
pub fn write_stream(db: &mut Connection, name: &str, kind: &str, rows: &[(f64, Vec<u8>)]) -> Result<()> {
    let tx = db.transaction()?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS _streams (name TEXT PRIMARY KEY, config TEXT NOT NULL)")?;
    for suffix in ["", "_blob", "_rtree"] {
        tx.execute_batch(&format!("DROP TABLE IF EXISTS {}", quote(&format!("{name}{suffix}"))))?;
    }
    let config = serde_json::json!({
        "payload_module": payload_module(kind),
        "codec_id": "lcm",
        "eager_blobs": false,
        "page_size": 256,
        "blob_store": { "class": "dimos.memory2.blobstore.sqlite.SqliteBlobStore", "config": { "path": null } },
        "vector_store": { "class": "dimos.memory2.vectorstore.sqlite.SqliteVectorStore", "config": { "path": null } },
        "notifier": { "class": "dimos.memory2.notifier.subject.SubjectNotifier", "config": {} },
    });
    tx.execute("INSERT OR REPLACE INTO _streams (name, config) VALUES (?1, ?2)", (name, config.to_string()))?;
    let table = quote(name);
    tx.execute_batch(&format!(
        "CREATE TABLE {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, value NUMERIC, \
         pose_x REAL, pose_y REAL, pose_z REAL, pose_qx REAL, pose_qy REAL, pose_qz REAL, pose_qw REAL, \
         tags BLOB DEFAULT (jsonb('{{}}')));
         CREATE TABLE {} (id INTEGER PRIMARY KEY, data BLOB NOT NULL);
         CREATE VIRTUAL TABLE {} USING rtree(id, x_min, x_max, y_min, y_max, z_min, z_max);",
        quote(&format!("{name}_blob")),
        quote(&format!("{name}_rtree")),
    ))?;
    {
        let mut meta = tx.prepare(&format!("INSERT INTO {table} (ts) VALUES (?1)"))?;
        let mut blob = tx.prepare(&format!("INSERT INTO {} (id, data) VALUES (?1, ?2)", quote(&format!("{name}_blob"))))?;
        for (ts, payload) in rows {
            meta.execute([ts])?;
            blob.execute((tx.last_insert_rowid(), payload))?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Creates stream `name` (memory2 layout) if it doesn't exist, holding `module` payloads encoded with `codec`.
pub fn ensure_stream(db: &Connection, name: &str, module: &str, codec: &str) -> Result<()> {
    db.execute_batch("CREATE TABLE IF NOT EXISTS _streams (name TEXT PRIMARY KEY, config TEXT NOT NULL)")?;
    let exists: Option<String> = db.query_row("SELECT config FROM _streams WHERE name = ?1", [name], |row| row.get(0)).optional()?;
    if exists.is_some() {
        return Ok(());
    }
    let config = serde_json::json!({
        "payload_module": module,
        "codec_id": codec,
        "eager_blobs": false,
        "page_size": 256,
        "blob_store": { "class": "dimos.memory2.blobstore.sqlite.SqliteBlobStore", "config": { "path": null } },
        "vector_store": { "class": "dimos.memory2.vectorstore.sqlite.SqliteVectorStore", "config": { "path": null } },
        "notifier": { "class": "dimos.memory2.notifier.subject.SubjectNotifier", "config": {} },
    });
    db.execute("INSERT INTO _streams (name, config) VALUES (?1, ?2)", (name, config.to_string()))?;
    let table = quote(name);
    db.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, value NUMERIC, \
         pose_x REAL, pose_y REAL, pose_z REAL, pose_qx REAL, pose_qy REAL, pose_qz REAL, pose_qw REAL, \
         tags BLOB DEFAULT (jsonb('{{}}')));
         CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, data BLOB NOT NULL);
         CREATE VIRTUAL TABLE IF NOT EXISTS {} USING rtree(id, x_min, x_max, y_min, y_max, z_min, z_max);",
        quote(&format!("{name}_blob")),
        quote(&format!("{name}_rtree")),
    ))?;
    Ok(())
}

/// Appends rows (ts, already-encoded blob) to an existing stream, in one transaction.
pub fn append_rows(db: &mut Connection, name: &str, rows: &[(f64, Vec<u8>)]) -> Result<()> {
    let tx = db.transaction()?;
    {
        let mut meta = tx.prepare(&format!("INSERT INTO {} (ts) VALUES (?1)", quote(name)))?;
        let mut blob = tx.prepare(&format!("INSERT INTO {} (id, data) VALUES (?1, ?2)", quote(&format!("{name}_blob"))))?;
        for (ts, payload) in rows {
            meta.execute([ts])?;
            blob.execute((tx.last_insert_rowid(), payload))?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Drops streams (their tables and `_streams` rows).
pub fn drop_streams(db: &mut Connection, names: &[String]) -> Result<()> {
    let tx = db.transaction()?;
    for name in names {
        for suffix in ["", "_blob", "_rtree"] {
            tx.execute_batch(&format!("DROP TABLE IF EXISTS {}", quote(&format!("{name}{suffix}"))))?;
        }
        tx.execute("DELETE FROM _streams WHERE name = ?1", [name])?;
    }
    tx.commit()?;
    Ok(())
}

/// (id, ts) of every row of a stream, in ts order: what to pick from before reading any payload.
pub fn timestamps(db: &Connection, name: &str) -> Result<Vec<(i64, f64)>> {
    let mut statement = db.prepare(&format!("SELECT id, ts FROM {} ORDER BY ts", quote(name)))?;
    let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// One row's decoded payload.
pub fn payload(db: &Connection, stream: &DbStream, id: i64) -> Result<Vec<u8>> {
    let data: Vec<u8> = db.query_row(&format!("SELECT data FROM {} WHERE id = ?1", quote(&format!("{}_blob", stream.name))), [id], |row| row.get(0))?;
    decode_blob(&stream.codec, &data)
}

/// The newest payload of `name`, if the stream exists.
pub fn latest(db: &Connection, name: &str) -> Result<Option<(f64, Vec<u8>)>> {
    let exists: Option<String> = db.query_row("SELECT config FROM _streams WHERE name = ?1", [name], |row| row.get(0)).optional()?;
    let Some(config) = exists else { return Ok(None) };
    let codec = serde_json::from_str::<serde_json::Value>(&config).ok().and_then(|c| c["codec_id"].as_str().map(String::from)).unwrap_or("lcm".into());
    let row: Option<(f64, Vec<u8>)> = db
        .query_row(
            &format!("SELECT m.ts, b.data FROM {} AS m JOIN {} AS b ON b.id = m.id ORDER BY m.ts DESC LIMIT 1", quote(name), quote(&format!("{name}_blob"))),
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    row.map(|(ts, data)| Ok((ts, decode_blob(&codec, &data)?))).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_streams_dimos_can_list_and_reads_them_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.db");
        let mut db = Connection::open(&path).unwrap();
        write_stream(&mut db, "map_builder_annotations", "std_msgs.String", &[(1.0, b"a".to_vec()), (2.0, b"b".to_vec())]).unwrap();
        // a rewrite replaces, never duplicates
        write_stream(&mut db, "map_builder_annotations", "std_msgs.String", &[(3.0, b"c".to_vec())]).unwrap();
        let listed = streams(&db).unwrap();
        assert_eq!(listed, vec![DbStream { name: "map_builder_annotations".into(), kind: "std_msgs.String".into(), codec: "lcm".into(), count: 1 }]);
        assert_eq!(latest(&db, "map_builder_annotations").unwrap(), Some((3.0, b"c".to_vec())));
        assert_eq!(latest(&db, "nope").unwrap(), None);
        let mut seen = Vec::new();
        for_each(&db, &listed[0], |row| {
            seen.push((row.ts, row.payload, row.pose));
            Ok(true)
        })
        .unwrap();
        assert_eq!(seen, vec![(3.0, b"c".to_vec(), None)]);
    }

    #[test]
    fn lz4_frames_decode() {
        let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
        std::io::Write::write_all(&mut encoder, b"hello lcm").unwrap();
        let framed = encoder.finish().unwrap();
        assert_eq!(decode_blob("lz4+lcm", &framed).unwrap(), b"hello lcm");
        assert!(decode_blob("pickle", b"").is_err());
    }

    #[test]
    fn pickled_bytes_round_trip_and_python_framing_reads() {
        assert_eq!(decode_blob("pickle", &pickle_bytes(b"abc")).unwrap(), b"abc");
        // pickle.dumps(b"abc", protocol=4)
        let python = [0x80, 4, 0x95, 8, 0, 0, 0, 0, 0, 0, 0, b'C', 3, b'a', b'b', b'c', 0x94, b'.'];
        // memoized (0x94 before STOP) isn't the bare shape: refused rather than misread
        assert_eq!(unpickle_bytes(&python), None);
        let python_unmemoized = [0x80, 4, 0x95, 7, 0, 0, 0, 0, 0, 0, 0, b'C', 3, b'a', b'b', b'c', b'.'];
        assert_eq!(unpickle_bytes(&python_unmemoized), Some(&b"abc"[..]));
    }

    #[test]
    fn appended_streams_list_count_and_drop() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Connection::open(dir.path().join("x.db")).unwrap();
        ensure_stream(&db, "memworld_frames", "builtins.bytes", "pickle").unwrap();
        ensure_stream(&db, "memworld_frames", "builtins.bytes", "pickle").unwrap();
        append_rows(&mut db, "memworld_frames", &[(1.0, pickle_bytes(b"one"))]).unwrap();
        append_rows(&mut db, "memworld_frames", &[(2.0, pickle_bytes(b"two"))]).unwrap();
        let stream = streams(&db).unwrap().into_iter().find(|s| s.name == "memworld_frames").unwrap();
        assert_eq!((stream.count, stream.codec.as_str()), (2, "pickle"));
        let ids = timestamps(&db, "memworld_frames").unwrap();
        assert_eq!(ids.iter().map(|r| r.1).collect::<Vec<_>>(), vec![1.0, 2.0]);
        assert_eq!(payload(&db, &stream, ids[1].0).unwrap(), b"two");
        drop_streams(&mut db, &["memworld_frames".into()]).unwrap();
        assert!(streams(&db).unwrap().is_empty());
    }
}
