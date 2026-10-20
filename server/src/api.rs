//! Every Memory World action as an HTTP endpoint (routes.rs): the page calls these, Desktop's agent calls the same ones
//! (served as openapi.json, repeated in dimos.yaml). Errors are `{ "error": "..." }`. docs/api.md has the types.
use crate::app::App;
use crate::render;
use crate::routes::Routes;
use axum::body::Bytes;
use axum::extract::{FromRequest, Path, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use memworld::generate;
use memworld::query::{self, Answer};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub const TITLE: &str = "Memory World";
/// the endpoints' API version (openapi.json's info.version): major = breaking, minor = added
pub const API_VERSION: &str = "1.0.0";
pub const WHEN: &str = "you need to find, count or measure things a robot saw in a recording";
pub const DESCRIPTION: &str = "Memory World: a robot recording's visual memory (three SigLIP2 models per image patch, stored INSIDE the recording) \
and questions about it answered as places in 3D, with photo evidence. No API key: everything runs on local models. Coordinates are meters in the \
recording's world frame (worldFrame: odom/map/world), +z up. Workflow: GET api/status (open recording, its memory, models, a running generation). \
If no recording is open, GET api/recordings then POST api/open {path}. If memory.state isn't \"ready\", POST api/generate (downloads the models \
first if needed; poll GET api/generate). Then ask: GET api/count?text=traffic cone (how many distinct places), GET api/dimensions?text=... (each \
place's size and height), or POST api/query {text} for everything (places, every sighting with its box, timings). Show the user the evidence: \
GET api/places/{id}/evidence?text=... is a boxed photo. Use short noun phrases as text (\"traffic cone\", \"fire extinguisher\"); a place is one \
physical thing seen from one or more keyframes.";

pub struct ApiError(StatusCode, String);

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(error: E) -> Self {
        let message = format!("{:#}", error.into());
        let status = if message.starts_with("no such") || message.starts_with("no recording is open") || message.contains("has no memory yet") || message.starts_with("no keyframe") || message.starts_with("no place") {
            StatusCode::NOT_FOUND
        } else if message.contains("aren't downloaded") || message.contains("already running") || message.contains("is running") {
            StatusCode::CONFLICT
        } else {
            StatusCode::BAD_REQUEST
        };
        ApiError(status, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

type Result<T> = std::result::Result<T, ApiError>;
type Args = Query<HashMap<String, String>>;

/// A JSON body, lenient: no body (or no content-type, as an agent may send) is `{}`.
pub struct Body(pub Value);

impl<S: Send + Sync> FromRequest<S> for Body {
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> std::result::Result<Self, ApiError> {
        let bytes = Bytes::from_request(request, state).await.map_err(|error| ApiError(StatusCode::BAD_REQUEST, error.to_string()))?;
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return Ok(Body(json!({})));
        }
        serde_json::from_slice(&bytes).map(Body).map_err(|error| ApiError(StatusCode::BAD_REQUEST, format!("the body isn't JSON: {error}")))
    }
}

fn s(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}
fn n(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}
fn b(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}
fn int(description: &str) -> Value {
    json!({ "type": "integer", "description": description })
}
fn required(mut spec: Value) -> Value {
    spec["required"] = json!(true);
    spec
}

pub fn routes() -> Routes<Arc<App>> {
    let text = || json!({ "text": required(s("what to look for, a short noun phrase (e.g. \"traffic cone\")")) });
    Routes::new()
        .endpoint("GET", "api/status", "Everything at a glance: the open recording, its compatibility check (what's missing, if anything), its memory (state none / partial / ready, keyframes), the models (downloaded? progress), the device they run on (cpu / coreml / cuda), the running or last generation, and whether the memory is loaded for questions.", json!({}), status)
        .role("context")
        .when("you need the current recording, memory, model or generation state")
        .endpoint("GET", "api/recordings", "The recordings in Desktop's recordings folder, newest first. Returns { dir, recordings: [{ id (file name, for api/recordings/{id}/check and api/open), name, path, format (db / mcap), bytes, modified (unix seconds), memory (state none / partial / ready, or null if unreadable) }] }.", json!({}), recordings)
        .when("you need to find which recordings exist and which have memory")
        .endpoint("GET", "api/recordings/{id}/check", "Whether a recording (id = its file name in the recordings folder) can be used: color images, the camera's intrinsics, a lidar, and poses (tf to the camera and the lidar, or a raw lidar + IMU that lite_record's post_process gives poses to first); `missing` names each thing it lacks. Also its memory state. 404 if no such recording.", json!({ "id": required(s("the recording's file name in the recordings folder (an id from GET api/recordings)")) }), check_one)
        .when("you need to know whether a recording can be given a memory")
        .endpoint("POST", "api/open", "Open a recording (every open page switches to it): path = absolute path (Desktop's GET /recordings lists them) or a file name in the recordings folder. Checks it and loads its memory (in the background) if it has one. Returns what GET api/status returns; 404 if the file doesn't exist, 409 while a generation is running.", json!({ "path": required(s("the recording file (.db / .mcap): an absolute path, or a file name in the recordings folder")) }), open)
        .when("you need to open a recording to work on")
        .endpoint("GET", "api/models", "The model files (downloaded at run time, about 5.7 GB in all, 7.1 GB on a Mac with its CoreML copies): per file bytes done and state, overall progress, speed and ETA.", json!({}), models)
        .when("you need to know whether the models are downloaded")
        .endpoint("POST", "api/models/download", "Download the models that are missing (resumes partial files, checks each sha256). Runs in the background; returns at once with what GET api/models returns, so poll that for progress.", json!({}), download)
        .when("you need to download the models before generating")
        .endpoint("POST", "api/generate", "Build the open recording's memory IN PLACE (new streams inside the recording): the lidar's voxel map, and one keyframe per everySeconds, each embedded by three SigLIP2 models per patch plus OWLv2's image half, with its pose and a JPEG. Runs lite_record's post_process first on a raw lidar + IMU recording; downloads missing models first. Runs in the background (poll GET api/generate); a stopped generation resumes where it left off. Returns what GET api/status returns; 404 with no open recording, 409 if one is already running.", json!({ "everySeconds": n("one keyframe per this many seconds, 0.05 to 60 (default 1.0)") }), start_generate)
        .when("the open recording has no ready memory yet")
        .endpoint("GET", "api/generate", "The running (or last) generation's progress: phase (models, post_process, planning, map, embedding, done, cancelled, failed), keyframes done / total, frames per second, ETA, bytes written into the recording. { phase: \"idle\" } when nothing has run since the app started.", json!({}), generate_progress)
        .when("you need to check how a memory generation is going")
        .endpoint("POST", "api/generate/cancel", "Stop the running generation after the keyframe in hand (what's done stays; POST api/generate resumes). Returns the generation's progress as GET api/generate does, or { phase: \"idle\" } if nothing was running.", json!({}), cancel)
        .when("you need to stop a running memory generation")
        .endpoint("DELETE", "api/memory", "Forget the open recording's memory (drops its tables from a .db; marks an .mcap's memory cleared). The recording's own data is untouched. Returns what GET api/status returns; 404 with no open recording, 409 while a generation is running (cancel it first).", json!({}), forget)
        .when("you need to delete the open recording's memory")
        .endpoint("POST", "api/query", "Ask where something is: the keyframes where all three models agree go to OWLv2 for boxes, each box is placed in 3D from the lidar map drawn into that keyframe's camera, and sightings of one thing merge into a place (along their lines of sight). Returns count, places (center, size, heightM, sightings, best sighting), every sighting with its box and 3D extent, the candidate frames, timings and a one-line summary. The page shows the answer.", json!({ "text": required(s("what to look for, a short noun phrase")), "frames": int("at most this many keyframes (most agreed heat first) are scored by OWLv2 (default 200)"), "owlThreshold": n("OWLv2 box score cutoff (default 0.2)"), "placeRadiusM": n("sightings closer than this are one place (default 0.75)"), "agreementOnly": b("skip OWLv2; boxes are where the three models agree (default false)") }), ask)
        .when("you need where something is, with every sighting and its boxes")
        .endpoint("GET", "api/count", "How many distinct things matching text were seen. Returns { query, count, places: [{ id, center, heightM, sightings (keyframes that saw it) }], worldFrame, summary }. Reuses the last answer when text matches it, else asks POST api/query with its defaults.", text(), count)
        .when("you need to know how many of something the robot saw")
        .endpoint("GET", "api/dimensions", "How big each thing matching text is. Returns { query, places: [{ id, center, size (x, y, z extents in meters), heightM (top minus bottom), bottomZ, topZ, sightings }], worldFrame, summary }. Reuses the last answer when text matches it, else asks POST api/query with its defaults.", text(), dimensions)
        .when("you need the size or height of things the robot saw")
        .endpoint("GET", "api/answer", "The last answer (what the page is showing), in POST api/query's shape (query, count, places, every sighting, summary), or null if nothing has been asked since the recording was opened.", json!({}), last_answer)
        .role("view")
        .when("you need the answer the page is currently showing")
        .endpoint("GET", "api/places/{id}/evidence", "A photo of a place: the keyframe of its best sighting as JPEG, the thing boxed. text = the question the place came from (asked again if it isn't the last answer). 404 if the answer has no such place.", json!({ "id": required(int("place id, from an answer's places (POST api/query, GET api/count)")), "text": required(s("the question the place came from, e.g. \"traffic cone\"")) }), evidence)
        .when("you need a photo proving where a found thing is")
        .endpoint("GET", "api/frames", "The memory's keyframes, the camera's path. Returns { worldFrame, frames: [{ index, ts (seconds), position [x, y, z], rotation (quaternion x, y, z, w) }] }; 404 if the open recording has no memory yet.", json!({}), frames)
        .when("you need the camera's path through the recording")
        .endpoint("GET", "api/frames/{index}/image", "A keyframe's stored image as a JPEG, with box=x0,y0,x1,y1 (pixels) outlined when given (e.g. a sighting's box from POST api/query). 404 if there's no such keyframe.", json!({ "index": required(int("keyframe index, from GET api/frames or a sighting's frame")), "box": s("optional box to outline: x0,y0,x1,y1 in pixels") }), frame_image)
        .when("you need to see what the camera saw at one keyframe")
        .plumbing("GET", "api/cloud", cloud)
        .plumbing("GET", "openapi.json", || async { Json(openapi()) })
}

/// The endpoints as an OpenAPI 3.1 document: the served openapi.json, and dimos.yaml's `provides:`.
pub fn openapi() -> Value {
    routes().openapi(TITLE, API_VERSION, WHEN, DESCRIPTION)
}

async fn status(State(app): State<Arc<App>>) -> Json<Value> {
    Json(app.status_json())
}

async fn recordings(State(app): State<Arc<App>>) -> Result<Json<Value>> {
    let dir = app.recordings_dir.clone();
    let list = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Value>> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(&dir) else { return Ok(out) };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(format) = memworld::recording::format_of(&path) else { continue };
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            let memory = memworld::recording::Recording::open(&path).and_then(|r| memworld::memory::status(&r)).ok();
            out.push(json!({
                "id": entry.file_name().to_string_lossy(),
                "name": path.file_stem().map(|s| s.to_string_lossy().to_string()),
                "path": path.display().to_string(),
                "format": if format == memworld::recording::Format::Db { "db" } else { "mcap" },
                "bytes": meta.len(),
                "modified": meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()),
                "memory": memory,
            }));
        }
        out.sort_by(|a, b| b["modified"].as_u64().cmp(&a["modified"].as_u64()));
        Ok(out)
    })
    .await??;
    Ok(Json(json!({ "dir": app.recordings_dir, "recordings": list })))
}

async fn check_one(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Value>> {
    let path = app.resolve(&id)?;
    let value = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let recording = memworld::recording::Recording::open(&path)?;
        let (check, _) = memworld::compat::check(&recording)?;
        let mut value = serde_json::to_value(check)?;
        value["memory"] = serde_json::to_value(memworld::memory::status(&recording)?)?;
        Ok(value)
    })
    .await??;
    Ok(Json(value))
}

async fn open(State(app): State<Arc<App>>, Body(body): Body) -> Result<Json<Value>> {
    let path = body["path"].as_str().or(body["id"].as_str()).ok_or_else(|| anyhow::anyhow!("path is required"))?;
    let path = app.resolve(path)?;
    app.open(path).await?;
    Ok(Json(app.status_json()))
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!(app.downloads.snapshot()))
}

async fn download(State(app): State<Arc<App>>) -> Json<Value> {
    drop(app.downloads.start());
    Json(json!(app.downloads.snapshot()))
}

async fn start_generate(State(app): State<Arc<App>>, Body(body): Body) -> Result<Json<Value>> {
    let mut options = generate::Options::default();
    if let Some(every) = body["everySeconds"].as_f64() {
        if !(0.05..=60.0).contains(&every) {
            return Err(anyhow::anyhow!("everySeconds must be between 0.05 and 60").into());
        }
        options.every_seconds = every;
    }
    app.start_generation(options)?;
    Ok(Json(app.status_json()))
}

async fn generate_progress(State(app): State<Arc<App>>) -> Json<Value> {
    Json(app.generation.lock().unwrap().as_ref().map(|p| json!(p)).unwrap_or_else(|| json!({ "phase": "idle", "done": 0, "total": 0, "message": "nothing has run" })))
}

async fn cancel(State(app): State<Arc<App>>) -> Json<Value> {
    app.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
    Json(app.generation.lock().unwrap().as_ref().map(|p| json!(p)).unwrap_or(json!({ "phase": "idle" })))
}

async fn forget(State(app): State<Arc<App>>) -> Result<Json<Value>> {
    if app.generating.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(anyhow::anyhow!("a generation is running: cancel it first").into());
    }
    let path = app.open_path()?;
    *app.loaded.write().unwrap() = None;
    *app.last_answer.write().unwrap() = None;
    let target = path.clone();
    tokio::task::spawn_blocking(move || memworld::generate::clear(&target)).await??;
    app.refresh_status(&path)?;
    Ok(Json(app.status_json()))
}

fn query_options(body: &Value) -> query::Options {
    let mut options = query::Options::default();
    if let Some(v) = body["frames"].as_u64() {
        options.frames = v.clamp(1, 200) as usize;
    }
    if let Some(v) = body["owlThreshold"].as_f64() {
        options.owl_threshold = v as f32;
    }
    if let Some(v) = body["placeRadiusM"].as_f64() {
        options.place_radius_m = v;
    }
    if let Some(v) = body["agreementOnly"].as_bool() {
        options.agreement_only = v;
    }
    options
}

async fn ask(State(app): State<Arc<App>>, Body(body): Body) -> Result<Json<Answer>> {
    let text = body["text"].as_str().ok_or_else(|| anyhow::anyhow!("text is required"))?;
    Ok(Json(app.ask(text, query_options(&body)).await?))
}

/// The last answer when it's for `text`, else a fresh one.
async fn answer_for(app: &Arc<App>, text: &str) -> anyhow::Result<Answer> {
    if let Some(answer) = app.last_answer.read().unwrap().as_ref().filter(|a| a.query.eq_ignore_ascii_case(text.trim())) {
        return Ok(answer.clone());
    }
    app.ask(text, query::Options::default()).await
}

fn text_arg(args: &HashMap<String, String>) -> anyhow::Result<String> {
    args.get("text").filter(|t| !t.trim().is_empty()).cloned().ok_or_else(|| anyhow::anyhow!("text is required (?text=traffic cone)"))
}

async fn count(State(app): State<Arc<App>>, Query(args): Args) -> Result<Json<Value>> {
    let answer = answer_for(&app, &text_arg(&args)?).await?;
    let places: Vec<Value> = answer.places.iter().map(|p| json!({ "id": p.id, "center": p.center, "heightM": p.height_m, "sightings": p.observations.len() })).collect();
    Ok(Json(json!({ "query": answer.query, "count": answer.count, "places": places, "worldFrame": answer.world_frame, "summary": answer.summary })))
}

async fn dimensions(State(app): State<Arc<App>>, Query(args): Args) -> Result<Json<Value>> {
    let answer = answer_for(&app, &text_arg(&args)?).await?;
    let places: Vec<Value> = answer.places.iter().map(|p| json!({ "id": p.id, "center": p.center, "size": p.size, "heightM": p.height_m, "bottomZ": p.bottom_z, "topZ": p.top_z, "sightings": p.observations.len() })).collect();
    Ok(Json(json!({ "query": answer.query, "places": places, "worldFrame": answer.world_frame, "summary": answer.summary })))
}

async fn last_answer(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!(*app.last_answer.read().unwrap()))
}

fn jpeg(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")], bytes).into_response()
}

async fn evidence(State(app): State<Arc<App>>, Path(id): Path<usize>, Query(args): Args) -> Result<Response> {
    let answer = answer_for(&app, &text_arg(&args)?).await?;
    let place = answer.places.iter().find(|p| p.id == id).ok_or_else(|| anyhow::anyhow!("no place {id} for {:?} ({} places)", answer.query, answer.places.len()))?;
    let sighting = &answer.observations[place.best];
    let loaded = app.memory().await?;
    let frame = loaded.memory.frames.get(sighting.frame).ok_or_else(|| anyhow::anyhow!("no keyframe {}", sighting.frame))?;
    Ok(jpeg(render::boxed_jpeg(&frame.jpeg, Some(sighting.bbox))?))
}

async fn frames(State(app): State<Arc<App>>) -> Result<Json<Value>> {
    let loaded = app.memory().await?;
    let list: Vec<Value> = loaded
        .memory
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let (t, r) = memworld::tf::parts(&f.pose);
            json!({ "index": i, "ts": f.ts, "position": t, "rotation": r })
        })
        .collect();
    Ok(Json(json!({ "worldFrame": loaded.memory.meta["worldFrame"], "frames": list })))
}

async fn frame_image(State(app): State<Arc<App>>, Path(index): Path<usize>, Query(args): Args) -> Result<Response> {
    let loaded = app.memory().await?;
    let frame = loaded.memory.frames.get(index).ok_or_else(|| anyhow::anyhow!("no keyframe {index} ({} keyframes)", loaded.memory.frames.len()))?;
    let bbox = match args.get("box") {
        Some(text) => {
            let values: Vec<f32> = text.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            if values.len() != 4 {
                return Err(anyhow::anyhow!("box is x0,y0,x1,y1").into());
            }
            Some([values[0], values[1], values[2], values[3]])
        }
        None => None,
    };
    Ok(jpeg(render::boxed_jpeg(&frame.jpeg, bbox)?))
}

async fn cloud(State(app): State<Arc<App>>) -> Result<Response> {
    let loaded = app.memory().await?;
    let cached = loaded.cloud.lock().unwrap().clone();
    let bytes = match cached {
        Some(bytes) => bytes,
        None => {
            let source = loaded.clone();
            let bytes = Arc::new(tokio::task::spawn_blocking(move || render::cloud(&source.memory)).await??);
            *loaded.cloud.lock().unwrap() = Some(bytes.clone());
            bytes
        }
    };
    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], bytes.as_ref().clone()).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body as HttpBody;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn dimos_yaml_provides_exactly_the_served_openapi() {
        let yaml: serde_yaml::Value = serde_yaml::from_str(include_str!("../../dimos.yaml")).unwrap();
        let mut listed: Value = serde_json::to_value(&yaml["provides"]).unwrap();
        listed.as_object_mut().unwrap().remove("x-private");
        assert_eq!(listed, openapi(), "dimos.yaml is stale: cargo run -p dimos-app-server -- --openapi-json | deno run -A scripts/dimos_yaml.ts");
        let table = routes().endpoints;
        assert!(table.len() >= 8);
        assert!(table.iter().all(|e| e["description"].as_str().is_some_and(|d| d.len() > 30)));
        assert!(table.iter().all(|e| e["when"].as_str().is_some_and(|w| !w.is_empty())));
        assert_eq!(yaml["provides"]["info"]["x-when"].as_str(), Some(WHEN));
    }

    async fn call(router: &axum::Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let request = axum::http::Request::builder().method(method).uri(uri).header("content-type", "application/json").body(HttpBody::from(body.map(|b| b.to_string()).unwrap_or_default())).unwrap();
        let response = router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn questions_without_a_recording_say_what_to_do() {
        let dir = tempfile::tempdir().unwrap();
        let app = App::new(dir.path().join("data"), dir.path().join("recordings"));
        let router = routes().router.with_state(app);
        let (status, body) = call(&router, "GET", "/api/status", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["recording"], Value::Null);
        // every model file is listed with its size (whether they're downloaded depends on MEMWORLD_MODELS)
        assert_eq!(body["models"]["files"].as_array().unwrap().len(), memworld::models::files().len());
        assert_eq!(body["models"]["totalBytes"], memworld::models::total_bytes());
        let (status, body) = call(&router, "POST", "/api/query", Some(json!({ "text": "a cone" }))).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert!(body["error"].as_str().unwrap().contains("POST api/open"));
        let (status, body) = call(&router, "GET", "/api/count", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains("text is required"));
        let (status, _) = call(&router, "POST", "/api/open", Some(json!({ "path": "/nope/none.db" }))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, body) = call(&router, "GET", "/openapi.json", None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["info"]["description"].as_str().unwrap().contains("api/count"));
    }
}
