# dim-mem-world-v1

Memory World as a [dimOS Desktop](https://github.com/jeff-hykin/dimos-desktop-mirror) app: pick a robot recording,
let it build the recording's visual memory, then ask where things are: "traffic cone" shows every cone it saw as a box
in 3D, with its size and a photo. No API key; everything runs on local models.

```sh
dimos-desktop install https://github.com/jeff-hykin/dim-mem-world-v1
```

## How it works

1. **Pick a recording** from Desktop's recordings folder (`.db` or `.mcap`). It needs color images, the camera's
   intrinsics (CameraInfo), a lidar, and poses: tf from `map` / `world` / `odom` to the camera and the lidar. A raw
   lidar + IMU recording (lite_record's) gets its poses from lite_record's `post_process` first, run in place (lite_record
   is part of the nix build). The check names whatever is missing.
2. **Models** are downloaded the first time (about 5.7 GB, 7.1 GB on a Mac; each file's sha256 checked, partial
   downloads resumed): three SigLIP2 models and OWLv2 (split into its image and text halves), from this repo's
   `models-v1`/`v2`/`v3` releases. **Where they run:** on a Mac the image models are native CoreML (fp16, the GPU;
   `tools/export_coreml.py`, compiled once on the machine, then loaded in about a second); on x86_64 Linux with an
   Nvidia GPU, ONNX Runtime's CUDA build; anywhere else, and whenever the GPU can't load or run a model (out of memory
   included), ONNX Runtime on the CPU. `MEMWORLD_DEVICE=cpu|coreml|cuda` overrides the choice; `GET api/status` says
   which. On an M-series Mac a keyframe takes ~0.2 s (CPU: ~18 s).
3. **Generate** writes the memory **into the recording itself** (new tables in a `.db`, new channels appended in place
   to an `.mcap`; nothing beside it): the lidar's 5 cm voxel map, and one keyframe per second, each with its camera pose,
   a JPEG, and **three per-patch embedding grids**: SigLIP2 base-patch16-224, base-patch16-NaFlex and
   so400m-patch16-NaFlex (1024-patch budget), every vision token pooled alone by the model's attention head so each patch
   lands in the text-aligned space; plus OWLv2's image half (a box and class embedding per patch). A stopped generation
   resumes where it left off.
4. **Ask**: a question's heat per patch (its score minus the best of a few background prompts) must be positive in
   **all three** models at once; the keyframes with agreed heat are scored by OWLv2 (its stored image half against the
   question's text embedding: a dot product per patch, milliseconds per frame) and its boxes must cover agreed cells; each box is placed with the lidar map drawn into that keyframe's camera (nearest surface wins), and is
   lifted into the world with the keyframe's pose; sightings within 0.75 m merge into one place. "How many" is the number
   of places; "how tall" is each place's height.

Every action is an HTTP endpoint ([docs/api.md](docs/api.md)), listed in `dimos.yaml` for Desktop's agent, so the agent
chat can answer "how many traffic cones were seen, and how tall are they?" by calling `api/count` and `api/dimensions`.

## Layout

- `server/`: the `dimos-app-server` binary (axum): endpoints, model downloads, background jobs. Also a CLI:
  `dimos-app-server --check <rec>`, `--generate <rec>`, `--ask <rec> "traffic cone"`.
- `crates/memworld/`: the core: models (`siglip`, `owl`), `compat`, `generate`, `store` (in-place writes), `memory`,
  `query`, `camera` (drawing the lidar map into a camera, placing a box in 3D), `lidar` (the voxel map), `lite_record`.
- `crates/dimos-recording/`: reading and appending to dimos recordings without dimos (vendored from the Map Editor, plus
  image and CameraInfo decoding).
- `frontend/`: the page (React + Vite + dimos-react).
- `tools/export_models.py`: how the `models-v1` release was made (maintainers only; the app never runs Python).

## Development

```sh
nix develop                   # Rust, Deno, and ORT_DYLIB_PATH for ONNX Runtime
cargo test                    # model tests also need MEMWORLD_MODELS=<a downloaded models folder>
cargo run -p dimos-app-server -- --port 7790   # then open http://127.0.0.1:7790/
(cd frontend && deno task dev)  # the page, against a running Desktop (DESKTOP_URL)
(cd frontend && deno task typecheck)
nix build .#dimosApp          # what Desktop builds
deno task update-hashes       # after changing frontend/deno.json or deno.lock: flake.nix's hashes for the page
```

After changing an endpoint, regenerate the manifest (a test fails until you do):
`cargo run -p dimos-app-server -- --openapi-json | deno run -A scripts/dimos_yaml.ts`.
