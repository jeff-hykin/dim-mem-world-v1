# dim-mem-world-v1

Memory World as a [dimOS Desktop](https://github.com/jeff-hykin/dimos-desktop-mirror) app: pick a robot recording,
let it build the recording's visual memory, then ask where things are: "traffic cone" shows every cone it saw as a box
in 3D, with its size and a photo. No API key; everything runs on local models.

```sh
dimos-desktop install https://github.com/jeff-hykin/dim-mem-world-v1
```

## How it works

1. **Pick a recording** from Desktop's recordings folder (`.db` or `.mcap`). It needs color images, depth images, both
   cameras' intrinsics (CameraInfo) and a tf path from `map` / `world` / `odom` to the color camera; the check names
   whatever is missing.
2. **Models** are downloaded the first time (about 5.6 GB, each file's sha256 checked, partial downloads resumed): three
   SigLIP2 models from this repo's [`models-v1` release](https://github.com/jeff-hykin/dim-mem-world-v1/releases/tag/models-v1)
   and OWLv2 from onnx-community on Hugging Face. They run on ONNX Runtime from nixpkgs.
3. **Generate** writes the memory **into the recording itself** (new tables in a `.db`, new channels appended in place
   to an `.mcap`; nothing beside it): one keyframe per second, each with its camera pose, its depth drawn into the color
   camera, a JPEG, and **three per-patch embedding grids**: SigLIP2 base-patch16-224, base-patch16-NaFlex and
   so400m-patch16-NaFlex (1024-patch budget), every vision token pooled alone by the model's attention head so each patch
   lands in the text-aligned space. A stopped generation resumes where it left off.
4. **Ask**: a question's heat per patch (its score minus the best of a few background prompts) must be positive in
   **all three** models at once; the keyframes with the most agreed heat go to OWLv2 for boxes; each box's depth is
   lifted into the world with the keyframe's pose; sightings within 0.75 m merge into one place. "How many" is the number
   of places; "how tall" is each place's height.

Every action is an HTTP endpoint ([docs/api.md](docs/api.md)), listed in `dimos.yaml` for Desktop's agent, so the agent
chat can answer "how many traffic cones were seen, and how tall are they?" by calling `api/count` and `api/dimensions`.

## Layout

- `server/`: the `dimos-app-server` binary (axum): endpoints, model downloads, background jobs. Also a CLI:
  `dimos-app-server --check <rec>`, `--generate <rec>`, `--ask <rec> "traffic cone"`.
- `crates/memworld/`: the core: models (`siglip`, `owl`), `compat`, `generate`, `store` (in-place writes), `memory`,
  `query`, `camera` (depth registration, placing a box in 3D).
- `crates/dimos-recording/`: reading and appending to dimos recordings without dimos (vendored from the Map Editor, plus
  image and CameraInfo decoding).
- `frontend/`: the page (React + Vite + dimos-react).
- `tools/export_models.py`: how the `models-v1` release was made (maintainers only; the app never runs Python).

## Development

```sh
nix develop                   # Rust, Node, and ORT_DYLIB_PATH for ONNX Runtime
cargo test                    # model tests also need MEMWORLD_MODELS=<a downloaded models folder>
cargo run -p dimos-app-server -- --port 7790   # then open http://127.0.0.1:7790/
nix build .#dimosApp          # what Desktop builds
```

After changing an endpoint, regenerate the manifest (a test fails until you do):
`cargo run -p dimos-app-server -- --agent-json | deno run -A scripts/dimos_yaml.ts`.
