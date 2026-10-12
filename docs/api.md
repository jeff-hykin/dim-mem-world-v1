# Memory World API

Everything the page does is one of these endpoints, and Desktop's agent (dimcode) calls the same ones (they're listed in
`dimos.yaml` `provides.endpoints` and served as `agent.json`). Paths are relative to the app (`/apps/<name>/`). JSON in
and out; errors are `{ "error": "..." }` with a 4xx/5xx status. No API key anywhere: questions run on local models.

Coordinates are meters in the recording's world frame (`worldFrame`: `odom`, `map` or `world`), +z up.

## Recordings

| Endpoint                        | Body / query          | Returns                                                                      |
| ------------------------------- | --------------------- | ---------------------------------------------------------------------------- |
| `GET api/recordings`            |                       | `{ dir, recordings: [{ id, name, path, format, bytes, modified, memory }] }` |
| `GET api/recordings/{id}/check` |                       | `Check` + `memory` (`Status`) for that recording                             |
| `POST api/open`                 | `{ path }` or `{ id }` | `AppStatus` — makes it the open recording (loads its memory if it has one)   |
| `GET api/status`                |                       | `AppStatus`                                                                  |

`id` is the file name inside Desktop's recordings folder. The page lists recordings from Desktop's own
`GET ../../recordings` and opens one by `path`.

```ts
type Check = {
    path: string
    format: "db" | "mcap"
    compatible: boolean
    missing: string[] // one sentence per missing input: color images, depth images, intrinsics, a camera pose
    color?: ImageStream
    depth?: ImageStream
    colorInfo?: string
    depthInfo?: string
    worldFrame?: string
    start: number
    end: number
    durationS: number
    notes: string[]
}
type ImageStream = { name: string; encoding: string; width: number; height: number; frameId: string; count: number }
type Status = { state: "none" | "partial" | "ready"; generation?: number; frames: number; planned: number; options: object; createdMs?: number }
type AppStatus = {
    recording: { path: string; name: string } | null
    check: Check | null
    memory: Status | null
    models: Models
    generation: Progress | null // the running or last generation
    loaded: { frames: number; patches: number[]; loadMs: number } | null // memory held in RAM for questions
    loading: boolean
}
```

## Models (downloaded at run time)

| Endpoint                   | Returns                                         |
| -------------------------- | ----------------------------------------------- |
| `GET api/models`           | `Models`                                        |
| `POST api/models/download` | `Models` — starts downloading what's missing     |

```ts
type Models = {
    dir: string
    ready: boolean
    totalBytes: number
    doneBytes: number // downloaded so far, all files
    bytesPerSecond: number | null
    etaS: number | null
    downloading: boolean
    error: string | null
    files: { name: string; model: string; bytes: number; doneBytes: number; state: "missing" | "downloading" | "verifying" | "ready" | "failed"; error?: string }[]
}
```

## Generating the memory (in place)

| Endpoint                    | Body                       | Returns                                                                               |
| --------------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| `POST api/generate`         | `{ everySeconds?: number }` | `AppStatus` — downloads missing models first, then embeds the open recording        |
| `GET api/generate`          |                            | `Progress` (`phase: "idle"` when nothing ran)                                         |
| `POST api/generate/cancel`  |                            | `Progress` — stops after the keyframe in hand; Generate again resumes                 |
| `DELETE api/memory`         |                            | `AppStatus` — forgets the open recording's memory (drops its tables / clears its mcap) |

```ts
type Progress = {
    phase: "idle" | "models" | "post_process" | "planning" | "map" | "embedding" | "done" | "cancelled" | "failed"
    done: number
    total: number
    skipped: number
    elapsedS: number
    etaS: number | null
    framesPerS: number | null
    writtenBytes: number
    generation: number
    message: string
}
```

## Questions

| Endpoint                                  | Body / query                                                          | Returns                                         |
| ----------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------- |
| `POST api/query`                          | `{ text, frames?, owlThreshold?, placeRadiusM?, agreementOnly? }`      | `Answer`                                        |
| `GET api/count?text=`                     |                                                                       | `{ query, count, places: [{ id, center, heightM, sightings }], summary }` |
| `GET api/dimensions?text=`                |                                                                       | `{ query, places: [{ id, center, size, heightM, bottomZ, topZ }], summary }` |
| `GET api/answer`                          |                                                                       | the last `Answer` (what the page shows), or `null` |
| `GET api/places/{id}/evidence?text=`      |                                                                       | `image/jpeg`: the place's best sighting, boxed   |
| `GET api/frames`                          |                                                                       | `{ worldFrame, frames: [{ index, ts, position, rotation }] }` |
| `GET api/frames/{index}/image?box=x0,y0,x1,y1` |                                                                  | `image/jpeg`: a stored keyframe, optionally boxed |
| `GET api/cloud`                           |                                                                       | binary point cloud of the memory (below)        |

```ts
type Answer = {
    query: string
    count: number // places
    places: Place[]
    observations: Observation[]
    candidates: { frame: number; ts: number; heat: number; grid: [number, number]; cells: [number, number][] }[]
    unplaced: number
    worldFrame: string
    timings: { textMs: number; searchMs: number; owlMs: number; placeMs: number; totalMs: number }
    summary: string
}
type Place = {
    id: number
    center: [number, number, number]
    size: [number, number, number] // x, y, z extents
    heightM: number
    bottomZ: number
    topZ: number
    observations: number[] // indexes into observations
    best: number
    score: number
    firstSeen: number
    lastSeen: number
}
type Observation = {
    frame: number // keyframe index (api/frames)
    ts: number
    bbox: [number, number, number, number] // pixels of the stored keyframe image
    source: "owl" | "agreement"
    score: number
    agreement: number
    placed: { center: number[]; size: number[]; min: number[]; max: number[]; support: number; range: number } | null
    place: number | null
}
```

`api/cloud` is little-endian: `u32 count`, then `count` × `f32 x, y, z`, then `count` × `u8 r, g, b`. It is the memory's
keyframes' depth, colored by their images, thinned to a 5 cm grid: the scene to draw answers in.
