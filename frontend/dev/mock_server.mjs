// Dev-only mock of Desktop (/recordings) and this app's API (/apps/dim-mem-world/api/*), per docs/api.md, so the page
// can be seen without the Rust backend. Not shipped.
//
//     deno run -A dev/mock_server.mjs     # :7099
//     DESKTOP_URL=http://127.0.0.1:7099 deno task dev
//
// MOCK_STATE=pick|incompatible|models|generating|ready picks where it starts (default ready). Generating and model
// downloads advance in real time.
import http from "node:http"

const PORT = Number(process.env.PORT ?? 7099)
const START = process.env.MOCK_STATE ?? "ready"
const APP = "/apps/dim-mem-world/"

const recordings = [
    { id: "go2_office_walk.db", name: "go2_office_walk.db", format: "db", size: 2.31e9, modified: Date.now() / 1000 - 3600 * 5, path: "/rec/go2_office_walk.db", writable: true },
    { id: "parking_lot_cones.mcap", name: "parking_lot_cones.mcap", format: "mcap", size: 812e6, modified: Date.now() / 1000 - 600, path: "/rec/parking_lot_cones.mcap", writable: true },
    { id: "lidar_only.mcap", name: "lidar_only.mcap", format: "mcap", size: 140e6, modified: Date.now() / 1000 - 86400 * 9, path: "/rec/lidar_only.mcap", writable: true },
]

const MODEL_FILES = [
    ["siglip2-so400m-patch16-512.vision.onnx", "siglip2-so400m", 1.72e9],
    ["siglip2-so400m-patch16-512.text.onnx", "siglip2-so400m", 1.13e9],
    ["siglip2-base-patch16-naflex.onnx", "siglip2-base", 0.81e9],
    ["siglip2-large-patch16-384.onnx", "siglip2-large", 1.24e9],
]
const models = {
    dir: "~/.cache/dimos/memworld/models",
    files: MODEL_FILES.map(([name, model, bytes]) => ({ name, model, bytes, doneBytes: START === "models" || START === "pick" || START === "incompatible" ? 0 : bytes, state: "ready" })),
    downloading: false,
    error: null,
}
for (const file of models.files) {
    file.state = file.doneBytes >= file.bytes ? "ready" : "missing"
}
let downloadStarted = 0

const check = (path) =>
    path.includes("lidar_only")
        ? { path, format: "mcap", compatible: false, missing: ["No color images: the recording has no sensor_msgs/Image or CompressedImage stream.", "No depth images: nothing publishes a 16UC1 or 32FC1 depth image.", "No camera pose: no tf chain from a world frame (odom, map or world) to the camera's frame."], start: 1760000000, end: 1760000240, durationS: 240, notes: [] }
        : {
              path,
              format: path.endsWith(".db") ? "db" : "mcap",
              compatible: true,
              missing: [],
              color: { name: "/camera/color/image_raw", encoding: "rgb8", width: 1280, height: 720, frameId: "camera_color_optical_frame", count: 5412 },
              depth: { name: "/camera/aligned_depth_to_color/image_raw", encoding: "16UC1", width: 1280, height: 720, frameId: "camera_color_optical_frame", count: 5398 },
              colorInfo: "/camera/color/camera_info",
              depthInfo: "/camera/aligned_depth_to_color/camera_info",
              worldFrame: "odom",
              start: 1760000000,
              end: 1760000181,
              durationS: 181,
              notes: ["Depth is 16UC1 in millimeters."],
          }

let open = START === "pick" ? null : START === "incompatible" ? "/rec/lidar_only.mcap" : "/rec/parking_lot_cones.mcap"
let memoryState = START === "ready" ? "ready" : "none"
let generation = START === "generating" ? { started: Date.now(), total: 181, from: 40 } : null
let lastProgress = null
let cancelled = 0
let lastAnswer = null

function modelsJson() {
    const now = Date.now()
    if (models.downloading) {
        const speed = 85e6
        let budget = ((now - downloadStarted) / 1000) * speed
        for (const file of models.files) {
            const take = Math.min(file.bytes, budget)
            budget -= take
            file.doneBytes = Math.max(file.doneBytes, take)
            file.state = file.doneBytes >= file.bytes ? "ready" : take > 0 ? "downloading" : "missing"
        }
        if (models.files.every((f) => f.state === "ready")) {
            models.downloading = false
        }
    }
    const totalBytes = models.files.reduce((a, f) => a + f.bytes, 0)
    const doneBytes = models.files.reduce((a, f) => a + f.doneBytes, 0)
    return { dir: models.dir, ready: doneBytes >= totalBytes, totalBytes, doneBytes, bytesPerSecond: models.downloading ? 85e6 : null, etaS: models.downloading ? (totalBytes - doneBytes) / 85e6 : null, downloading: models.downloading, error: models.error, files: models.files }
}

function progressJson() {
    if (!generation) {
        return lastProgress ?? { phase: "idle", done: 0, total: 0, skipped: 0, elapsedS: 0, etaS: null, framesPerS: null, writtenBytes: 0, generation: 0, message: "" }
    }
    const m = modelsJson()
    const elapsedS = (Date.now() - generation.started) / 1000
    if (!m.ready) {
        return { phase: "models", done: 0, total: generation.total, skipped: 0, elapsedS, etaS: null, framesPerS: null, writtenBytes: 0, generation: 1, message: "downloading the models" }
    }
    if (!generation.embedStart) {
        generation.embedStart = Date.now()
    }
    const t = (Date.now() - generation.embedStart) / 1000
    if (t < 1.5) {
        return { phase: "planning", done: 0, total: 0, skipped: 0, elapsedS, etaS: null, framesPerS: null, writtenBytes: 0, generation: 1, message: "choosing a keyframe every 1.0 s" }
    }
    const done = Math.min(generation.total, (generation.from ?? 0) + Math.floor((t - 1.5) * 6.2))
    if (done >= generation.total) {
        generation = null
        memoryState = "ready"
        lastProgress = { phase: "done", done, total: done, skipped: 3, elapsedS, etaS: 0, framesPerS: 6.2, writtenBytes: done * 2.4e6, generation: 1, message: "" }
        return lastProgress
    }
    return { phase: "embedding", done, total: generation.total, skipped: 3, elapsedS, etaS: (generation.total - done) / 6.2, framesPerS: 6.2, writtenBytes: done * 2.4e6, generation: 1, message: `keyframe ${done} at 0:${String(done % 60).padStart(2, "0")}` }
}

function status() {
    const generationProgress = progressJson()
    const name = open?.split("/").pop()
    return {
        recording: open ? { path: open, name } : null,
        check: open ? check(open) : null,
        memory: open ? { state: generation ? (generationProgress.done > 0 ? "partial" : "none") : cancelled && memoryState !== "ready" ? "partial" : memoryState, generation: 1, frames: memoryState === "ready" ? 178 : generationProgress.done || cancelled, planned: 181, options: { everySeconds: 1 }, createdMs: Date.now() - 3600e3 } : null,
        models: modelsJson(),
        generation: generation || lastProgress ? generationProgress : null,
        loaded: memoryState === "ready" ? { frames: 178, patches: [1024, 576, 729], loadMs: 840 } : null,
        loading: false,
    }
}

// a little scene: a floor, two walls, a couple of cones, a bench; the camera path loops through it
function makeCloud() {
    const pts = []
    const add = (x, y, z, r, g, b) => pts.push([x, y, z, r, g, b])
    const jitter = () => (Math.random() - 0.5) * 0.04
    for (let x = -6; x <= 8; x += 0.07) {
        for (let y = -5; y <= 5; y += 0.07) {
            const shade = 70 + Math.random() * 25 + (Math.floor(x) % 2 === 0 ? 8 : 0)
            add(x + jitter(), y + jitter(), jitter() * 0.3, shade, shade + 4, shade + 10)
        }
    }
    for (let x = -6; x <= 8; x += 0.06) {
        for (let z = 0; z <= 2.6; z += 0.06) {
            add(x, 5 + jitter() * 0.3, z, 150 + Math.random() * 20, 140, 120)
        }
    }
    for (let y = -5; y <= 5; y += 0.06) {
        for (let z = 0; z <= 2.6; z += 0.06) {
            add(-6 + jitter() * 0.3, y, z, 110, 125 + Math.random() * 20, 150)
        }
    }
    const cone = (cx, cy) => {
        for (let z = 0; z <= 0.7; z += 0.025) {
            const r = 0.18 * (1 - z / 0.75) + 0.02
            for (let a = 0; a < Math.PI * 2; a += 0.2) {
                const stripe = z > 0.3 && z < 0.42
                add(cx + r * Math.cos(a), cy + r * Math.sin(a), z, stripe ? 240 : 255, stripe ? 240 : 110, stripe ? 240 : 20)
            }
        }
    }
    cone(2.4, -1.2)
    cone(5.1, 2.3)
    for (let x = 0; x <= 1.6; x += 0.05) {
        for (let y = 2.5; y <= 3.0; y += 0.05) {
            add(x - 3, y, 0.45, 120, 80, 50)
        }
    }
    const count = pts.length
    const buffer = Buffer.alloc(4 + count * 15)
    buffer.writeUInt32LE(count, 0)
    pts.forEach(([x, y, z, r, g, b], i) => {
        buffer.writeFloatLE(x, 4 + i * 12)
        buffer.writeFloatLE(y, 4 + i * 12 + 4)
        buffer.writeFloatLE(z, 4 + i * 12 + 8)
        const c = 4 + count * 12 + i * 3
        buffer[c] = Math.min(255, r)
        buffer[c + 1] = Math.min(255, g)
        buffer[c + 2] = Math.min(255, b)
    })
    return buffer
}
const cloud = makeCloud()
const frames = Array.from({ length: 178 }, (_, i) => {
    const a = (i / 178) * Math.PI * 2
    return { index: i, ts: 1760000000 + i, position: [1 + 4.5 * Math.cos(a), 0.2 + 3 * Math.sin(a), 0.42], rotation: [0, 0, Math.sin(a / 2), Math.cos(a / 2)] }
})

function answer(text) {
    const cones = /cone/i.test(text)
    const bench = /bench|seat|chair/i.test(text)
    const places = cones
        ? [
              { id: 0, center: [2.4, -1.2, 0.36], size: [0.42, 0.41, 0.72], heightM: 0.71, bottomZ: 0, topZ: 0.71, observations: [0, 1, 2, 3], best: 1, score: 0.31, firstSeen: 1760000012.4, lastSeen: 1760000097.1 },
              { id: 1, center: [5.1, 2.3, 0.35], size: [0.4, 0.43, 0.7], heightM: 0.7, bottomZ: 0, topZ: 0.7, observations: [4, 5], best: 4, score: 0.27, firstSeen: 1760000041.0, lastSeen: 1760000140.6 },
          ]
        : bench
          ? [{ id: 0, center: [-2.2, 2.75, 0.24], size: [1.65, 0.55, 0.5], heightM: 0.48, bottomZ: 0, topZ: 0.48, observations: [0, 1, 2], best: 0, score: 0.22, firstSeen: 1760000070, lastSeen: 1760000120 }]
          : []
    return {
        query: text,
        count: places.length,
        places,
        observations: places.flatMap((p) => p.observations.map(() => ({ frame: 10, ts: p.firstSeen, bbox: [400, 200, 600, 500], source: "owl", score: 0.3, agreement: 3, placed: null, place: p.id }))),
        candidates: [],
        unplaced: cones ? 1 : 0,
        worldFrame: "odom",
        timings: { textMs: 42, searchMs: 180, owlMs: 1210, placeMs: 35, totalMs: 1467 },
        summary: places.length ? `${places.length} ${text}${places.length === 1 ? "" : "s"}: ${places.map((p) => `${p.heightM.toFixed(2)} m tall at (${p.center[0].toFixed(1)}, ${p.center[1].toFixed(1)})`).join("; ")}.` : `No ${text} in the memory.`,
    }
}

function evidenceSvg(id, text) {
    const hue = id === 0 ? 30 : 200
    return `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400" viewBox="0 0 640 400">
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="hsl(${hue},25%,55%)"/><stop offset="1" stop-color="hsl(${hue},15%,22%)"/></linearGradient></defs>
<rect width="640" height="400" fill="url(#g)"/><rect y="250" width="640" height="150" fill="#3a3d44"/>
<polygon points="320,140 290,290 350,290" fill="#ff6a14"/><rect x="300" y="200" width="40" height="18" fill="#eee"/><rect x="270" y="288" width="100" height="10" fill="#e35a10"/>
<rect x="255" y="125" width="130" height="185" fill="none" stroke="#22e4ff" stroke-width="4"/>
<text x="258" y="118" font-family="sans-serif" font-size="18" fill="#22e4ff">${text} #${id + 1}</text></svg>`
}

const json = (res, body, code = 200) => {
    res.writeHead(code, { "content-type": "application/json" })
    res.end(JSON.stringify(body))
}

http.createServer((req, res) => {
    const url = new URL(req.url, "http://x")
    let path = url.pathname
    if (path === "/recordings") {
        return json(res, { dir: "~/dimos/recordings", extraDirs: [], recordings })
    }
    if (!path.startsWith(APP)) {
        return json(res, { error: "not found" }, 404)
    }
    path = path.slice(APP.length)
    let body = ""
    req.on("data", (chunk) => (body += chunk))
    req.on("end", () => {
        const input = body ? JSON.parse(body) : {}
        const slow = (ms, fn) => setTimeout(fn, ms)
        if (path === "api/status") return json(res, status())
        if (path === "api/recordings") return json(res, { dir: "~/dimos/recordings", recordings: recordings.map((r) => ({ id: r.id, name: r.name, path: r.path, format: r.format, bytes: r.size, modified: r.modified, memory: r.path === "/rec/parking_lot_cones.mcap" ? { state: memoryState, frames: 178, planned: 181, options: {} } : r.path.endsWith(".db") ? { state: "partial", frames: 60, planned: 300, options: {} } : null })) })
        if (path === "api/open") {
            open = input.path
            memoryState = open === "/rec/parking_lot_cones.mcap" && START === "ready" ? "ready" : "none"
            lastProgress = null
            return slow(400, () => json(res, status()))
        }
        if (path === "api/models") return json(res, modelsJson())
        if (path === "api/models/download") {
            if (!models.downloading && !modelsJson().ready) {
                models.downloading = true
                downloadStarted = Date.now()
            }
            return json(res, modelsJson())
        }
        if (path === "api/generate" && req.method === "POST") {
            if (!modelsJson().ready && !models.downloading) {
                models.downloading = true
                downloadStarted = Date.now()
            }
            generation = { started: Date.now(), total: 181, from: cancelled }
            lastProgress = null
            return json(res, status())
        }
        if (path === "api/generate/cancel") {
            const p = progressJson()
            cancelled = p.done
            generation = null
            models.downloading = false
            lastProgress = { ...p, phase: "cancelled", etaS: null, framesPerS: null }
            return json(res, lastProgress)
        }
        if (path === "api/memory" && req.method === "DELETE") {
            memoryState = "none"
            cancelled = 0
            lastProgress = null
            lastAnswer = null
            return json(res, status())
        }
        if (path === "api/query") {
            lastAnswer = answer(String(input.text ?? ""))
            return slow(1400, () => json(res, lastAnswer))
        }
        if (path === "api/answer") return json(res, lastAnswer)
        if (path === "api/frames") return json(res, { worldFrame: "odom", frames })
        if (path === "api/cloud") {
            res.writeHead(200, { "content-type": "application/octet-stream" })
            return res.end(cloud)
        }
        const evidence = path.match(/^api\/places\/(\d+)\/evidence$/)
        if (evidence) {
            res.writeHead(200, { "content-type": "image/svg+xml" })
            return res.end(evidenceSvg(Number(evidence[1]), url.searchParams.get("text") ?? ""))
        }
        json(res, { error: `no mock for ${path}` }, 404)
    })
}).listen(PORT, "127.0.0.1", () => console.log(`mock Desktop + Memory World API on :${PORT} (state ${START})`))
