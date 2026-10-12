// Client for this app's backend (docs/api.md: the same endpoints Desktop's agent calls, at ./api) and Desktop's shared
// recordings list (../../recordings).

export type Vec3 = [number, number, number]

export interface DesktopRecording {
    id: string
    name: string
    format: "mcap" | "db" | string
    size: number
    modified: number
    path: string
    writable?: boolean
}

export interface ImageStream {
    name: string
    encoding: string
    width: number
    height: number
    frameId: string
    count: number
}

export interface Check {
    path: string
    format: "db" | "mcap"
    compatible: boolean
    missing: string[]
    color?: ImageStream
    colorInfo?: string
    /** the lidar the 3D map (and where things are) comes from */
    lidar?: string
    lidarFrame?: string
    worldFrame?: string
    /** set when Generate runs lite_record's post_process first to get poses (why) */
    postProcess?: string
    start: number
    end: number
    durationS: number
    notes: string[]
}

export interface MemoryStatus {
    state: "none" | "partial" | "ready"
    generation?: number
    frames: number
    planned: number
    options: Record<string, unknown>
    createdMs?: number
}

export interface ModelFile {
    name: string
    model: string
    bytes: number
    doneBytes: number
    state: "missing" | "downloading" | "verifying" | "ready" | "failed"
    error?: string
}

export interface Models {
    dir: string
    ready: boolean
    totalBytes: number
    doneBytes: number
    bytesPerSecond: number | null
    etaS: number | null
    downloading: boolean
    error: string | null
    files: ModelFile[]
}

export type Phase = "idle" | "models" | "post_process" | "planning" | "map" | "embedding" | "done" | "cancelled" | "failed"

export interface Progress {
    phase: Phase
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

export interface AppStatus {
    recording: { path: string; name: string } | null
    check: Check | null
    memory: MemoryStatus | null
    models: Models
    generation: Progress | null
    loaded: { frames: number; patches: number[]; loadMs: number } | null
    loading: boolean
}

export interface AppRecording {
    id: string
    name: string
    path: string
    format: string
    bytes: number
    modified: number
    memory: MemoryStatus | null
}

export interface Place {
    id: number
    center: Vec3
    size: Vec3
    heightM: number
    bottomZ: number
    topZ: number
    observations: number[]
    best: number
    score: number
    firstSeen: number
    lastSeen: number
}

export interface Observation {
    frame: number
    ts: number
    bbox: [number, number, number, number]
    source: "owl" | "agreement"
    score: number
    agreement: number
    placed: { center: number[]; size: number[]; min: number[]; max: number[]; support: number; range: number } | null
    place: number | null
}

export interface Answer {
    query: string
    count: number
    places: Place[]
    observations: Observation[]
    candidates: unknown[]
    unplaced: number
    worldFrame: string
    timings: { textMs: number; searchMs: number; owlMs: number; placeMs: number; totalMs: number }
    summary: string
}

export interface Keyframe {
    index: number
    ts: number
    position: Vec3
    rotation: number[]
}

export interface Cloud {
    count: number
    positions: Float32Array
    colors: Uint8Array
}

const here = (path: string) => new URL(path, location.href).href

async function call<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(here(path), { headers: { "content-type": "application/json" }, ...init })
    const body = await response.json().catch(() => ({}))
    if (!response.ok) {
        throw new Error(body?.error ?? `${response.status} ${response.statusText}`)
    }
    return body as T
}

const post = (body?: unknown): RequestInit => ({ method: "POST", body: body === undefined ? undefined : JSON.stringify(body) })

/** `u32 count`, then count × f32 xyz, then count × u8 rgb (little-endian) */
export function parseCloud(buffer: ArrayBuffer): Cloud {
    if (buffer.byteLength < 4) {
        return { count: 0, positions: new Float32Array(0), colors: new Uint8Array(0) }
    }
    const count = new DataView(buffer).getUint32(0, true)
    const expected = 4 + count * 12 + count * 3
    if (buffer.byteLength < expected) {
        throw new Error(`the point cloud is cut short (${buffer.byteLength} of ${expected} bytes)`)
    }
    // f32s at byte 4 are aligned, so a view works without copying
    const positions = new Float32Array(buffer, 4, count * 3)
    const colors = new Uint8Array(buffer, 4 + count * 12, count * 3)
    return { count, positions, colors }
}

export const api = {
    status: () => call<AppStatus>("api/status"),
    recordings: () => call<{ dir: string; recordings: AppRecording[] }>("api/recordings"),
    open: (path: string) => call<AppStatus>("api/open", post({ path })),
    models: () => call<Models>("api/models"),
    downloadModels: () => call<Models>("api/models/download", post({})),
    generate: (everySeconds: number) => call<AppStatus>("api/generate", post({ everySeconds })),
    cancel: () => call<Progress>("api/generate/cancel", post({})),
    forget: () => call<AppStatus>("api/memory", { method: "DELETE" }),
    query: (text: string) => call<Answer>("api/query", post({ text })),
    answer: () => call<Answer | null>("api/answer"),
    frames: () => call<{ worldFrame: string; frames: Keyframe[] }>("api/frames"),
    cloud: async (): Promise<Cloud> => {
        const response = await fetch(here("api/cloud"))
        if (!response.ok) {
            const body = await response.json().catch(() => ({}))
            throw new Error(body?.error ?? `${response.status} ${response.statusText}`)
        }
        return parseCloud(await response.arrayBuffer())
    },
    evidenceUrl: (place: number, text: string) => here(`api/places/${place}/evidence?text=${encodeURIComponent(text)}`),
}

/** Desktop's shared recordings folder */
export const desktop = {
    recordings: () => call<{ dir: string; extraDirs?: string[]; recordings: DesktopRecording[] }>("../../recordings"),
}
