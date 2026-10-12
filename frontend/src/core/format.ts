// Numbers as people read them: sizes, rates, durations, clock times.

export function bytes(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) {
        return "–"
    }
    if (value >= 1e9) {
        return `${(value / 1e9).toFixed(2)} GB`
    }
    if (value >= 1e6) {
        return `${(value / 1e6).toFixed(1)} MB`
    }
    if (value >= 1e3) {
        return `${(value / 1e3).toFixed(0)} KB`
    }
    return `${Math.round(value)} B`
}

export function gb(value: number): string {
    return `${(value / 1e9).toFixed(2)} GB`
}

export function rate(bytesPerSecond: number | null | undefined): string {
    return bytesPerSecond == null || !Number.isFinite(bytesPerSecond) ? "–" : `${(bytesPerSecond / 1e6).toFixed(1)} MB/s`
}

/** m:ss (h:mm:ss past an hour); "–" when unknown */
export function clock(seconds: number | null | undefined): string {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) {
        return "–"
    }
    const total = Math.round(seconds)
    const h = Math.floor(total / 3600)
    const m = Math.floor((total % 3600) / 60)
    const s = String(total % 60).padStart(2, "0")
    return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`
}

/** "4 min 12 s" style, for a recording's length */
export function duration(seconds: number | null | undefined): string {
    if (seconds == null || !Number.isFinite(seconds)) {
        return "–"
    }
    if (seconds < 60) {
        return `${Math.round(seconds)} s`
    }
    const minutes = Math.floor(seconds / 60)
    return minutes < 60 ? `${minutes} min ${Math.round(seconds % 60)} s` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

export function age(epochSeconds: number): string {
    // Desktop's `modified` is epoch seconds; tolerate ms
    const seconds = epochSeconds > 1e12 ? epochSeconds / 1000 : epochSeconds
    const ago = Date.now() / 1000 - seconds
    if (ago < 3600) {
        return `${Math.max(1, Math.round(ago / 60))} min ago`
    }
    if (ago < 86400 * 2) {
        return `${Math.round(ago / 3600)} h ago`
    }
    return new Date(seconds * 1000).toLocaleDateString()
}

export const meters = (value: number) => `${value.toFixed(2)} m`

export const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString()} ${count === 1 ? one : many}`
