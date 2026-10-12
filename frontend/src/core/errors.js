// Errors → Desktop's agent. A page reports what went wrong to Desktop's error feed (`POST /api/errors`), which a
// connected agent sees (its `recent_errors` tool, and the newest unacknowledged ones in `desktop_context`).
//
//     import { captureErrors, reportError } from "https://esm.sh/gh/jeff-hykin/dim-app@v0.7.0/errors.js"
//     captureErrors()                                   // uncaught errors + unhandled rejections, once per page
//     reportError("Couldn't save the map", String(e))   // a handled failure the user (and agent) should know about
//
// The URL is relative to the app (`/apps/<name>/` → `../../api/errors`). Reporting never throws, and a failed report is
// dropped (it never reports itself). Throttled: the same message at most once per 10 s, at most 20 reports a minute.

const SAME_MESSAGE_MS = 10_000
const PER_MINUTE = 20

/** @type {Map<string, number>} message → last sent (ms) */
const lastSent = new Map()
/** @type {number[]} send times in the last minute */
let recent = []
let installed = false

function appName() {
    try {
        const meta = document.querySelector('meta[name="dim-app"]')
        if (meta?.content) {
            return meta.content
        }
        const match = location.pathname.match(/^\/apps\/([^/]+)/)
        return match ? decodeURIComponent(match[1]) : "app"
    } catch {
        return "app"
    }
}

function allowed(message) {
    const now = Date.now()
    recent = recent.filter((at) => now - at < 60_000)
    if (recent.length >= PER_MINUTE) {
        return false
    }
    const last = lastSent.get(message)
    if (last !== undefined && now - last < SAME_MESSAGE_MS) {
        return false
    }
    lastSent.set(message, now)
    if (lastSent.size > 200) {
        lastSent.delete(lastSent.keys().next().value)
    }
    recent.push(now)
    return true
}

/**
 * Sends one error to Desktop's error feed. Resolves to `{ id, count }`, or null when it was throttled or couldn't be sent.
 * @param {string} message one line, readable
 * @param {unknown} [detail] a stack, a response body, anything (stringified)
 * @param {{ level?: "error" | "warning", source?: string }} [options]
 * @returns {Promise<{ id: number, count: number } | null>}
 */
export async function reportError(message, detail, options = {}) {
    try {
        const text = String(message ?? "").slice(0, 2000) || "error"
        if (!allowed(text)) {
            return null
        }
        const body = { source: options.source ?? appName(), message: text, level: options.level ?? "error" }
        if (detail !== undefined && detail !== null && detail !== "") {
            body.detail = (typeof detail === "string" ? detail : safeString(detail)).slice(0, 20_000)
        }
        const response = await fetch(new URL("../../api/errors", location.href), {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
        })
        return response.ok ? await response.json() : null
    } catch {
        return null
    }
}

function safeString(value) {
    try {
        if (value instanceof Error) {
            return value.stack || `${value.name}: ${value.message}`
        }
        return JSON.stringify(value) ?? String(value)
    } catch {
        return String(value)
    }
}

/**
 * Reports this page's uncaught errors and unhandled promise rejections. Idempotent: a second call does nothing.
 * @param {{ source?: string }} [options]
 */
export function captureErrors(options = {}) {
    if (installed || typeof globalThis.addEventListener !== "function") {
        return
    }
    installed = true
    globalThis.addEventListener("error", (event) => {
        // a failed <img>/<script> load is an Event without a message; only script errors are reported
        if (!event?.message) {
            return
        }
        const where = event.filename ? ` (${event.filename}:${event.lineno}:${event.colno})` : ""
        reportError(event.message, (event.error?.stack ?? "") + where, { source: options.source })
    })
    globalThis.addEventListener("unhandledrejection", (event) => {
        const reason = event?.reason
        const message = reason instanceof Error ? reason.message : String(reason)
        reportError(`Unhandled rejection: ${message}`, reason instanceof Error ? reason.stack : undefined, {
            source: options.source,
        })
    })
}
