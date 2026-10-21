// A Vite plugin that bundles https imports (dimos-react from esm.sh, as deno.json names it) instead of leaving them
// for the browser: each URL is fetched once into a cache folder, and the modules it imports by absolute path ("/gh/…")
// or relative path resolve against its URL. Bare imports esm.sh leaves (react, react-dom, three, kept external with
// ?external=…) resolve to this page's own, so there is one copy of each.
//   ESM_CACHE   the cache folder (default node_modules/.esm-cache)
//   ESM_OFFLINE set: a URL not in the cache is an error (nix builds offline from a prefetched cache)
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import process from "node:process"
import type { Plugin } from "vite"

const isUrl = (id: string) => /^https?:\/\//.test(id)

export function esmFetch(aliases: Record<string, string>): Plugin {
    const cacheDir = process.env.ESM_CACHE ?? join(process.cwd(), "node_modules", ".esm-cache")
    const offline = !!process.env.ESM_OFFLINE
    const fileFor = (url: string) => join(cacheDir, createHash("sha256").update(url).digest("hex") + ".js")
    return {
        name: "esm-fetch",
        enforce: "pre",
        resolveId(id, importer) {
            if (id in aliases) {
                return aliases[id]
            }
            if (isUrl(id)) {
                return id
            }
            // inside a fetched module: its absolute and relative imports are on its host
            if (importer && isUrl(importer) && (id.startsWith("/") || id.startsWith("."))) {
                return new URL(id, importer).href
            }
            return null
        },
        async load(id) {
            if (!isUrl(id)) {
                return null
            }
            const file = fileFor(id)
            if (existsSync(file)) {
                return readFileSync(file, "utf8")
            }
            if (offline) {
                throw new Error(`${id} isn't in the prefetched cache ${cacheDir} (ESM_OFFLINE): its nix hash needs updating`)
            }
            const response = await fetch(id)
            if (!response.ok) {
                throw new Error(`${id}: ${response.status} ${response.statusText}`)
            }
            const code = await response.text()
            mkdirSync(cacheDir, { recursive: true })
            writeFileSync(file, code)
            return code
        },
    }
}
