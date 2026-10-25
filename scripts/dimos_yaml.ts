// Writes dimos.yaml's `provides:` (an OpenAPI 3.1 document) from the server's own endpoint table (`dimos-app-server
// --openapi-json`), so the manifest Desktop reads and the routes the server serves can't drift (a test in server/src/api.rs
// checks the two).
//
//     cargo run -p dimos-app-server -- --openapi-json | deno run --allow-read --allow-write scripts/dimos_yaml.ts
import { parse, stringify } from "jsr:@std/yaml@1"

const openapi = JSON.parse(await new Response(Deno.stdin.readable).text())
const file = new URL("../dimos.yaml", import.meta.url)
const before = (parse(await Deno.readTextFile(file).catch(() => "")) ?? {}) as { provides?: { paths?: object } }
// keys in a readable order: the ones listed first, the rest after as they come
const ordered = (value: Record<string, unknown>, first: string[]) =>
    Object.fromEntries([
        ...first.filter((k) => k in value).map((k) => [k, value[k]]),
        ...Object.entries(value).filter(([k]) => !first.includes(k)),
    ])
const operation = (op: Record<string, unknown>) => {
    const out = ordered(op, ["summary", "description", "x-when", "x-dimos-role", "parameters", "requestBody"])
    if (Array.isArray(out.parameters)) {
        out.parameters = out.parameters.map((p: Record<string, unknown>) => ordered(p, ["name", "in", "required", "schema", "description"]))
    }
    return out
}
// dimos.yaml's path order first, new paths after
const paths = Object.fromEntries(
    Object.entries(ordered(openapi.paths, Object.keys(before.provides?.paths ?? {}))).map(([path, item]) => [
        path,
        Object.fromEntries(Object.entries(item as object).map(([m, op]) => [m, operation(op as Record<string, unknown>)])),
    ]),
)
const header = `title: Memory World
version: "0.4.0" # this app's own version (its latest release tag); server/Cargo.toml has to match
spec-version: v1.5 # the dimos.yaml spec version this file follows
uses:
    # Desktop's shared recordings folder: the page lists it to pick a recording
    "@desktop-gateway":
        # what this server reads of DIMOS_APP; \`app check\` says when Desktop doesn't set one, and a launch that
        # can't resolve one says so in this app's own log
        env:
            DIMOS_APP.socket: the unix socket this server binds
            DIMOS_APP.dataDir: its own folder, for the worlds it keeps
            DIMOS_APP.recordingsDir:
                required: false
                description: where the recordings it reads are, when the gateway can't say
        endpoints:
            - GET /recordings
            # the page's uncaught errors go to Desktop's error feed (core/errors.js)
            - POST /ui/errors
            # the "Open Recordings" button looks for the Recordings app among the installed apps (dimos-react's desktop.js)
            - GET /info/apps
            # the zenoh client asks where the zenoh-gateway and this app's namespace are
            - GET /management/gateways/zenoh
    # the dim-app SDK's zenoh client: Desktop's "recordings changed" events refresh the picker
    "@zenoh-gateway": ">=0.5 <0.6"
# Desktop's nix builds of this app pull from this cache (.github/workflows/ci.yml fills it)
caches:
    - url: https://dimos-desktop.cachix.org
      key: "dimos-desktop.cachix.org-1:A4P35aGJGmCan92LWyamtSFXMqaVE+VRFYnrJ8QMTeQ="
`
const info = ordered(openapi.info, ["title", "version", "description", "x-when"])
const provides = stringify({ provides: { openapi: openapi.openapi, info, "x-private": ["api/cloud"], paths } }, { indent: 4, lineWidth: 120 })
await Deno.writeTextFile(file, header + provides)
console.log(`dimos.yaml: ${Object.values(paths).flatMap(Object.keys).length} operations`)
