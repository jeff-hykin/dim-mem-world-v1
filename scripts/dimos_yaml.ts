// Writes dimos.yaml's `provides:` from the server's own endpoint table (`dimos-app-server --agent-json`), so the
// manifest Desktop reads and the routes the server serves can't drift (server/src/manifest_test.rs checks the two).
//
//     cargo run -p dimos-app-server -- --agent-json | deno run --allow-read --allow-write scripts/dimos_yaml.ts
import { stringify } from "jsr:@std/yaml@1"

const manifest = JSON.parse(await new Response(Deno.stdin.readable).text())
const header = `title: Memory World
spec-version: v1.0 # the dimos.yaml spec version this file follows
uses:
    # Desktop's shared recordings folder: the page lists it to pick a recording
    "@desktop-gateway":
        - GET /recordings
    # the dim-app SDK's zenoh client: Desktop's "recordings changed" events refresh the picker
    "@zenoh-gateway": ">=0.5 <0.6"
# Desktop's nix builds of this app pull from this cache (CI fills it)
caches:
    - url: https://dimos-desktop.cachix.org
      key: "dimos-desktop.cachix.org-1:A4P35aGJGmCan92LWyamtSFXMqaVE+VRFYnrJ8QMTeQ="
`
const provides = stringify({ provides: { private: ["api/cloud"], description: manifest.description, endpoints: manifest.endpoints } }, { indent: 4, lineWidth: 120 })
await Deno.writeTextFile(new URL("../dimos.yaml", import.meta.url), header + provides)
console.log(`dimos.yaml: ${manifest.endpoints.length} endpoints`)
