// Refreshes flake.nix's fixed-output hashes for this machine: the page's `deno install` (frontendModules, one hash per
// system) and its dimos-react prefetch (frontendEsm, one for all). Each is blanked, built, and set to the hash nix reports.
// Run after changing frontend/deno.json, deno.lock or the dimos-react tag; other systems' frontendModules hashes are
// refreshed the same way on a machine of that system (CI's ubuntu runner is x86_64-linux).
//   deno task update-hashes
const flake = new URL("../flake.nix", import.meta.url)
const system = new TextDecoder().decode(
    (await new Deno.Command("nix", { args: ["eval", "--impure", "--raw", "--expr", "builtins.currentSystem"] })
        .output()).stdout,
)

/** builds `.#attr` and returns the hash nix says it got, or null when it built (the hash was already right) */
async function gotHash(attr: string): Promise<string | null> {
    const { stderr, success } = await new Deno.Command("nix", {
        args: ["build", `.#${attr}`, "--no-link"],
        cwd: new URL("..", import.meta.url),
        stderr: "piped",
    }).output()
    const text = new TextDecoder().decode(stderr)
    const got = text.match(/got:\s+(sha256-[A-Za-z0-9+/=]+)/)?.[1]
    if (got) {
        return got
    }
    if (!success) {
        throw new Error(`nix build .#${attr} failed:\n${text.slice(-2000)}`)
    }
    return null
}

/** sets the hash on the line `pattern` matches (its `= "…"`) and returns the old one */
function setHash(pattern: RegExp, hash: string): string {
    const text = Deno.readTextFileSync(flake)
    const match = text.match(pattern)
    if (!match) {
        throw new Error(`no ${pattern} in flake.nix`)
    }
    Deno.writeTextFileSync(flake, text.replace(pattern, (line) => line.replace(/"[^"]*"/, `"${hash}"`)))
    return match[0].match(/"([^"]*)"/)![1]
}

const targets: [string, RegExp][] = [
    ["frontendModules", new RegExp(`${system} = "[^"]*";`)],
    ["frontendEsm", /outputHash = "[^"]*";\n(?=[\s\S]*?# the page: React)/],
]
for (const [attr, pattern] of targets) {
    const old = setHash(pattern, "")
    const got = await gotHash(attr)
    setHash(pattern, got ?? old)
    console.log(
        `${attr} (${attr === "frontendModules" ? system : "every system"}): ${
            got && got !== old ? `${old || "(none)"} -> ${got}` : "unchanged"
        }`,
    )
}
