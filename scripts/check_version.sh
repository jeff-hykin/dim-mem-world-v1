#!/usr/bin/env bash
# server/Cargo.toml's version is the app's own: `CARGO_PKG_VERSION` is what the server logs at startup and what
# `app_version` in a recording's metadata is. The repo bumps dimos.yaml on release, so the crate has to move with it --
# it sat at 0.1.0 while dimos.yaml said 0.3.1, stamping a version that never shipped onto everything it wrote.
#
# A shell check and not a #[test], because the nix build narrows its source to the crate and so cannot read dimos.yaml:
# an include_str!("../dimos.yaml") passes locally and fails in the build. `dimos-desktop app check .` reports the same
# drift from the outside (Desktop 0.2.227).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate=$(sed -n '/^\[package\]/,/^\[/p' "$root/server/Cargo.toml" | sed -n 's/^version *= *"\([^"]*\)".*/\1/p' | head -1)
manifest=$(sed -n 's/^version: *"\{0,1\}\([^"]*\)"\{0,1\} *\(#.*\)\{0,1\}$/\1/p' "$root/dimos.yaml" | head -1)
if [ "$crate" != "$manifest" ]; then
    echo "server/Cargo.toml says version $crate, dimos.yaml says $manifest: that version is what the server reports" >&2
    echo "of itself and writes into a recording's app_version, so a release bumps both" >&2
    exit 1
fi
echo "server/Cargo.toml and dimos.yaml agree on $crate"
