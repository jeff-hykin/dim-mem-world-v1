{
    description = "dim-mem-world: Memory World as a dimOS Desktop app: a robot recording's visual memory, and where things are in 3D";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        rust-overlay.url = "github:oxalica/rust-overlay";
        rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    };

    outputs = { self, nixpkgs, rust-overlay }:
        let
            systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
            forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; overlays = [ (import rust-overlay) ]; }));
            # ONNX Runtime from nixpkgs, loaded by the server at run time (ort's load-dynamic): no model code is compiled in
            onnxLibrary = pkgs: "${pkgs.onnxruntime}/lib/libonnxruntime${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
        in {
            packages = forAllSystems (pkgs:
                let
                    rust = pkgs.rust-bin.stable.latest.default;
                    rustPlatform = pkgs.makeRustPlatform { cargo = rust; rustc = rust; };

                    # the page: React + Vite (type-checked first), from package-lock.json
                    frontend = pkgs.buildNpmPackage {
                        pname = "dim-mem-world-frontend";
                        version = "0.1.0";
                        src = ./frontend;
                        npmDepsHash = "sha256-fhrNxskKd16Tz179vo6/HZ8pvDnfHc2IWjCIezxIW6E=";
                        # npm unpacks the dimos-react git dependency through its cache
                        makeCacheWritable = true;
                        installPhase = ''
                            cp -r dist $out
                        '';
                    };

                    # the backend: one Rust binary (recordings, generation, questions, downloads); no Python at runtime
                    server = rustPlatform.buildRustPackage {
                        pname = "dim-mem-world-server";
                        version = "0.1.0";
                        # only what cargo reads, so a dimos.yaml or frontend change doesn't rebuild the server
                        src = pkgs.lib.fileset.toSource {
                            root = ./.;
                            fileset = pkgs.lib.fileset.unions [ ./Cargo.toml ./Cargo.lock ./crates ./server ];
                        };
                        cargoLock.lockFile = ./Cargo.lock;
                        cargoBuildFlags = [ "-p" "dimos-app-server" ];
                        # the tests need the downloaded models; `cargo test` runs them in the dev shell
                        doCheck = false;
                    };
                in {
                    inherit frontend server;
                    # what Desktop builds: bin/dimos-app-server, which serves everything under /apps/<name>/
                    dimosApp = pkgs.runCommand "dim-mem-world" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                        mkdir -p $out/bin
                        makeWrapper ${server}/bin/dimos-app-server $out/bin/dimos-app-server \
                            --set-default MEMWORLD_FRONTEND ${frontend} \
                            --set-default ORT_DYLIB_PATH ${onnxLibrary pkgs}
                        cp ${./icon.svg} $out/icon.svg
                    '';
                    default = self.packages.${pkgs.system}.dimosApp;
                });

            devShells = forAllSystems (pkgs: {
                default = pkgs.mkShell {
                    packages = [ pkgs.rust-bin.stable.latest.default pkgs.nodejs_22 pkgs.onnxruntime ];
                    ORT_DYLIB_PATH = onnxLibrary pkgs;
                };
            });
        };
}
