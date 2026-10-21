{
    description = "dim-mem-world: Memory World as a dimOS Desktop app: a robot recording's visual memory, and where things are in 3D";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        # the page's deno: nixos-25.11's 2.6.10, the version dimos pins (unstable's newer deno refuses packages younger than a day)
        nixpkgs-deno.url = "github:NixOS/nixpkgs/nixos-25.11";
        rust-overlay.url = "github:oxalica/rust-overlay";
        rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
        # deps build once into their own derivation (keyed on Cargo.toml / Cargo.lock), so a commit that only edits our code rebuilds just our crates
        crane.url = "github:ipetkov/crane";
        # lite_record's post_process: poses for a raw lidar + IMU recording (run in place, as the Map Editor does)
        lite_record.url = "github:jeff-hykin/lite_record/461c194e5d804ba6012c58442b65b87494113e1b";
    };

    outputs = { self, nixpkgs, nixpkgs-deno, rust-overlay, lite_record, crane }:
        let
            systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
            # unfree: only the CUDA libraries the x86_64 Linux build loads for Nvidia GPUs
            forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; overlays = [ (import rust-overlay) ]; config.allowUnfree = true; }));
            # x86_64 Linux: Microsoft's CUDA 12 build of ONNX Runtime (it also runs on the CPU when there's no Nvidia GPU),
            # its CUDA provider pointed at nixpkgs' CUDA 12 libraries; the driver's libcuda comes from the machine at run time
            onnxGpu = pkgs:
                let cuda = pkgs.cudaPackages_12;
                in pkgs.stdenv.mkDerivation {
                    pname = "onnxruntime-gpu";
                    version = "1.22.0";
                    src = pkgs.fetchzip {
                        url = "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz";
                        hash = "sha256-sOuAeyYT7xBm7kNJuBfE/A6enBNCQipC8T1HkRdnMKE=";
                    };
                    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
                    buildInputs = [ pkgs.stdenv.cc.cc.lib cuda.cuda_cudart cuda.libcublas cuda.libcurand cuda.libcufft cuda.cudnn ];
                    # TensorRT isn't shipped (its provider is never asked for); libcuda is the driver's
                    autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" "libnvinfer.so.10" "libnvinfer_plugin.so.10" "libnvonnxparser.so.10" ];
                    installPhase = ''
                        mkdir -p $out
                        cp -r lib include $out/
                        rm -f $out/lib/libonnxruntime_providers_tensorrt.so
                    '';
                };
            # ONNX Runtime, loaded by the server at run time (ort's load-dynamic): no model code is compiled in. nixpkgs'
            # has CoreML on a Mac; Linux on ARM stays on the CPU
            onnxLibrary = pkgs:
                if pkgs.stdenv.hostPlatform.system == "x86_64-linux" then "${onnxGpu pkgs}/lib/libonnxruntime.so"
                else "${pkgs.onnxruntime}/lib/libonnxruntime${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
        in {
            packages = forAllSystems (pkgs:
                let
                    rust = pkgs.rust-bin.stable.latest.default;
                    craneLib = (crane.mkLib pkgs).overrideToolchain (_: rust);

                    deno = nixpkgs-deno.legacyPackages.${pkgs.stdenv.hostPlatform.system}.deno;
                    denoEnv = "export HOME=$TMPDIR DENO_DIR=$TMPDIR/deno DENO_NO_UPDATE_CHECK=1 DENO_NO_PROMPT=1";
                    frontendFiles = paths: pkgs.lib.fileset.toSource { root = ./frontend; fileset = pkgs.lib.fileset.unions paths; };
                    # `deno install --frozen` (deno.lock pins every package) with network, so fixed-output. Its hash is per
                    # build machine (vite's native rollup and esbuild differ by OS and CPU): `deno task update-hashes` refreshes
                    # this machine's after frontend/deno.json or deno.lock change.
                    frontendModules = pkgs.stdenvNoCC.mkDerivation {
                        name = "dim-mem-world-frontend-modules";
                        src = frontendFiles [ ./frontend/deno.json ./frontend/deno.lock ];
                        nativeBuildInputs = [ deno ];
                        buildPhase = ''
                            ${denoEnv}
                            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
                            deno install --frozen
                            # deno's own record of the install: not needed, and not byte-identical between installs
                            rm -f node_modules/.deno/.setup-cache.bin
                        '';
                        installPhase = "cp -R node_modules $out";
                        dontFixup = true;
                        outputHashMode = "recursive";
                        outputHashAlgo = "sha256";
                        outputHash = {
                            aarch64-darwin = "sha256-oO+P8sfp+UkDIpcmAzY4FqTDhNeop0Y4AHHbxGjwf+Q=";
                            x86_64-darwin = "";
                            x86_64-linux = "sha256-UHGas9HqIgcuMMgGITTDW7mGb93EZH7dw7TW0SpmCJw=";
                            aarch64-linux = "sha256-ClPYW8y4sARYumBUMrq1Ntpyy73cDF1sVob461bjc8M=";
                        }.${pkgs.stdenv.hostPlatform.system};
                    };

                    pageFiles = frontendFiles [ ./frontend/deno.json ./frontend/deno.lock ./frontend/tsconfig.json ./frontend/index.html ./frontend/vite.config.ts ./frontend/tools ./frontend/src ./frontend/public ];
                    # dimos-react as esm.sh builds its tag (frontend/tools/esm_fetch.ts): a vite build with network that
                    # keeps only what it fetched, so fixed-output; the same on every system. `deno task update-hashes` refreshes it.
                    frontendEsm = pkgs.stdenvNoCC.mkDerivation {
                        name = "dim-mem-world-frontend-esm";
                        src = pageFiles;
                        nativeBuildInputs = [ deno ];
                        buildPhase = ''
                            ${denoEnv}
                            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
                            cp -R ${frontendModules} node_modules
                            chmod -R u+w node_modules
                            ESM_CACHE=$TMPDIR/esm deno run -A --cached-only --node-modules-dir=manual npm:vite build --logLevel warn
                        '';
                        installPhase = "cp -R $TMPDIR/esm $out";
                        dontFixup = true;
                        outputHashMode = "recursive";
                        outputHashAlgo = "sha256";
                        outputHash = "sha256-YzmFYPT9K/klsDkks1C/t+xpXsB78sqfd/6s2G+dDi4=";
                    };

                    # the page: React + Vite against those modules and that esm.sh cache, offline
                    frontend = pkgs.stdenvNoCC.mkDerivation {
                        name = "dim-mem-world-frontend";
                        src = pageFiles;
                        nativeBuildInputs = [ deno ];
                        buildPhase = ''
                            ${denoEnv}
                            cp -R ${frontendModules} node_modules
                            chmod -R u+w node_modules
                            ESM_CACHE=${frontendEsm} ESM_OFFLINE=1 deno run -A --cached-only --node-modules-dir=manual npm:vite build
                        '';
                        installPhase = "cp -R dist $out";
                    };

                    # the backend: one Rust binary (recordings, generation, questions, downloads); no Python at runtime
                    # crane: buildDepsOnly compiles every dependency against a dummy of our crates (so its hash only moves with Cargo.toml / Cargo.lock), buildPackage starts from that target dir
                    serverCommon = {
                        pname = "dim-mem-world-server";
                        version = "0.1.0";
                        # only what cargo reads, so a dimos.yaml or frontend change doesn't rebuild the server
                        src = pkgs.lib.fileset.toSource {
                            root = ./.;
                            fileset = pkgs.lib.fileset.unions [ ./Cargo.toml ./Cargo.lock ./crates ./server ];
                        };
                        cargoExtraArgs = "--locked -p dimos-app-server";
                        # the tests need the downloaded models; `cargo test` runs them in the dev shell
                        doCheck = false;
                        strictDeps = true;
                    };
                    server = craneLib.buildPackage (serverCommon // {
                        cargoArtifacts = craneLib.buildDepsOnly (serverCommon // {
                            # only the build's artifacts are reused; a `cargo check` pass would be a second compile nobody reads
                            cargoCheckCommand = "true";
                        });
                        # nothing builds on this target dir, so don't archive it into $out
                        doInstallCargoArtifacts = false;
                    });
                in {
                    inherit frontend frontendModules frontendEsm server;
                    # what Desktop builds: bin/dimos-app-server, which serves everything under /apps/<name>/
                    dimosApp = pkgs.runCommand "dim-mem-world" { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
                        mkdir -p $out/bin
                        makeWrapper ${server}/bin/dimos-app-server $out/bin/dimos-app-server \
                            --set-default MEMWORLD_FRONTEND ${frontend} \
                            --set-default ORT_DYLIB_PATH ${onnxLibrary pkgs} \
                            --set-default LITE_RECORD_BIN ${lite_record.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/lite_record
                        cp ${./icon.svg} $out/icon.svg
                    '';
                    default = self.packages.${pkgs.stdenv.hostPlatform.system}.dimosApp;
                });

            devShells = forAllSystems (pkgs: {
                default = pkgs.mkShell {
                    packages = [ pkgs.rust-bin.stable.latest.default nixpkgs-deno.legacyPackages.${pkgs.stdenv.hostPlatform.system}.deno pkgs.onnxruntime ];
                    ORT_DYLIB_PATH = onnxLibrary pkgs;
                };
            });
        };
}
