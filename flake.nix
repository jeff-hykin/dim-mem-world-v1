{
    description = "dim-mem-world: Memory World as a dimOS Desktop app: a robot recording's visual memory, and where things are in 3D";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        rust-overlay.url = "github:oxalica/rust-overlay";
        rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
        # lite_record's post_process: poses for a raw lidar + IMU recording (run in place, as the Map Editor does)
        lite_record.url = "github:jeff-hykin/lite_record/461c194e5d804ba6012c58442b65b87494113e1b";
    };

    outputs = { self, nixpkgs, rust-overlay, lite_record }:
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
                            --set-default ORT_DYLIB_PATH ${onnxLibrary pkgs} \
                            --set-default LITE_RECORD_BIN ${lite_record.packages.${pkgs.system}.default}/bin/lite_record
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
