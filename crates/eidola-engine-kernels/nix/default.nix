# Ahead-of-time GPU kernel build: pinned CUDA toolkit + pinned upstream kernel
# sources -> cubins, fatbins, SASS inspection, and manifest.json.
#
# Self-contained: with no arguments it pins its own nixpkgs (the same revision
# as the repository flake) and evaluates for the current Linux system. The
# repository flake can import it later as
#
#   import ./crates/eidola-engine-kernels/nix { nixpkgsSrc = nixpkgs; inherit system; }
#
# Unfree software is admitted by an allow-list of the CUDA toolkit components
# this build actually uses, never by a blanket allowUnfree, so nothing else
# NVIDIA-licensed can enter the closure unnoticed.
{
  nixpkgsSrc ? builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/78e9c786dc08cd4f3420c2395cd977206a9b1da2.tar.gz";
    sha256 = "sha256-VfjaoJ1Uyb7JZTrBgE5Jf2nhjtQPmD9KOd19HJwmZwM=";
  },
  nixpkgsRev ? "78e9c786dc08cd4f3420c2395cd977206a9b1da2",
  system ? builtins.currentSystem,
}:
let
  cudaComponents = [
    "cuda_nvcc"
    "cuda_cudart"
    "cuda_cccl"
    "cuda_crt"
    "cuda_cuobjdump"
    "cuda_nvdisasm"
    "cuda_cuxxfilt"
    "libcurand"
    "libnvvm"
  ];
  lib = import "${nixpkgsSrc}/lib";
  pkgs = import nixpkgsSrc {
    inherit system;
    config.allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) cudaComponents;
  };

  # CUDA 13.2: the newest toolkit this nixpkgs packages completely (13.3's CCCL
  # is marked unsupported here). Blackwell datacenter targets sm_100a/sm_103a
  # and the sm_100f family arrived in 12.9; 13.x is the line CUTLASS 4.8 and
  # current DeepGEMM/FlashInfer are developed against.
  cuda = pkgs.cudaPackages_13_2;
  hostCc = cuda.backendStdenv.cc;

  sources = import ./sources.nix { inherit (pkgs) fetchFromGitHub; };

  toolchain = {
    nixpkgs = nixpkgsRev;
    cuda = cuda.cuda_nvcc.version;
    host_compiler = "gcc ${hostCc.version}";
    components = lib.genAttrs cudaComponents (c: cuda.${c}.version);
  };

  sourcesRecord = lib.mapAttrs (_: s: {
    repository = "https://github.com/${s.owner}/${s.repo}";
    inherit (s) rev license;
    nar_hash = s.hash;
  }) sources;

  cudaInclude = lib.concatStringsSep ":" [
    "${lib.getInclude cuda.cuda_cudart}/include"
    "${lib.getInclude cuda.cuda_crt}/include"
    "${lib.getInclude cuda.libcurand}/include"
  ];
in
cuda.backendStdenv.mkDerivation {
  pname = "eidola-engine-kernels";
  version = "0";

  src = lib.fileset.toSource {
    root = ../.;
    # Everything this derivation evaluates or reads, so the build can hash it
    # into the manifest's inputs: the recipe itself included.
    fileset = lib.fileset.unions [
      ../csrc
      ./build-kernels.sh
      ./default.nix
      ./render-flashinfer-sink.sh
      ./sources.nix
    ];
  };

  nativeBuildInputs = [
    cuda.cuda_nvcc
    cuda.cuda_cuobjdump
    cuda.cuda_nvdisasm
    cuda.cuda_cuxxfilt
    pkgs.jq
  ];

  # The nvcc setup hook would add -Xfatbin=-compress-all to NVCC_PREPEND_FLAGS;
  # every flag is explicit in the build script instead.
  dontSetupCUDAToolkitCompilers = true;
  hardeningDisable = [ "all" ];
  dontConfigure = true;
  dontFixup = true;

  env = {
    CUTLASS_SRC = "${sources.cutlass.src}";
    DEEPGEMM_SRC = "${sources.deepgemm.src}";
    FLASHINFER_SRC = "${sources.flashinfer.src}";
    CUDA_INCLUDE = cudaInclude;
    HOST_CC_BIN = "${hostCc}/bin";
    TOOLCHAIN_JSON = builtins.toJSON toolchain;
    SOURCES_JSON = builtins.toJSON sourcesRecord;
  };

  buildPhase = ''
    runHook preBuild
    export KERNELS_DIR=$PWD OUT=$out JOBS=$NIX_BUILD_CORES SCRATCH_DIR=$TMPDIR/scratch
    bash nix/build-kernels.sh
    runHook postBuild
  '';
  dontInstall = true;

  passthru = { inherit cuda sources toolchain; };

  meta = {
    description = "AOT-compiled Blackwell kernels with a reproducibility manifest";
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };
}
