# Upstream kernel sources, fetched by commit and verified by NAR hash. Nothing
# from these trees is committed to this repository; the build reads their
# headers (and, for FlashInfer, two template strings) from the store.
#
# Moving a pin changes kernel bytes: update the rev and hash here, rebuild,
# and commit the regenerated manifest in the same change.
{ fetchFromGitHub }:
let
  pin =
    attrs:
    attrs
    // {
      src = fetchFromGitHub { inherit (attrs) owner repo rev hash; };
    };
in
{
  # Tag v4.8.0.
  cutlass = pin {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "098de2a652cf8f00fd70b2df54051c7eccbb855a";
    hash = "sha256-FiXSg6DMUAV9Gf5LOePKAwmGLVPxphq6MVwHZwwsKDM=";
    license = "BSD-3-Clause";
  };

  # Upstream's 2026-09-30 public release; device headers only (no JIT, no
  # torch). Its own CUTLASS submodule is not fetched: the kernels compile
  # against the CUTLASS pin above.
  deepgemm = pin {
    owner = "deepseek-ai";
    repo = "DeepGEMM";
    rev = "057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7";
    hash = "sha256-qTNMUhWGKxD9RYSpvao1omo7+vfQvc2gbKOSLOMWaTA=";
    license = "MIT";
  };

  # Tag v0.7.0.post1. Only include/ and two template strings are read; no
  # submodule (its bundled CUTLASS/CCCL) is used.
  flashinfer = pin {
    owner = "flashinfer-ai";
    repo = "flashinfer";
    rev = "946200de1ae94fc93fdd0926f0a13afd1fa7f0f1";
    hash = "sha256-lv+JQRsQJXRN6cc0ywJsXmKqLGUYZq0fs6PnvSUKudQ=";
    license = "Apache-2.0";
  };
}
