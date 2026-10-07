#!/usr/bin/env bash
# Compile every kernel in csrc/kernels.json to cubins (one per target arch),
# bundle them into per-kernel fatbins, inspect the SASS, and write
# manifest.json. Run by the Nix derivation in default.nix; it can also be run
# by hand inside the same toolchain (the reproducibility experiments do, to
# vary the build directory and parallelism).
#
# Inputs (environment):
#   KERNELS_DIR      crate root (contains csrc/ and nix/)
#   CUTLASS_SRC      pinned upstream trees
#   DEEPGEMM_SRC
#   FLASHINFER_SRC
#   CUDA_INCLUDE     colon-separated toolkit include dirs (cudart, crt, curand)
#   HOST_CC_BIN      directory holding the host compiler nvcc drives
#   TOOLCHAIN_JSON   toolchain identity record (written into the manifest)
#   SOURCES_JSON     upstream pin record (written into the manifest)
#   OUT              output directory
#   JOBS             parallel compiler invocations (default: nproc)
#   SCRATCH_DIR      scratch directory (default: a fresh mktemp dir)
#
# Output bytes must be a function of the sources, the flags, and the
# toolchain only. Two things are pinned here for that:
#   * nvcc names internal-linkage device symbols after a module ID that cicc
#     derives from the absolute source path, so every source is compiled with
#     a fixed `--orig_src_path_name` (its path in the repository) instead of
#     wherever the build happens to run;
#   * the environment's NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS and the host
#     compiler wrapper's NIX_CFLAGS_COMPILE / NIX_HARDENING_ENABLE are cleared,
#     so no flag reaches nvcc or its preprocessor that is not in the manifest.
set -euo pipefail
export LC_ALL=C
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS NVCC_CCBIN NIX_CFLAGS_COMPILE NIX_CFLAGS_LINK \
  NIX_LDFLAGS NIX_HARDENING_ENABLE CUDA_NVCC_FLAGS

: "${KERNELS_DIR:?}" "${CUTLASS_SRC:?}" "${DEEPGEMM_SRC:?}" "${FLASHINFER_SRC:?}"
: "${CUDA_INCLUDE:?}" "${HOST_CC_BIN:?}" "${TOOLCHAIN_JSON:?}" "${SOURCES_JSON:?}" "${OUT:?}"
JOBS=${JOBS:-$(nproc)}
SCRATCH_DIR=${SCRATCH_DIR:-$(mktemp -d)}

spec="$KERNELS_DIR/csrc/kernels.json"
repo_csrc="crates/eidola-engine-kernels/csrc"

mkdir -p "$SCRATCH_DIR/inc/generated" "$SCRATCH_DIR/obj" "$OUT/cubin" "$OUT/fatbin" "$OUT/inspect"
cp -r "$KERNELS_DIR/csrc" "$SCRATCH_DIR/csrc"
ln -sfn "$CUTLASS_SRC" "$SCRATCH_DIR/inc/cutlass"
ln -sfn "$DEEPGEMM_SRC" "$SCRATCH_DIR/inc/deepgemm"
ln -sfn "$FLASHINFER_SRC" "$SCRATCH_DIR/inc/flashinfer"
bash "$KERNELS_DIR/nix/render-flashinfer-sink.sh" "$FLASHINFER_SRC" \
  "$SCRATCH_DIR/inc/generated/flashinfer_sink"

toolkit_includes=()
IFS=: read -r -a cuda_include_dirs <<<"$CUDA_INCLUDE"
for dir in "${cuda_include_dirs[@]}"; do
  toolkit_includes+=("-I$dir")
done

# One line per (kernel, arch): name source profile arch.
jq -r '.archs[] as $arch | .kernels[] | [.name, .source, .profile, $arch] | @tsv' "$spec" \
  >"$SCRATCH_DIR/jobs.tsv"

compile_one() {
  local name=$1 source=$2 profile=$3 arch=$4
  local flags=() includes=()
  mapfile -t flags < <(jq -r --arg p "$profile" '.common_flags[], .profiles[$p].flags[]' "$spec")
  while IFS= read -r inc; do
    includes+=("-I$SCRATCH_DIR/inc/$inc")
  done < <(jq -r --arg p "$profile" '.profiles[$p].includes[]' "$spec")
  local cubin="$OUT/cubin/$name.$arch.cubin"
  (
    cd "$SCRATCH_DIR"
    nvcc "--compiler-bindir=$HOST_CC_BIN" "-arch=$arch" "${flags[@]}" \
      -Xcicc --orig_src_path_name -Xcicc "$repo_csrc/$source" \
      -I"$SCRATCH_DIR/csrc" "${includes[@]}" "${toolkit_includes[@]}" \
      "csrc/$source" -o "$cubin"
  ) >"$SCRATCH_DIR/obj/$name.$arch.log" 2>&1 || {
    cat "$SCRATCH_DIR/obj/$name.$arch.log" >&2
    echo "build-kernels: $name for $arch failed" >&2
    return 1
  }
}
export -f compile_one
export spec repo_csrc SCRATCH_DIR OUT HOST_CC_BIN
export TOOLKIT_INCLUDES="${toolkit_includes[*]}"
# Re-split the toolkit include list inside the worker (arrays do not export).
compile_job() {
  local toolkit_includes
  read -r -a toolkit_includes <<<"$TOOLKIT_INCLUDES"
  compile_one "$@"
}
export -f compile_job

tr '\t' '\n' <"$SCRATCH_DIR/jobs.tsv" | xargs -d '\n' -n 4 -P "$JOBS" bash -c 'compile_job "$@"' _

# Per-kernel fatbin over the arch-specific cubins. Compression is pinned off so
# the images inside are the cubins byte for byte.
fatbin_archs=$(jq -r '.fatbin_archs[]' "$spec")
for name in $(jq -r '.kernels[].name' "$spec"); do
  images=()
  for arch in $fatbin_archs; do
    images+=("--image3=kind=elf,sm=${arch#sm_},file=$OUT/cubin/$name.$arch.cubin")
  done
  fatbinary --64 --compress=false --create="$OUT/fatbin/$name.fatbin" "${images[@]}"
done

# SASS inspection: entry symbols (with demangled names, each bound to its
# launch-contract record) and a histogram of the tensor-core instruction
# families, so the manifest states which MMA path each kernel actually takes.
meta_json() {
  cuobjdump -symbols "$1" | awk '$1 == "STT_OBJECT" && $4 ~ /_meta$/ { print $4 }' | sort |
    jq -R . | jq -s .
}
# An entry's record is `<entry>_meta` when the image has one, or the record
# kernels.json's `meta_aliases` maps to the entry (mangled template instances
# cannot carry a matching name). Every entry must resolve to exactly one record
# and every record to exactly one entry, or the build fails.
entries_json() {
  local cubin=$1 name=$2 arch=$3 symbols metas aliases
  symbols=$(cuobjdump -symbols "$cubin" | awk '$3 == "STO_ENTRY" { print $4 }' | sort |
    while read -r sym; do
      jq -n --arg symbol "$sym" --arg demangled "$(cu++filt "$sym")" \
        '{symbol: $symbol, demangled: $demangled}'
    done | jq -s .)
  metas=$(meta_json "$cubin")
  aliases=$(jq -c --arg n "$name" '.kernels[] | select(.name == $n) | .meta_aliases // {}' "$spec")
  jq -n --argjson entries "$symbols" --argjson metas "$metas" --argjson aliases "$aliases" \
    --arg where "$name $arch" '
    def has_meta($m): any($metas[]; . == $m);
    def records($s):
      [$aliases | to_entries[] | select(.value == $s) | .key]
      + [$s + "_meta" | select(has_meta(.))];
    [$entries[] | . + {records: records(.symbol)}] as $resolved
    | [$resolved[] | select((.records | length) != 1) | .symbol] as $bad
    | if ($bad | length) > 0 then
        error("\($where): entries without exactly one launch-contract record: \($bad)")
      else . end
    | [$resolved[] | {symbol, demangled, meta: .records[0]}] as $out
    | if ([$out[].meta] | sort) != ($metas | sort) then
        error("\($where): records \($metas) do not correspond one to one with entries \([$out[].meta])")
      else $out end'
}
mma_json() {
  cuobjdump -sass "$1" >"$SCRATCH_DIR/sass.txt"
  # Opcode = first token after the address comment and any predicate guard,
  # without modifiers.
  awk 'match($0, /^[ \t]+\/\*[0-9a-f]+\*\/[ \t]+/) {
         rest = substr($0, RLENGTH + 1)
         sub(/^@!?U?P[0-9T]+[ \t]+/, "", rest)
         split(rest, tok, /[ .;]/)
         print tok[1]
       }' "$SCRATCH_DIR/sass.txt" |
    grep -E '^(UTC|HMMA|QMMA|OMMA|IMMA|HGMMA|QGMMA)' | sort | uniq -c |
    awk '{ printf "{\"%s\": %d}\n", $2, $1 }' | jq -s 'add // {}'
}

kernels_json="$SCRATCH_DIR/kernels.json"
: >"$kernels_json"
while IFS=$'\t' read -r name source profile arch; do
  cubin="$OUT/cubin/$name.$arch.cubin"
  cuobjdump -sass "$cubin" >"$OUT/inspect/$name.$arch.sass"
  cuobjdump -res-usage "$cubin" >"$OUT/inspect/$name.$arch.res-usage" 2>&1
  entries=$(entries_json "$cubin" "$name" "$arch")
  jq -n \
    --arg name "$name" --arg arch "$arch" --arg source "csrc/$source" --arg profile "$profile" \
    --arg file "cubin/$name.$arch.cubin" \
    --arg sha256 "$(sha256sum "$cubin" | cut -d' ' -f1)" \
    --argjson size "$(stat -c %s "$cubin")" \
    --argjson entries "$entries" \
    --argjson meta "$(meta_json "$cubin")" \
    --argjson mma "$(mma_json "$cubin")" \
    '{name: $name, arch: $arch, source: $source, profile: $profile, file: $file,
      sha256: $sha256, size: $size, entries: $entries, meta: $meta, mma_sass: $mma}' \
    >>"$kernels_json"
done <"$SCRATCH_DIR/jobs.tsv"

fatbins_json="$SCRATCH_DIR/fatbins.json"
: >"$fatbins_json"
for name in $(jq -r '.kernels[].name' "$spec"); do
  fatbin="$OUT/fatbin/$name.fatbin"
  jq -n --arg name "$name" --arg file "fatbin/$name.fatbin" \
    --arg sha256 "$(sha256sum "$fatbin" | cut -d' ' -f1)" \
    --argjson size "$(stat -c %s "$fatbin")" \
    --argjson archs "$(jq '.fatbin_archs' "$spec")" \
    '{name: $name, file: $file, sha256: $sha256, size: $size, archs: $archs}' >>"$fatbins_json"
done

inputs_json="$SCRATCH_DIR/inputs.json"
(
  cd "$KERNELS_DIR"
  for f in csrc/* nix/build-kernels.sh nix/render-flashinfer-sink.sh; do
    jq -n --arg path "$f" --arg sha256 "$(sha256sum "$f" | cut -d' ' -f1)" \
      '{path: $path, sha256: $sha256}'
  done
  jq -n --arg path "generated/flashinfer_sink/batch_prefill_config.inc" \
    --arg sha256 "$(sha256sum "$SCRATCH_DIR/inc/generated/flashinfer_sink/batch_prefill_config.inc" | cut -d' ' -f1)" \
    '{path: $path, sha256: $sha256}'
) | jq -s 'sort_by(.path)' >"$inputs_json"

cp "$SCRATCH_DIR/inc/generated/flashinfer_sink/batch_prefill_config.inc" "$OUT/inspect/"

jq -n \
  --argjson toolchain "$TOOLCHAIN_JSON" \
  --argjson sources "$SOURCES_JSON" \
  --slurpfile spec "$spec" \
  --slurpfile inputs "$inputs_json" \
  --slurpfile kernels "$kernels_json" \
  --slurpfile fatbins "$fatbins_json" \
  '{
     schema_version: 2,
     toolchain: $toolchain,
     sources: $sources,
     compile: {
       per_source_flags: ["-Xcicc", "--orig_src_path_name", "-Xcicc", "crates/eidola-engine-kernels/<source>"],
       common_flags: $spec[0].common_flags,
       profiles: $spec[0].profiles
     },
     inputs: $inputs[0],
     kernels: ($kernels | sort_by(.name, .arch)),
     fatbins: ($fatbins | sort_by(.name))
   }' >"$OUT/manifest.json"
