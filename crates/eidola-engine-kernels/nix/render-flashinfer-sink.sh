#!/usr/bin/env bash
# Render FlashInfer's FA2 paged-prefill configuration for the attention-sink
# variant, the same translation-unit prologue FlashInfer's Python JIT writes
# (`gen_customize_batch_prefill_module`, backend "fa2"), without Python.
#
#   render-flashinfer-sink.sh <flashinfer-src> <out-dir>
#
# Writes <out-dir>/batch_prefill_config.inc. Every piece of text comes from the
# pinned upstream tree: the config template is
# csrc/batch_prefill_customize_config.jinja, and the variant body is the
# `attention_sink_fa2_decl` string in flashinfer/jit/attention/variants.py.
# Only the template variables are filled in here, with the values FlashInfer's
# `BatchAttentionWithAttentionSinkWrapper` passes for BF16 Q/KV/O, int32
# indices, head_dim_qk 192 / head_dim_vo 128, no positional encoding in the
# kernel, and a sliding window. The output is refused if any template syntax
# survives, so an upstream template change fails the build instead of
# compiling something nobody reviewed.
set -euo pipefail

src=$1
out=$2
mkdir -p "$out"

template="$src/csrc/batch_prefill_customize_config.jinja"
variants="$src/flashinfer/jit/attention/variants.py"

# The variant is the body of a Python raw string: everything strictly between
# the `attention_sink_fa2_decl = r"""` line and the next line that is exactly
# `"""`.
awk '
  /^attention_sink_fa2_decl = r"""$/ { inside = 1; found = 1; next }
  inside && /^"""$/ { inside = 0; exit }
  inside { print }
  END { if (!found) exit 1 }
' "$variants" >"$out/attention_sink_fa2_decl.inc"
grep -q 'struct AttentionSink' "$out/attention_sink_fa2_decl.inc"

# Host-binding macros (ADDITIONAL_FUNC_PARAMS / ADDITIONAL_PARAMS_SETTER) only
# expand inside FlashInfer's own host binding, which is never compiled here, so
# they render empty. The FP4-KV block is dropped: KV is 16-bit.
awk '
  /^\{% if require_fp4_kv_cache %\}$/ { skip = 1; next }
  skip && /^\{% endif %\}$/ { skip = 0; next }
  skip { next }
  /^\{\{ variant_decl \}\}$/ { while ((getline line < decl) > 0) print line; next }
  { print }
' decl="$out/attention_sink_fa2_decl.inc" "$template" |
  sed \
    -e 's/{{ additional_func_params }}//' \
    -e 's/{{ additional_params_setter }}//' \
    -e 's/{{ variant_name }}/AttentionSink/' \
    -e 's/{{ dtype_q }}/nv_bfloat16/' \
    -e 's/{{ dtype_kv }}/nv_bfloat16/' \
    -e 's/{{ dtype_o }}/nv_bfloat16/' \
    -e 's/{{ idtype }}/int32_t/' \
    -e 's/{{ head_dim_qk }}/192/' \
    -e 's/{{ head_dim_vo }}/128/' \
    -e 's/{{ use_fp16_qk_reduction }}/false/' \
    -e 's/{{ use_logits_soft_cap }}/false/' \
    -e 's/{{ pos_encoding_mode }}/PosEncodingMode::kNone/' \
    -e 's/{{ use_sliding_window }}/true/' \
    -e 's/{{ paged_kv_stride_mode | upper }}/INDEPENDENT/' \
    -e 's/{{ additional_params_decl }}/float* sink;\n  double sm_scale;/' \
    >"$out/batch_prefill_config.inc"

if grep -n '{{\|{%' "$out/batch_prefill_config.inc"; then
  echo "render-flashinfer-sink: unrendered template syntax above" >&2
  exit 1
fi
rm "$out/attention_sink_fa2_decl.inc"
