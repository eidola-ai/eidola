// The DeepSeek-V4.1 kernels' bodies as a host library: one function per
// launch the executor makes, running the body the device compiles
// (`eidola-engine-kernels`' `csrc/ds_ops.cuh`, `ds_attention.cuh`,
// `ds_index.cuh`) thread by thread over the grid the launch wrapper chose.
// `tests/ds_common` loads it as a device's kernels, so the executor's host
// scenarios run over the same C++ a GPU runs and a wrong parameter, grid or
// assumption about a kernel fails there.
//
// Dev-only: never a build input of the kernels or of anything shipped. Each
// function repeats its kernel's entry (`csrc/ds_*.cu`): which thread calls
// which body, with the lane sums a device collects by warp shuffles gathered
// in a loop, and shared memory as a local. Blocks and threads run last first,
// so a body that leaned on the order of a launch would differ.
//
//   c++ -std=c++17 -O2 -ffp-contract=off -shared -fPIC \
//       -I ../eidola-engine-kernels/csrc oracle/ds_launch_host.cc -o libds_launch_host.so

#include <memory>
#include <vector>

#include "ds_attention.cuh"
#include "ds_index.cuh"
#include "ds_ops.cuh"

namespace {

// A launch of independent threads over grid (gx, gy, gz), last block and
// thread first.
template <class F>
void launch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t threads, F body) {
  for (uint32_t z = gz; z-- > 0;)
    for (uint32_t y = gy; y-- > 0;)
      for (uint32_t x = gx; x-- > 0;)
        for (uint32_t t = threads; t-- > 0;) body(x, y, z, t);
}

// The 16 lane sums of one lane group.
template <class L>
void lanes16(float* lanes, L lane) {
  for (uint32_t j = 16; j-- > 0;) lanes[j] = lane(j);
}

}  // namespace

extern "C" {

void dsh_embed(const DsEmbedParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::embed_thread(*p, x, y, t); });
}

void dsh_quant_fp8(const DsQuantFp8Params* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kBlockThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::quant_fp8_thread(*p, x, y, t); });
}

// `eidola_ds_rot_quant_e4m3` for the global KV's format, `eidola_ds_rot_quant` otherwise, as
// the wrapper picks.
void dsh_rot_quant(const DsRotQuantParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  const bool e4m3 = p->format == DS_KV_FP4_E4M3;
  launch(gx, gy, 1, dsk::kBlockThreads, [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) {
    dsk::rot_quant_thread(*p, x, y, t, e4m3);
  });
}

void dsh_rms_norm(const DsRmsNormParams* p, uint32_t gx, uint32_t, uint32_t) {
  for (uint32_t row = gx; row-- > 0;) {
    float lanes[16];
    lanes16(lanes, [&](uint32_t j) { return dsk::rms_lane(*p, row, j); });
    const float factor = dsk::rms_finish(*p, row, lanes);
    for (uint32_t t = dsk::kElementThreads; t-- > 0;) dsk::rms_store(*p, row, t, factor);
  }
}

void dsh_mhc_mix(const DsMhcMixParams* p, uint32_t gx, uint32_t, uint32_t) {
  for (uint32_t row = gx; row-- > 0;) {
    float dots[dsk::kMixThreads / 16];
    for (uint32_t g = dsk::mix_groups(p->copies); g-- > 0;) {
      float lanes[16];
      lanes16(lanes, [&](uint32_t j) { return dsk::mhc_lane(*p, row, g, j); });
      dots[g] = dsk::mhc_group(*p, row, g, lanes);
    }
    dsk::mhc_finish(*p, row, dots);
  }
}

void dsh_collapse(const DsCollapseParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::collapse_thread(*p, x, y, t); });
}

void dsh_expand(const DsExpandParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::expand_thread(*p, x, y, t); });
}

void dsh_linear(const DsLinearParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  for (uint32_t row = gy; row-- > 0;) {
    for (uint32_t bx = gx; bx-- > 0;) {
      for (uint32_t g = 16; g-- > 0;) {
        const uint32_t n0 = (bx * 16 + g) * dsk::kLinearChains;
        float acc[16][dsk::kLinearChains];
        for (uint32_t lane = 16; lane-- > 0;) dsk::linear_lane(*p, row, n0, lane, acc[lane]);
        for (uint32_t u = 0; u < dsk::kLinearChains; ++u) {
          float lanes[16];
          lanes16(lanes, [&](uint32_t j) { return acc[j][u]; });
          if (n0 + u < p->n) dsk::linear_finish(*p, row, n0 + u, lanes);
        }
      }
    }
  }
}

void dsh_head(const DsLinearParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  for (uint32_t by = gy; by-- > 0;) {
    const uint32_t row0 = by * dsk::kHeadChains;
    for (uint32_t bx = gx; bx-- > 0;) {
      for (uint32_t g = 16; g-- > 0;) {
        const uint32_t n = bx * 16 + g;
        float acc[16][dsk::kHeadChains];
        for (uint32_t lane = 16; lane-- > 0;) dsk::head_lane(*p, row0, n, lane, acc[lane]);
        for (uint32_t u = 0; u < dsk::kHeadChains; ++u) {
          float lanes[16];
          lanes16(lanes, [&](uint32_t j) { return acc[j][u]; });
          if (n < p->n && row0 + u < p->rows) dsk::linear_finish(*p, row0 + u, n, lanes);
        }
      }
    }
  }
}

void dsh_pool(const DsPoolParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::pool_thread(*p, x, y, t); });
}

void dsh_route(const DsRouteParams* p, uint32_t gx, uint32_t, uint32_t) {
  for (uint32_t row = gx; row-- > 0;) {
    float score[dsk::kMaxExperts], biased[dsk::kMaxExperts];
    for (uint32_t e = p->experts; e-- > 0;) dsk::route_score(*p, row, e, score, biased);
    dsk::route_select(*p, row, score, biased);
  }
}

// Both placement launches, each complete before the next.
void dsh_place(const DsPlaceParams* p, uint32_t gx, uint32_t, uint32_t) {
  launch(gx, 1, 1, dsk::kBlockThreads,
         [&](uint32_t x, uint32_t, uint32_t, uint32_t t) { dsk::place_count_thread(*p, x, t); });
  launch(gx, 1, 1, dsk::kBlockThreads,
         [&](uint32_t x, uint32_t, uint32_t, uint32_t t) { dsk::place_thread(*p, x, t); });
}

void dsh_place_rows(const DsPlaceRowsParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads, [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) {
    dsk::place_rows_thread(*p, x, y, t);
  });
}

void dsh_swiglu(const DsSwigluParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kBlockThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::swiglu_thread(*p, x, y, t); });
}

void dsh_combine(const DsCombineParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::combine_thread(*p, x, y, t); });
}

void dsh_engram_gather(const DsEngramGatherParams* p, uint32_t gx, uint32_t gy, uint32_t gz) {
  launch(gx, gy, gz, dsk::kBlockThreads, [&](uint32_t x, uint32_t y, uint32_t z, uint32_t t) {
    dsk::engram_gather_thread(*p, x, y, z, t);
  });
}

void dsh_engram_inject(const DsEngramInjectParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  for (uint32_t row = gy; row-- > 0;) {
    for (uint32_t copy = gx; copy-- > 0;) {
      float sums[3];
      for (uint32_t g = 3; g-- > 0;) {
        float lanes[16];
        lanes16(lanes, [&](uint32_t j) { return dsk::inject_lane(*p, row, copy, g, j); });
        sums[g] = dsk::inject_group(*p, row, copy, g, lanes);
      }
      const float gate = dsk::inject_gate(*p, sums);
      for (uint32_t t = dsk::kElementThreads; t-- > 0;) dsk::inject_store(*p, row, copy, t, gate);
    }
  }
}

void dsh_lin8(const DsLin8Params* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::lin8_thread(*p, x, y, t); });
}

void dsh_lin4(const DsLin4Params* p, uint32_t gx, uint32_t gy, uint32_t) {
  launch(gx, gy, 1, dsk::kElementThreads,
         [&](uint32_t x, uint32_t y, uint32_t, uint32_t t) { dsk::lin4_thread(*p, x, y, t); });
}

void dsh_grouped_dot(const DsGroupedDotParams* p, uint32_t gx, uint32_t gy, uint32_t gz) {
  for (uint32_t row = gz; row-- > 0;) {
    for (uint32_t group = gy; group-- > 0;) {
      for (uint32_t bx = gx; bx-- > 0;) {
        for (uint32_t g = 16; g-- > 0;) {
          const uint32_t i0 = (bx * 16 + g) * dsk::kLinearChains;
          float acc[16][dsk::kLinearChains];
          for (uint32_t lane = 16; lane-- > 0;) {
            dsk::grouped_lane(*p, row, group, i0, lane, acc[lane]);
          }
          for (uint32_t u = 0; u < dsk::kLinearChains; ++u) {
            float lanes[16];
            lanes16(lanes, [&](uint32_t j) { return acc[j][u]; });
            if (i0 + u < p->rank) dsk::grouped_finish(*p, row, group, i0 + u, lanes);
          }
        }
      }
    }
  }
}

void dsh_attention(const DsAttnParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  auto sh = std::make_unique<ds_attn::Shared>();
  for (uint32_t row = gy; row-- > 0;) {
    for (uint32_t bx = gx; bx-- > 0;) {
      ds_attn::attend(*p, *sh, bx, gx, row, ds::Par{ds_attn::kThreads, true});
    }
  }
}

void dsh_index_score(const DsIndexScoreParams* p, uint32_t gx, uint32_t gy, uint32_t) {
  auto sh = std::make_unique<ds_index::ScoreShared>();
  for (uint32_t row = gy; row-- > 0;) {
    for (uint32_t bx = gx; bx-- > 0;) {
      ds_index::index_score(*p, *sh, bx, gx, row, ds::Par{ds_index::kScoreThreads, true});
    }
  }
}

void dsh_select(const DsSelectParams* p, uint32_t gx, uint32_t, uint32_t) {
  auto sh = std::make_unique<ds_index::SelectShared>();
  for (uint32_t row = gx; row-- > 0;) {
    ds_index::select(*p, *sh, row, ds::Par{ds_index::kSelectThreads, true});
  }
}

}  // extern "C"
