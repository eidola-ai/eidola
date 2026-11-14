// Host check of the DeepSeek attention, index and selection kernels: the bodies
// in `csrc/ds_attention.cuh` and `csrc/ds_index.cuh`, compiled for the host
// and run block by block and phase by phase (`csrc/ds_body.cuh`), in ascending
// and in descending thread order.
//
// Dev-only: never a build input of the kernels or of anything shipped. It
// shows the bodies' arithmetic and bookkeeping are the reference's; what nvcc
// emits for the device is held by the `exact` profile and the GPU tests.
//
//   c++ -std=c++17 -O2 -ffp-contract=off -I csrc oracle/ds_attention_host.cc -o ds_attention_host
//   ./ds_attention_host vectors tests/data/ds_attention.vectors
//   ./ds_attention_host jobs <in> <out>
//
// `vectors` checks every record of the file `eidola-engine-model`'s
// `tests/deepseek_attention_vectors.rs` emits (which documents the format).
// `jobs` runs launches as a device would: `eidola-engine-cuda`'s tests build
// kernels' parameter structs and one buffer holding every array they name, and
// run the same case here when there is no device. A job is
//
//   u32 launches, u32 0, u64 buffer bytes, u64 the address the parameters give
//   the buffer's first byte
//   per launch: u32 kernel (0 attention, 1 window slots, 2 index scores,
//   3 selection), u32 grid x, u32 grid y, u32 parameter bytes, then the
//   parameter struct padded to 8 bytes
//   the buffer, padded to 8 bytes
//
// and the output file is each job's buffer after its launches in order, run in
// ascending thread order, followed by the same run in descending order.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ds_attention.cuh"
#include "ds_index.cuh"

namespace {

using Bytes = std::vector<uint8_t>;

[[noreturn]] void die(const std::string& what) {
  std::cerr << "ds_attention_host: " << what << "\n";
  std::exit(2);
}

// ---- launches ----

// Shared memory starts as garbage on a device.
template <class T>
std::unique_ptr<T> garbage() {
  auto sh = std::make_unique<T>();
  std::memset(static_cast<void*>(sh.get()), 0xa5, sizeof(T));
  return sh;
}

void launch_attn(const DsAttnParams& p, uint32_t gx, uint32_t gy, bool descending) {
  for (uint32_t row = 0; row < gy; ++row) {
    for (uint32_t bx = 0; bx < gx; ++bx) {
      auto sh = garbage<ds_attn::Shared>();
      ds_attn::attend(p, *sh, bx, gx, row, ds::Par{ds_attn::kThreads, descending});
    }
  }
}

void launch_window(const DsWindowSlotsParams& p, uint32_t gx, uint32_t gy, bool descending) {
  for (uint32_t row = 0; row < gy; ++row) {
    for (uint32_t bx = 0; bx < gx; ++bx) {
      for (uint32_t i = 0; i < ds_attn::kSlotThreads; ++i) {
        ds_attn::window_slot(p, bx, row, descending ? ds_attn::kSlotThreads - 1 - i : i);
      }
    }
  }
}

void launch_score(const DsIndexScoreParams& p, uint32_t gx, uint32_t gy, bool descending) {
  for (uint32_t row = 0; row < gy; ++row) {
    for (uint32_t bx = 0; bx < gx; ++bx) {
      auto sh = garbage<ds_index::ScoreShared>();
      ds_index::index_score(p, *sh, bx, gx, row, ds::Par{ds_index::kScoreThreads, descending});
    }
  }
}

void launch_select(const DsSelectParams& p, uint32_t gx, bool descending) {
  for (uint32_t row = 0; row < gx; ++row) {
    auto sh = garbage<ds_index::SelectShared>();
    ds_index::select(p, *sh, row, ds::Par{ds_index::kSelectThreads, descending});
  }
}

// ---- jobs ----

struct Reloc {
  uint64_t from;
  uint8_t* to;
  uint64_t bytes;
  template <class T>
  void operator()(T*& ptr) const {
    const uint64_t v = reinterpret_cast<uint64_t>(ptr);
    if (v == 0) return;
    if (v < from || v - from > bytes) die("a parameter points outside the job's buffer");
    ptr = reinterpret_cast<T*>(to + (v - from));
  }
};

template <class T>
T read_params(const Bytes& raw) {
  if (raw.size() != sizeof(T)) die("parameter struct size");
  T p;
  std::memcpy(&p, raw.data(), sizeof(T));
  return p;
}

void run_job(uint32_t kernel, uint32_t gx, uint32_t gy, const Bytes& raw, const Reloc& at,
             bool descending) {
  switch (kernel) {
    case 0: {
      auto p = read_params<DsAttnParams>(raw);
      at(p.q), at(p.slots), at(p.sink), at(p.out), at(p.a.base), at(p.b.base);
      launch_attn(p, gx, gy, descending);
      break;
    }
    case 1: {
      auto p = read_params<DsWindowSlotsParams>(raw);
      at(p.positions), at(p.first), at(p.tables), at(p.table_row), at(p.slots), at(p.status);
      launch_window(p, gx, gy, descending);
      break;
    }
    case 2: {
      auto p = read_params<DsIndexScoreParams>(raw);
      at(p.q_codes), at(p.q_scales), at(p.head_dots), at(p.tables), at(p.table_row);
      at(p.visible), at(p.scores), at(p.status), at(p.keys.base);
      launch_score(p, gx, gy, descending);
      break;
    }
    case 3: {
      auto p = read_params<DsSelectParams>(raw);
      at(p.scores), at(p.visible), at(p.mask_in), at(p.mask_out), at(p.tables);
      at(p.table_row), at(p.slots), at(p.status);
      launch_select(p, gx, descending);
      break;
    }
    default:
      die("unknown kernel");
  }
}

int jobs(const char* in_path, const char* out_path) {
  std::ifstream in(in_path, std::ios::binary);
  if (!in) die(std::string("cannot read ") + in_path);
  std::ofstream out(out_path, std::ios::binary);
  struct Call {
    uint32_t head[4];
    Bytes raw;
  };
  auto read = [&](void* to, size_t bytes) {
    if (!in.read(static_cast<char*>(to), static_cast<std::streamsize>(bytes))) die("truncated job");
  };
  uint32_t launches[2];
  size_t count = 0;
  while (in.read(reinterpret_cast<char*>(launches), sizeof launches)) {
    uint64_t sizes[2];
    read(sizes, sizeof sizes);
    std::vector<Call> calls(launches[0]);
    for (Call& call : calls) {
      read(call.head, sizeof call.head);
      call.raw.resize((call.head[3] + 7u) & ~7u);
      read(call.raw.data(), call.raw.size());
      call.raw.resize(call.head[3]);
    }
    // 8-byte words keep the buffer aligned for every array a kernel reads.
    std::vector<uint64_t> buffer((sizes[0] + 7) / 8);
    read(buffer.data(), buffer.size() * 8);
    for (int descending = 0; descending < 2; ++descending) {
      std::vector<uint64_t> work = buffer;
      const Reloc at{sizes[1], reinterpret_cast<uint8_t*>(work.data()), sizes[0]};
      for (const Call& call : calls) {
        run_job(call.head[0], call.head[1], call.head[2], call.raw, at, descending != 0);
      }
      out.write(reinterpret_cast<const char*>(work.data()),
                static_cast<std::streamsize>(work.size() * 8));
    }
    ++count;
  }
  out.close();
  if (!out) die(std::string("cannot write ") + out_path);
  std::cout << "ran " << count << " jobs\n";
  return 0;
}

// ---- vectors ----

struct Line {
  std::vector<std::string> tokens;
  size_t at = 0;
  const std::string& next() {
    if (at >= tokens.size()) die("short record");
    return tokens[at++];
  }
  uint64_t num() { return std::stoull(next(), nullptr, 16); }
  uint32_t u32() { return static_cast<uint32_t>(num()); }
  void expect(const char* token) {
    if (next() != token) die(std::string("expected ") + token);
  }
  template <class T>
  std::vector<T> list(size_t n) {
    std::vector<T> v(n);
    for (auto& x : v) x = static_cast<T>(num());
    return v;
  }
  // A run of bytes as one token of hex digits, `-` when empty.
  Bytes blob() {
    const std::string& s = next();
    Bytes b;
    if (s == "-") return b;
    if (s.size() % 2) die("odd hex blob");
    for (size_t i = 0; i < s.size(); i += 2) {
      b.push_back(static_cast<uint8_t>(std::stoul(s.substr(i, 2), nullptr, 16)));
    }
    return b;
  }
};

// A plane of `n` entries given back to back (each its codes, then its scales), laid out as a
// pool: blocks of `per_block` entries, each block its codes then its scales after a gap.
struct Pool {
  Bytes bytes;
  DsKvPlane plane;
  Pool(const Bytes& entries, uint32_t format, uint32_t width, uint32_t per_block) {
    const uint32_t cb = ds_kv_code_bytes(format, width), sb = ds_kv_scale_bytes(format, width);
    const size_t n = entries.size() / (cb + sb);
    if (entries.size() % (cb + sb)) die("plane bytes");
    const uint64_t gap = 24;
    const uint64_t block_bytes = gap + static_cast<uint64_t>(per_block) * (cb + sb) + 8;
    const size_t blocks = (n + per_block - 1) / per_block;
    bytes.assign(blocks * block_bytes + 1, 0xee);
    for (size_t e = 0; e < n; ++e) {
      uint8_t* block = bytes.data() + (e / per_block) * block_bytes;
      const size_t o = e % per_block;
      std::memcpy(block + gap + o * cb, &entries[e * (cb + sb)], cb);
      std::memcpy(block + gap + per_block * cb + 8 + o * sb, &entries[e * (cb + sb) + cb], sb);
    }
    plane = DsKvPlane{bytes.data(), block_bytes, gap, gap + per_block * cb + 8, per_block,
                      format,       width,       0};
  }
};

template <class T>
bool same(const std::vector<T>& got, const std::vector<T>& want, const char* what, size_t line) {
  if (got == want) return true;
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::cerr << "line " << line << ": " << what << " differs at " << i << ": got " << std::hex
            << (i < got.size() ? static_cast<uint64_t>(got[i]) : 0) << ", want "
            << (i < want.size() ? static_cast<uint64_t>(want[i]) : 0) << std::dec << "\n";
  return false;
}

// attn <heads> <dim> <scale> <layout> <a format> <b format> <a slots> <b slots> <a per block>
//   <b per block> : <q> <sink> <slots> <a entries> <b entries> = <out>
bool check_attn(Line& l, size_t line) {
  DsAttnParams p{};
  p.rows = 1;
  p.heads = l.u32();
  p.head_dim = l.u32();
  p.scale = ds::f32_from_bits(l.u32());
  p.layout = l.u32();
  const uint32_t a_format = l.u32(), b_format = l.u32();
  p.a_slots = l.u32();
  p.b_slots = l.u32();
  const uint32_t a_per_block = l.u32(), b_per_block = l.u32();
  l.expect(":");
  const size_t width = static_cast<size_t>(p.heads) * p.head_dim;
  const auto q = l.list<uint16_t>(width);
  const auto sink_bits = l.list<uint32_t>(p.heads);
  // Two rows of slots, the first never read: a stride larger than the list.
  p.slot_stride = p.a_slots + p.b_slots + 3;
  auto slots = l.list<uint32_t>(p.a_slots + p.b_slots);
  slots.resize(p.slot_stride, 0x12345678u);
  const Pool a(l.blob(), a_format, p.head_dim, a_per_block);
  const Pool b(l.blob(), b_format, p.head_dim, b_per_block);
  l.expect("=");
  const auto want = l.list<uint16_t>(width);
  std::vector<float> sink(p.heads);
  for (size_t h = 0; h < sink.size(); ++h) sink[h] = ds::f32_from_bits(sink_bits[h]);
  p.q = q.data();
  p.slots = slots.data();
  p.sink = sink.data();
  p.a = a.plane;
  p.b = b.plane;
  bool ok = true;
  // Every grid width a host may pick, in both thread orders.
  for (uint32_t group = 1; group <= ds_attn::kMaxGroup; ++group) {
    const uint32_t gx = (p.heads + group - 1) / group;
    if (ds_attn::heads_per_block(p.heads, gx) != group) continue;
    for (int descending = 0; descending < 2; ++descending) {
      std::vector<uint16_t> out(width, 0xdead);
      p.out = out.data();
      launch_attn(p, gx, 1, descending != 0);
      ok = same(out, want, "attention output", line) && ok;
    }
  }
  return ok;
}

// win <order> <window> <positions per block> <position> <first> <table words> : <table>
//   = <slots> <status>
bool check_window(Line& l, size_t line) {
  DsWindowSlotsParams p{};
  p.rows = 1;
  p.order = l.u32();
  p.window = l.u32();
  p.block_positions = l.u32();
  const uint32_t position = l.u32(), first = l.u32();
  p.table_stride = l.u32();
  l.expect(":");
  const auto table = l.list<int32_t>(p.table_stride);
  l.expect("=");
  const auto want = l.list<uint32_t>(p.window);
  const uint32_t want_status = l.u32();
  const uint32_t table_row = 0;
  p.positions = &position;
  p.first = &first;
  p.tables = table.data();
  p.table_row = &table_row;
  p.slot_stride = p.window + 2;
  bool ok = true;
  for (int descending = 0; descending < 2; ++descending) {
    std::vector<uint32_t> slots(p.slot_stride, 0x5a5a5a5au);
    uint32_t status = 0;
    p.slots = slots.data();
    p.status = &status;
    launch_window(p, (p.window + ds_attn::kSlotThreads - 1) / ds_attn::kSlotThreads, 1,
                  descending != 0);
    ok = slots[p.window] == 0x5a5a5a5au && ok;
    slots.resize(p.window);
    ok = same(slots, want, "window slots", line) && ok;
    ok = same(std::vector<uint32_t>{status}, {want_status}, "status", line) && ok;
  }
  return ok;
}

// score <heads> <dim> <weight scale> <keys per block> <visible> : <q codes> <q scales>
//   <head dots> <key entries> = <scores>
bool check_score(Line& l, size_t line) {
  DsIndexScoreParams p{};
  p.rows = 1;
  p.heads = l.u32();
  p.dim = l.u32();
  p.weight_scale = ds::f32_from_bits(l.u32());
  const uint32_t per_block = l.u32();
  const uint32_t visible = l.u32();
  l.expect(":");
  const Bytes q_codes = l.blob(), q_scales = l.blob();
  const auto head_dots = l.list<uint16_t>(p.heads);
  const Pool keys(l.blob(), DS_KV_FP4_E8M0, p.dim, per_block);
  l.expect("=");
  const auto want = l.list<uint16_t>(visible);
  // The identity table, reversed, over a pool laid out to match.
  const uint32_t blocks = (visible + per_block - 1) / per_block;
  std::vector<int32_t> table(blocks + 1);
  for (uint32_t b = 0; b < blocks; ++b) table[b] = static_cast<int32_t>(b);
  const uint32_t table_row = 0;
  p.q_codes = q_codes.data();
  p.q_scales = q_scales.data();
  p.head_dots = head_dots.data();
  p.tables = table.data();
  p.table_row = &table_row;
  p.visible = &visible;
  p.keys = keys.plane;
  p.table_stride = blocks + 1;
  p.score_stride = visible + 5;
  bool ok = true;
  for (uint32_t gx : {1u, 2u, 7u}) {
    for (int descending = 0; descending < 2; ++descending) {
      std::vector<uint16_t> scores(p.score_stride, 0xbeef);
      uint32_t status = 0;
      p.scores = scores.data();
      p.status = &status;
      launch_score(p, gx, 1, descending != 0);
      ok = scores[visible] == 0xbeef && status == 0 && ok;
      scores.resize(visible);
      ok = same(scores, want, "index scores", line) && ok;
    }
  }
  return ok;
}

// select <visible> <top k> <top blocks> <block entries> <entries per block> <mask words>
//   <mask in?> <mask out?> : <scores> <mask in> <table> = <slots> <mask out> <status>
bool check_select(Line& l, size_t line) {
  DsSelectParams p{};
  p.rows = 1;
  const uint32_t visible = l.u32();
  p.top_k = l.u32();
  p.top_blocks = l.u32();
  p.block_entries = l.u32();
  p.entries_per_block = l.u32();
  p.mask_words = l.u32();
  const bool has_in = l.u32() != 0, has_out = l.u32() != 0;
  l.expect(":");
  auto scores = l.list<uint16_t>(visible);
  const auto mask_in = l.list<uint32_t>(has_in ? p.mask_words : 0);
  p.table_stride = (visible + p.entries_per_block - 1) / p.entries_per_block + 1;
  const auto table = l.list<int32_t>(p.table_stride);
  l.expect("=");
  const auto want = l.list<uint32_t>(p.top_k);
  const auto want_mask = l.list<uint32_t>(has_out ? p.mask_words : 0);
  const uint32_t want_status = l.u32();
  p.score_stride = visible + 3;
  scores.resize(p.score_stride, 0x7f80);
  const uint32_t table_row = 0;
  p.scores = scores.data();
  p.visible = &visible;
  p.mask_in = has_in ? mask_in.data() : nullptr;
  p.tables = table.data();
  p.table_row = &table_row;
  p.slot_first = 3;
  p.slot_stride = p.slot_first + p.top_k + 2;
  bool ok = true;
  for (int descending = 0; descending < 2; ++descending) {
    std::vector<uint32_t> slots(p.slot_stride, 0x5a5a5a5au);
    std::vector<uint32_t> mask(p.mask_words + 1, 0x33333333u);
    uint32_t status = 0;
    p.slots = slots.data();
    p.mask_out = has_out ? mask.data() : nullptr;
    p.status = &status;
    launch_select(p, 1, descending != 0);
    ok = slots[0] == 0x5a5a5a5au && slots[p.slot_first + p.top_k] == 0x5a5a5a5au &&
         mask[p.mask_words] == 0x33333333u && ok;
    const std::vector<uint32_t> got(slots.begin() + p.slot_first,
                                    slots.begin() + p.slot_first + p.top_k);
    ok = same(got, want, "selection", line) && ok;
    if (has_out) {
      mask.resize(p.mask_words);
      ok = same(mask, want_mask, "candidate blocks", line) && ok;
    }
    ok = same(std::vector<uint32_t>{status}, {want_status}, "status", line) && ok;
  }
  return ok;
}

int vectors(const char* path) {
  std::ifstream in(path);
  if (!in) die(std::string("cannot read ") + path);
  std::string text;
  size_t line = 0, failed = 0;
  size_t counts[4] = {0, 0, 0, 0};
  while (std::getline(in, text)) {
    ++line;
    if (text.empty() || text[0] == '#') continue;
    Line l;
    std::istringstream words(text);
    for (std::string w; words >> w;) l.tokens.push_back(w);
    const std::string kind = l.next();
    bool ok;
    if (kind == "attn") {
      ok = check_attn(l, line), ++counts[0];
    } else if (kind == "win") {
      ok = check_window(l, line), ++counts[1];
    } else if (kind == "score") {
      ok = check_score(l, line), ++counts[2];
    } else if (kind == "select") {
      ok = check_select(l, line), ++counts[3];
    } else {
      die("unknown record " + kind);
    }
    if (l.at != l.tokens.size()) die("long record at line " + std::to_string(line));
    failed += !ok;
  }
  std::cout << "attn " << counts[0] << " win " << counts[1] << " score " << counts[2] << " select "
            << counts[3] << " failed " << failed << "\n";
  return failed == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "vectors") return vectors(argv[2]);
  if (argc == 4 && std::string(argv[1]) == "jobs") return jobs(argv[2], argv[3]);
  std::cerr << "usage: ds_attention_host vectors <file> | jobs <in> <out>\n";
  return 2;
}
