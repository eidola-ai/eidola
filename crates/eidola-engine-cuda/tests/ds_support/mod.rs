//! Shared by the DeepSeek kernel tests: a case as one buffer holding every
//! array its launches name, the launches' parameter structs addressed into it,
//! and the bytes the reference expects where the kernels write. A job runs
//! unchanged on a device (the buffer uploaded, the addresses moved to it) and
//! on the host, through `eidola-engine-kernels`' `oracle/ds_attention_host.cc`:
//! the kernel bodies compiled with the system C++ compiler and run block by
//! block, in ascending and descending thread order. The host run needs no
//! device, so everything of a GPU test but the device's own code runs in a
//! plain `cargo test`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use eidola_engine_cuda::ds::abi::{
    AttnParams, IndexScoreParams, KvPlane, SelectParams, WindowSlotsParams, kv_code_bytes,
    kv_scale_bytes,
};
use eidola_engine_cuda::ds::attention::{self, DsAttention};
use eidola_engine_cuda::ds::index::DsIndex;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{Gpu, ImageArch, KernelDir, KernelModule};

#[path = "../../../eidola-engine-model/tests/deepseek_attention_cases/mod.rs"]
pub mod cases;

/// The address a job's parameters give its buffer's first byte.
pub const BASE: u64 = 1 << 40;
/// Bytes between arrays, never written.
const GAP: usize = 24;
const GAP_FILL: u8 = 0xc7;

/// One kernel launch.
#[derive(Clone, Copy, Debug)]
pub enum Call {
    /// With the heads per block.
    Attn(AttnParams, u32),
    Window(WindowSlotsParams),
    /// With the grid's width.
    Score(IndexScoreParams, u32),
    Select(SelectParams),
}

fn bytes_of<T: Copy>(x: &T) -> Vec<u8> {
    // SAFETY: the parameter structs are `repr(C)` without padding (their layouts are asserted
    // field by field), so every byte is initialised.
    unsafe { std::slice::from_raw_parts(x as *const T as *const u8, std::mem::size_of::<T>()) }
        .to_vec()
}

impl Call {
    fn kernel(&self) -> u32 {
        match self {
            Call::Attn(..) => 0,
            Call::Window(..) => 1,
            Call::Score(..) => 2,
            Call::Select(..) => 3,
        }
    }

    /// The grid the library launches it over.
    pub fn grid(&self) -> [u32; 3] {
        match self {
            Call::Attn(p, group) => attention::attn_grid(p, *group).unwrap(),
            Call::Window(p) => attention::window_slots_grid(p),
            Call::Score(p, width) => [*width, p.rows, 1],
            Call::Select(p) => [p.rows, 1, 1],
        }
    }

    fn params(&self) -> Vec<u8> {
        match self {
            Call::Attn(p, _) => bytes_of(p),
            Call::Window(p) => bytes_of(p),
            Call::Score(p, _) => bytes_of(p),
            Call::Select(p) => bytes_of(p),
        }
    }

    /// The launch with its arrays in a buffer at `base`.
    pub fn moved_to(&self, base: u64) -> Call {
        self.moved(BASE, base)
    }

    /// The same launch with every array at `to` where it was at `from`.
    fn moved(&self, from: u64, to: u64) -> Call {
        let at = |a: u64| if a == 0 { 0 } else { a - from + to };
        let plane = |k: KvPlane| KvPlane {
            base: at(k.base),
            ..k
        };
        match *self {
            Call::Attn(p, group) => Call::Attn(
                AttnParams {
                    q: at(p.q),
                    slots: at(p.slots),
                    sink: at(p.sink),
                    out: at(p.out),
                    a: plane(p.a),
                    b: plane(p.b),
                    ..p
                },
                group,
            ),
            Call::Window(p) => Call::Window(WindowSlotsParams {
                positions: at(p.positions),
                first: at(p.first),
                tables: at(p.tables),
                table_row: at(p.table_row),
                slots: at(p.slots),
                status: at(p.status),
                ..p
            }),
            Call::Score(p, width) => Call::Score(
                IndexScoreParams {
                    q_codes: at(p.q_codes),
                    q_scales: at(p.q_scales),
                    head_dots: at(p.head_dots),
                    tables: at(p.tables),
                    table_row: at(p.table_row),
                    visible: at(p.visible),
                    scores: at(p.scores),
                    status: at(p.status),
                    keys: plane(p.keys),
                    ..p
                },
                width,
            ),
            Call::Select(p) => Call::Select(SelectParams {
                scores: at(p.scores),
                visible: at(p.visible),
                mask_in: at(p.mask_in),
                mask_out: at(p.mask_out),
                tables: at(p.tables),
                table_row: at(p.table_row),
                slots: at(p.slots),
                status: at(p.status),
                ..p
            }),
        }
    }
}

/// The buffer of a job under construction.
#[derive(Default)]
pub struct Arena {
    pub bytes: Vec<u8>,
}

pub trait Word: Copy {
    fn put(self, out: &mut Vec<u8>);
}
impl Word for u8 {
    fn put(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}
impl Word for u16 {
    fn put(self, out: &mut Vec<u8>) {
        out.extend(self.to_le_bytes());
    }
}
impl Word for u32 {
    fn put(self, out: &mut Vec<u8>) {
        out.extend(self.to_le_bytes());
    }
}
impl Word for i32 {
    fn put(self, out: &mut Vec<u8>) {
        out.extend(self.to_le_bytes());
    }
}
impl Word for f32 {
    fn put(self, out: &mut Vec<u8>) {
        out.extend(self.to_bits().to_le_bytes());
    }
}

pub fn le_bytes<T: Word>(data: &[T]) -> Vec<u8> {
    let mut out = Vec::new();
    for x in data {
        x.put(&mut out);
    }
    out
}

impl Arena {
    /// Appends `data` behind a gap, at a multiple of 8, and returns its offset.
    pub fn put<T: Word>(&mut self, data: &[T]) -> usize {
        self.put_bytes(&le_bytes(data), 0)
    }

    /// Appends bytes behind a gap, `skew` bytes past a multiple of 8.
    pub fn put_bytes(&mut self, data: &[u8], skew: usize) -> usize {
        let at = (self.bytes.len() + GAP).next_multiple_of(8) + skew;
        self.bytes.resize(at, GAP_FILL);
        self.bytes.extend_from_slice(data);
        at
    }

    pub fn addr(offset: usize) -> u64 {
        BASE + offset as u64
    }
}

/// Under `EIDOLA_DS_DIAG`, a job that differs prints every differing output (how many bytes,
/// where, the first of them) and the run goes on to the other jobs before it fails.
static DIAG_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bytes a job must leave at an offset.
pub struct Output {
    pub what: String,
    pub at: usize,
    pub want: Vec<u8>,
}

/// Launches over one buffer, and what they must leave in it.
pub struct Job {
    pub label: String,
    pub calls: Vec<Call>,
    pub arena: Vec<u8>,
    pub outputs: Vec<Output>,
}

impl Job {
    pub fn new(label: impl Into<String>, mut arena: Arena) -> Job {
        // A trailing gap, and whole 8-byte words.
        let end = (arena.bytes.len() + GAP).next_multiple_of(8);
        arena.bytes.resize(end, GAP_FILL);
        Job {
            label: label.into(),
            calls: Vec::new(),
            arena: arena.bytes,
            outputs: Vec::new(),
        }
    }

    pub fn expect<T: Word>(&mut self, what: impl Into<String>, at: usize, want: &[T]) {
        self.outputs.push(Output {
            what: what.into(),
            at,
            want: le_bytes(want),
        });
    }

    /// Requires `got` (the buffer after the launches) to hold every expected
    /// output and, everywhere else, exactly what the buffer held before.
    pub fn verify(&self, got: &[u8], run: &str) {
        assert_eq!(got.len(), self.arena.len(), "{} ({run})", self.label);
        let mut written = vec![false; got.len()];
        if std::env::var_os("EIDOLA_DS_DIAG").is_some() {
            let mut failed = false;
            for o in &self.outputs {
                let bad: Vec<usize> = (0..o.want.len())
                    .filter(|i| got[o.at + i] != o.want[*i])
                    .collect();
                if !bad.is_empty() {
                    failed = true;
                    let lo = bad[0] - bad[0] % 16;
                    let hi = (lo + 24).min(o.want.len());
                    eprintln!(
                        "DIAG {} ({run}): {}: {} of {} bytes differ, first {} last {} got {:02x?} want {:02x?}",
                        self.label,
                        o.what,
                        bad.len(),
                        o.want.len(),
                        bad[0],
                        bad[bad.len() - 1],
                        &got[o.at + lo..o.at + hi],
                        &o.want[lo..hi]
                    );
                }
            }
            if failed {
                DIAG_FAILED.store(true, Ordering::Relaxed);
            } else {
                eprintln!("DIAG ok {} ({run})", self.label);
            }
            return;
        }
        for o in &self.outputs {
            let range = o.at..o.at + o.want.len();
            if let Some(i) = (0..o.want.len()).find(|i| got[o.at + i] != o.want[*i]) {
                let window = |b: &[u8]| b[i - i % 4..(i - i % 4 + 8).min(b.len())].to_vec();
                panic!(
                    "{} ({run}): {} differs at byte {i} of {}: got {:02x?}, want {:02x?}",
                    self.label,
                    o.what,
                    o.want.len(),
                    window(&got[range.clone()]),
                    window(&o.want)
                );
            }
            written[range].fill(true);
        }
        if let Some(i) = (0..got.len()).find(|i| !written[*i] && got[*i] != self.arena[*i]) {
            panic!(
                "{} ({run}): byte {i} of the buffer changed outside every output",
                self.label
            );
        }
    }
}

/// A pool of packed entries: blocks of `per_block` entries, each block its
/// codes, a gap, its scales, and a tail; the plane is addressed off an odd
/// byte, as nothing aligns a pool's planes.
pub fn pool(
    arena: &mut Arena,
    format: u32,
    width: usize,
    per_block: usize,
    entries: &[Vec<u8>],
) -> KvPlane {
    let w = u32::try_from(width).unwrap();
    let cb = kv_code_bytes(format, w) as usize;
    let sb = kv_scale_bytes(format, w) as usize;
    let block_bytes = 5 + per_block * (cb + sb) + 11;
    let blocks = entries.len().div_ceil(per_block).max(1);
    let mut bytes = vec![0x99u8; blocks * block_bytes];
    for (e, entry) in entries.iter().enumerate() {
        assert_eq!(entry.len(), cb + sb);
        let block = (e / per_block) * block_bytes;
        let o = e % per_block;
        bytes[block + 5 + o * cb..][..cb].copy_from_slice(&entry[..cb]);
        bytes[block + 5 + per_block * cb + 3 + o * sb..][..sb].copy_from_slice(&entry[cb..]);
    }
    let at = arena.put_bytes(&bytes, 1);
    KvPlane {
        base: Arena::addr(at),
        block_bytes: block_bytes as u64,
        codes_offset: 5,
        scales_offset: (5 + per_block * cb + 3) as u64,
        entries_per_block: u32::try_from(per_block).unwrap(),
        format,
        width: w,
        reserved: 0,
    }
}

// ---- the host run ----

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../eidola-engine-kernels")
}

/// The host check's binary, built once per test process; `None` (after saying
/// so) when there is no C++ compiler.
fn host_binary() -> Option<&'static PathBuf> {
    static BINARY: OnceLock<Option<PathBuf>> = OnceLock::new();
    BINARY
        .get_or_init(|| {
            let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
            if Command::new(&cxx).arg("--version").output().is_err() {
                eprintln!("no C++ compiler ({cxx}): the kernel bodies did not run on the host");
                return None;
            }
            let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
                .join(format!("ds_attention_host_{}", std::process::id()));
            let status = Command::new(&cxx)
                .args(["-std=c++17", "-O2", "-ffp-contract=off", "-I"])
                .arg(kernels_dir().join("csrc"))
                .arg(kernels_dir().join("oracle/ds_attention_host.cc"))
                .arg("-o")
                .arg(&out)
                .status()
                .expect("run the C++ compiler");
            assert!(
                status.success(),
                "{cxx} failed on oracle/ds_attention_host.cc"
            );
            Some(out)
        })
        .as_ref()
}

/// Whether the host run is available.
pub fn host_available() -> bool {
    host_binary().is_some()
}

/// Runs every job on the host, in both thread orders, and verifies each.
/// Returns `false` without a C++ compiler.
pub fn run_on_host(jobs: &[Job]) -> bool {
    let Some(binary) = host_binary() else {
        return false;
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let stem = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "ds_jobs_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let (input, output) = (stem.with_extension("in"), stem.with_extension("out"));
    let mut file = Vec::new();
    for job in jobs {
        file.extend(u32::try_from(job.calls.len()).unwrap().to_le_bytes());
        file.extend(0u32.to_le_bytes());
        file.extend((job.arena.len() as u64).to_le_bytes());
        file.extend(BASE.to_le_bytes());
        for call in &job.calls {
            let params = call.params();
            let grid = call.grid();
            for word in [
                call.kernel(),
                grid[0],
                grid[1],
                u32::try_from(params.len()).unwrap(),
            ] {
                file.extend(word.to_le_bytes());
            }
            file.extend(&params);
            file.resize(file.len().next_multiple_of(8), 0);
        }
        file.extend(&job.arena);
    }
    std::fs::write(&input, &file).unwrap();
    let ran = Command::new(binary)
        .arg("jobs")
        .arg(&input)
        .arg(&output)
        .output()
        .expect("run the host check");
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let got = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    let mut at = 0;
    for job in jobs {
        for run in ["host, ascending threads", "host, descending threads"] {
            job.verify(&got[at..at + job.arena.len()], run);
            at += job.arena.len();
        }
    }
    assert_eq!(at, got.len());
    true
}

// ---- the device run ----

/// The kernels of one image.
pub struct Ops {
    pub arch: ImageArch,
    pub attention: DsAttention,
    pub index: DsIndex,
}

impl Ops {
    pub fn load(gpu: &Gpu, dir: &KernelDir, arch: ImageArch) -> Ops {
        let module = |name: &str| {
            KernelModule::load_from(gpu, dir, name, eidola_engine_cuda::ImageSource::Cubin(arch))
                .unwrap_or_else(|e| panic!("{name} {arch:?}: {e}"))
        };
        Ops {
            arch,
            attention: DsAttention::from_module(module(DsAttention::IMAGE)).unwrap(),
            index: DsIndex::from_module(module(DsIndex::IMAGE)).unwrap(),
        }
    }

    /// Launches `call` through the library's wrapper.
    ///
    /// # Safety
    ///
    /// The call's arrays must be live device memory holding what its
    /// parameters say.
    pub unsafe fn launch(&self, gpu: &Gpu, call: &Call) {
        // SAFETY: the caller's contract.
        unsafe {
            match call {
                Call::Attn(p, group) => self.attention.attend_grouped(gpu, p, *group),
                Call::Window(p) => self.attention.window_slots(gpu, p),
                Call::Score(p, width) => self.index.score_on(gpu, p, *width),
                Call::Select(p) => self.index.select(gpu, p),
            }
        }
        .unwrap_or_else(|e| panic!("{call:?}: {e}"));
    }
}

/// Runs every job on the device with one image's kernels and verifies each.
pub fn run_on_device(gpu: &Gpu, ops: &Ops, jobs: &[Job]) {
    let s = gpu.stream();
    for job in jobs {
        let buffer = s.clone_htod(&job.arena).unwrap();
        let base = dptr(&buffer, s);
        for call in &job.calls {
            // SAFETY: every array of the call lies in `buffer` (the job's arena, moved),
            // which lives until the read-back below has synchronized.
            unsafe { ops.launch(gpu, &call.moved_to(base)) };
        }
        let got = s.clone_dtoh(&buffer).unwrap();
        job.verify(&got, &format!("device, {:?}", ops.arch));
    }
    assert!(
        !DIAG_FAILED.load(Ordering::Relaxed),
        "jobs differed (listed above)"
    );
}

// ---- timing ----

/// Microseconds per launch of `launch`, run `iters` times after one warm-up.
pub fn time_us(gpu: &Gpu, iters: usize, launch: &dyn Fn()) -> f64 {
    use cudarc::driver::sys::CUevent_flags;
    launch();
    gpu.synchronize().unwrap();
    let ctx = gpu.context();
    let start = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    start.record(gpu.stream()).unwrap();
    for _ in 0..iters {
        launch();
    }
    end.record(gpu.stream()).unwrap();
    let ms = start.elapsed_ms(&end).unwrap();
    f64::from(ms) * 1000.0 / iters as f64
}

/// One timing of a bench: a kernel at a shape and a row count.
pub struct Timing {
    pub kernel: &'static str,
    pub shape: String,
    pub rows: u32,
    pub launch_us: f64,
}

/// Writes a bench's timings as JSON to `$EIDOLA_DS_BENCH_DIR/<name>.json`
/// (the test's temporary directory when unset), prints them, and returns the
/// path.
pub fn write_bench(name: &str, arch: ImageArch, timings: &[Timing]) -> PathBuf {
    let dir = std::env::var_os("EIDOLA_DS_BENCH_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
    let rows: Vec<serde_json::Value> = timings
        .iter()
        .map(|t| {
            println!(
                "{:<24} {:<44} rows {:>4}  {:>12.1} us/launch  {:>10.2} us/query",
                t.kernel,
                t.shape,
                t.rows,
                t.launch_us,
                t.launch_us / f64::from(t.rows)
            );
            serde_json::json!({
                "kernel": t.kernel,
                "shape": t.shape,
                "queries": t.rows,
                "us_per_launch": t.launch_us,
                "us_per_query": t.launch_us / f64::from(t.rows),
            })
        })
        .collect();
    let path = dir.join(format!("{name}.json"));
    let doc = serde_json::json!({ "bench": name, "image": format!("{arch:?}"), "timings": rows });
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("bench summary: {}", path.display());
    path
}
