//! A device suite's checks, each run to its end whatever the others did, and
//! the JSON summary an unattended run leaves behind.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::Instant;

use eidola_engine_cuda::Gpu;
use serde_json::{Value, json};

use super::device::{How, Kernels, Prepared};
use super::{Case, Regime, Rng, build};

pub struct Summary {
    name: &'static str,
    device: Value,
    checks: Vec<Value>,
    bench: Vec<Value>,
    failed: usize,
}

impl Summary {
    pub fn new(name: &'static str, gpu: &Gpu) -> Summary {
        let info = gpu.info();
        Summary {
            name,
            device: json!({
                "name": info.name,
                "compute_capability": [info.compute_capability.0, info.compute_capability.1],
                "sm_count": info.sm_count,
                "driver_version": info.driver_version,
            }),
            checks: Vec::new(),
            bench: Vec::new(),
            failed: 0,
        }
    }

    /// Run one check; a panic is recorded as its failure and the suite goes on.
    pub fn check(&mut self, name: impl Into<String>, body: impl FnOnce()) {
        let name = name.into();
        let start = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(body));
        let seconds = start.elapsed().as_secs_f64();
        let error = outcome.err().map(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "panic".to_owned())
        });
        match &error {
            None => eprintln!("ok      {name} ({seconds:.1} s)"),
            Some(e) => {
                self.failed += 1;
                eprintln!("FAILED  {name}: {e}");
            }
        }
        self.checks.push(
            json!({"check": name, "ok": error.is_none(), "error": error, "seconds": seconds}),
        );
    }

    /// A kernel at the released model's largest shapes: every output bit
    /// against the reference, then `iters` launches timed back to back.
    pub fn released(&mut self, gpu: &Gpu, k: &Kernels, kernel: &'static str, iters: u32) {
        let mut row = None;
        self.check(format!("released/{kernel}"), || {
            let case = build(kernel, Regime::Released, &mut Rng(0xbe9c));
            super::device::check(gpu, k, &case, How::default(), "released");
            row = Some(time(gpu, k, &case, iters));
        });
        if let Some(row) = row {
            eprintln!("bench   {row}");
            self.bench.push(row);
        }
    }

    /// Write `<name>.json` under `EIDOLA_DS_SUMMARY_DIR` (else the test
    /// target's temporary directory), then fail if any check did.
    pub fn finish(self) {
        let dir = std::env::var_os("EIDOLA_DS_SUMMARY_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.json", self.name));
        let summary = json!({
            "suite": self.name,
            "device": self.device,
            "passed": self.checks.len() - self.failed,
            "failed": self.failed,
            "checks": self.checks,
            "bench": self.bench,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        eprintln!(
            "{}: {} checks, {} failed; summary in {}",
            self.name,
            self.checks.len(),
            self.failed,
            path.display()
        );
        assert_eq!(self.failed, 0, "{} checks failed", self.failed);
    }
}

/// Device time of one launch of `case`: `iters` launches enqueued back to
/// back, timed from the first to the stream's end.
fn time(gpu: &Gpu, k: &Kernels, case: &Case, iters: u32) -> Value {
    let prepared = Prepared::new(gpu, case);
    for _ in 0..3 {
        prepared.launch(gpu, k, How::default()).unwrap();
    }
    gpu.synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        prepared.launch(gpu, k, How::default()).unwrap();
    }
    gpu.synchronize().unwrap();
    let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    let rows = ["rows", "entries", "layout_rows"]
        .iter()
        .find_map(|name| case.n.get(name))
        .copied()
        .unwrap_or(1);
    json!({
        "kernel": case.kernel,
        "shape": case.n,
        "rows": rows,
        "us_per_launch": us,
        "us_per_row": us / f64::from(rows),
    })
}
