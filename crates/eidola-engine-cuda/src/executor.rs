//! [`CudaExecutor`]: the serving core's [`Executor`] on one GPU.
//!
//! A step runs the seam's sequence exactly (`eidola-engine/src/executor.rs`):
//! maintenance in order, table updates in order, the target forward over every
//! host token (KV written through the tables), then sampling on the device with
//! the core's sampling semantics over the sampleable vocabulary. Speculative
//! decoding is not implemented yet: the executor reports no draft width, and a
//! step asking for drafts is a host bug (as is one asking for drafts at
//! position 0, which the seam forbids outright).
//!
//! Contract checks mirror the CPU reference executor's: a KV write into a block
//! mapped by more than one table entry, or a read through an unmapped entry,
//! panics before anything is launched.

use std::sync::Arc;

use crate::module::KernelDir;
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::sampling::Stream;
use eidola_engine::spec::{AttentionKind, Bucket, KvGroupSpec, KvRole, ModelSpec};
use eidola_engine_model::config::AttentionKind as ModelAttention;
use eidola_engine_model::safetensors::WeightSet;

use crate::attention::{AttnPlan, AttnRequest};
use crate::device::ImageArch;
use crate::kv::KvStore;
use crate::model::{ForwardInput, GpuModel, Kernels, group_geometry, group_layers, load_weights};
use crate::sampler::{STATUS_NON_FINITE, SampleRow};
use crate::{CudaError, Gpu, Result};

/// Executor configuration: memory geometry and step limits.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaExecutorConfig {
    /// Positions per KV block (every group).
    pub block_size: u32,
    /// Physical blocks per KV group (global first, then sliding, in the order
    /// the model's layers first use them), each including the null block 0.
    pub num_blocks: Vec<u32>,
    pub num_state_slots: u32,
    pub max_model_len: u32,
    /// Captured step shapes; the last bounds every step.
    pub buckets: Vec<Bucket>,
    /// The tokenizer's vocabulary size (never taken from the weights).
    pub sampleable_vocab_size: u32,
    /// Kernel image to run (the device's own when `None`; `Sm100f` runs the
    /// family image on any CC 10.x part).
    pub image: Option<ImageArch>,
}

pub struct CudaExecutor {
    gpu: Gpu,
    spec: ModelSpec,
    cfg: CudaExecutorConfig,
    kv: KvStore,
    model: GpuModel,
    // Sampling scratch.
    sample_rows: cudarc::driver::CudaSlice<SampleRow>,
    probs: cudarc::driver::CudaSlice<f64>,
    tokens: cudarc::driver::CudaSlice<u32>,
    status: cudarc::driver::CudaSlice<u32>,
    steps: u64,
    /// Testing aid: compute logits at every host token of every step and keep
    /// the last step's (`[tokens, vocab]`, see [`CudaExecutor::take_all_logits`]).
    pub record_all_logits: bool,
    all_logits: Option<Vec<f32>>,
}

impl std::fmt::Debug for CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaExecutor")
            .field("spec", &self.spec)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

impl CudaExecutor {
    /// Load the model (optionally only `keep_layers` of the checkpoint) and
    /// allocate every pool and buffer.
    pub fn new(
        gpu: Gpu,
        kernels_dir: &KernelDir,
        store: Arc<WeightSet>,
        keep_layers: Option<&[usize]>,
        cfg: CudaExecutorConfig,
    ) -> Result<CudaExecutor> {
        let weights = load_weights(&gpu, store, keep_layers)?;
        crate::support::check_sampleable(
            cfg.sampleable_vocab_size as usize,
            weights.config.vocab_size,
        )?;
        let kernels = Kernels::load(&gpu, kernels_dir, cfg.image)?;
        let config = weights.config.clone();
        let (keys, layer_kv) = group_layers(&config);
        if cfg.num_blocks.len() != keys.len() {
            return Err(CudaError::new(format!(
                "{} block counts for {} KV groups",
                cfg.num_blocks.len(),
                keys.len()
            )));
        }
        let geometry = group_geometry(&keys, &layer_kv, cfg.block_size, &cfg.num_blocks);
        let kv_groups = keys
            .iter()
            .enumerate()
            .map(|(g, a)| {
                let attention = match a.kind {
                    ModelAttention::Global => AttentionKind::Full,
                    ModelAttention::Sliding { window } => AttentionKind::Sliding {
                        window: window as u32,
                    },
                };
                KvGroupSpec {
                    name: match attention {
                        AttentionKind::Full => "global".into(),
                        AttentionKind::Sliding { window } => format!("sliding-{window}"),
                    },
                    role: KvRole::Target,
                    attention,
                    num_layers: geometry[g].num_layers,
                    num_kv_heads: geometry[g].num_kv_heads,
                    head_dim_qk: geometry[g].head_dim_qk,
                    head_dim_v: geometry[g].head_dim_v,
                    num_blocks: geometry[g].num_blocks,
                }
            })
            .collect();
        let spec = ModelSpec {
            vocab_size: config.vocab_size as u32,
            sampleable_vocab_size: cfg.sampleable_vocab_size,
            block_size: cfg.block_size,
            max_model_len: cfg.max_model_len,
            kv_groups,
            max_draft_tokens: 0,
            num_state_slots: cfg.num_state_slots,
            buckets: cfg.buckets.clone(),
        };
        spec.validate().map_err(CudaError::new)?;
        let last = *spec.buckets.last().expect("validated: a bucket");
        let max_tokens = last.max_tokens as usize;
        let max_rows = last.max_seqs as usize;
        let kv = KvStore::new(
            &gpu,
            geometry,
            cfg.num_state_slots,
            spec.max_blocks_per_seq(),
            0,
        )?;
        let model = GpuModel::new(
            &gpu,
            weights,
            kernels,
            layer_kv,
            max_tokens,
            max_tokens,
            cfg.max_model_len as usize,
        )?;
        let s = gpu.stream();
        let v = cfg.sampleable_vocab_size as usize;
        Ok(CudaExecutor {
            sample_rows: s.alloc_zeros(max_rows.max(1))?,
            probs: s.alloc_zeros(max_rows.max(1) * v)?,
            tokens: s.alloc_zeros(max_rows.max(1))?,
            status: s.alloc_zeros(1)?,
            gpu,
            spec,
            cfg,
            kv,
            model,
            steps: 0,
            record_all_logits: false,
            all_logits: None,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn kv(&self) -> &KvStore {
        &self.kv
    }

    pub fn model(&self) -> &GpuModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut GpuModel {
        &mut self.model
    }

    pub fn config(&self) -> &CudaExecutorConfig {
        &self.cfg
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The logits of every host token of the last step, when
    /// [`CudaExecutor::record_all_logits`] is set.
    pub fn take_all_logits(&mut self) -> Option<Vec<f32>> {
        self.all_logits.take()
    }

    /// Whether every byte of a block is zero (copied back from the device).
    pub fn block_is_zero(&self, group: u32, block: u32) -> Result<bool> {
        Ok(self
            .kv
            .read_block(&self.gpu, group as usize, block)?
            .iter()
            .all(|&x| x == 0))
    }

    fn step(&mut self, step: &StepInput) -> Result<StepOutput> {
        if step.seqs.len() as u32 > step.bucket.max_seqs
            || step.query_tokens() > step.bucket.max_tokens
        {
            return Err(CudaError::new("batch exceeds its bucket"));
        }
        // 1-2. Maintenance, then table updates.
        self.kv
            .apply(&self.gpu, &step.maintenance, &step.table_updates)?;

        // Validate rows; gather token-level inputs, KV targets and plans.
        let groups = self.spec.kv_groups.clone();
        let bs = self.spec.block_size;
        let mut kv_targets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); groups.len()];
        let mut requests: Vec<Vec<AttnRequest>> = vec![Vec::new(); groups.len()];
        let mut logit_rows = Vec::new();
        let mut q_start = 0u32;
        for e in &step.seqs {
            assert!(e.num_tokens >= 1, "a row needs at least one host token");
            let start = e.token_start as usize;
            let end = start + e.num_tokens as usize;
            assert_eq!(
                start as u32, q_start,
                "rows' host tokens must be contiguous in row order"
            );
            for (i, &pos) in step.positions[start..end].iter().enumerate() {
                assert_eq!(
                    pos,
                    e.context_len + i as u32,
                    "positions disagree with context_len"
                );
            }
            let p = e.context_len + e.num_tokens - 1;
            let k = if e.sample { e.num_drafts } else { 0 };
            // No drafter row exists for position 0, and the seam forbids asking.
            assert!(
                p > 0 || k == 0,
                "slot {}: {k} drafts requested at position 0",
                e.slot
            );
            assert!(k == 0, "{k} drafts requested; this executor drafts none");
            assert!(p < self.spec.max_model_len, "row past the model length");
            for (g, group) in groups.iter().enumerate() {
                for pos in e.context_len..=p {
                    let block = self.kv.writable_block(e.slot, g, pos);
                    kv_targets[g].push((block, pos % bs));
                }
                // Visible KV of the row: from the first position its first
                // query sees, whole pages, through p.
                let first = group.attention.first_visible(e.context_len);
                let first_page = first / bs;
                let pages: Vec<u32> = (first_page..=p / bs)
                    .map(|i| self.kv.block_of(e.slot, g, i * bs))
                    .collect();
                requests[g].push(AttnRequest {
                    q_start,
                    qo_len: e.num_tokens,
                    pages,
                    kv_len: p + 1 - first_page * bs,
                });
            }
            if e.sample && !self.record_all_logits {
                logit_rows.push(q_start + e.num_tokens - 1);
            }
            q_start += e.num_tokens;
        }
        let total = q_start as usize;
        if self.record_all_logits {
            logit_rows = (0..q_start).collect();
        }
        assert_eq!(
            total,
            step.token_ids.len(),
            "token_ids holds every row's host tokens"
        );
        let vocab = self.spec.vocab_size;
        for &t in &step.token_ids {
            if t >= vocab {
                return Err(CudaError::new(format!("token {t} outside vocab {vocab}")));
            }
        }
        let mut plans: Vec<AttnPlan> = Vec::with_capacity(groups.len());
        for (g, group) in groups.iter().enumerate() {
            let nq = self
                .model
                .config()
                .layers
                .iter()
                .zip(&self.model.layer_kv)
                .find(|(_, l)| l.group == g)
                .map(|(l, _)| l.attention.num_q_heads as u32)
                .expect("a layer per group");
            plans.push(self.model.kernels.attention.plan(
                &self.gpu,
                &requests[g],
                nq / group.num_kv_heads,
                bs,
            )?);
        }

        // 3. The forward.
        self.model.forward(
            &self.gpu,
            &self.kv,
            &ForwardInput {
                tokens: &step.token_ids,
                positions: &step.positions,
                kv_targets: &kv_targets,
                plans: &plans,
                logit_rows: &logit_rows,
            },
        )?;

        // 4. Sampling over the sampleable vocabulary.
        let s = self.gpu.stream().clone();
        let mut sample_rows = Vec::new();
        let mut which = Vec::new();
        for (i, e) in step.seqs.iter().enumerate() {
            if e.sample {
                let p = e.context_len + e.num_tokens - 1;
                let logit_row = if self.record_all_logits {
                    e.token_start + e.num_tokens - 1
                } else {
                    sample_rows.len() as u32
                };
                sample_rows.push(SampleRow::new(&e.sampling, p + 1, logit_row));
                which.push(i);
            }
        }
        if self.record_all_logits {
            let n = total * vocab as usize;
            self.all_logits = Some(s.clone_dtoh(&self.model.logits().slice(..n))?);
        }
        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; step.seqs.len() * stride as usize],
            stride,
            num_tokens: vec![0; step.seqs.len()],
            logits: step.return_logits.then(Vec::new),
        };
        if !sample_rows.is_empty() {
            let n = sample_rows.len();
            s.memcpy_htod(&sample_rows, &mut self.sample_rows.slice_mut(..n))?;
            s.memset_zeros(&mut self.status)?;
            self.model.kernels.sampler.sample(
                &self.gpu,
                self.model.logits(),
                vocab as usize,
                self.spec.sampleable_vocab_size,
                &self.sample_rows,
                n as u32,
                Some(Stream::Sample),
                &mut self.probs,
                &mut self.tokens,
                &mut self.status,
            )?;
            let tokens = s.clone_dtoh(&self.tokens.slice(..n))?;
            let status = s.clone_dtoh(&self.status)?[0];
            if status & STATUS_NON_FINITE != 0 {
                return Err(CudaError::new("non-finite logits"));
            }
            let logits = if step.return_logits {
                let rows: Vec<usize> = sample_rows.iter().map(|r| r.logit_row as usize).collect();
                let all = s.clone_dtoh(
                    &self
                        .model
                        .logits()
                        .slice(..(rows.iter().max().unwrap() + 1) * vocab as usize),
                )?;
                let v = vocab as usize;
                Some(
                    rows.iter()
                        .flat_map(|&r| all[r * v..(r + 1) * v].to_vec())
                        .collect::<Vec<f32>>(),
                )
            } else {
                None
            };
            for (j, &i) in which.iter().enumerate() {
                out.tokens[i * stride as usize] = tokens[j];
                out.num_tokens[i] = 1;
            }
            if let (Some(all), Some(rows)) = (logits, out.logits.as_mut()) {
                let mut j = 0;
                for e in &step.seqs {
                    if e.sample {
                        rows.push(vec![
                            all[j * vocab as usize..(j + 1) * vocab as usize].to_vec(),
                        ]);
                        j += 1;
                    } else {
                        rows.push(Vec::new());
                    }
                }
            }
        } else if let Some(rows) = out.logits.as_mut() {
            rows.resize(step.seqs.len(), Vec::new());
        }
        Ok(out)
    }
}

impl Executor for CudaExecutor {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> std::result::Result<StepOutput, ExecutorError> {
        self.steps += 1;
        self.step(step).map_err(|e| ExecutorError(e.to_string()))
    }
}
