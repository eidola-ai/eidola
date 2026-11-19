//! The mock's recurrent state: a model whose layers keep, per sequence, a state of fixed
//! size beside its full-attention KV (the Qwen3.8 shape, in hashes).
//!
//! The state at position `t` (after positions `0 .. t`) is an order-sensitive fold of the
//! tokens and their positions, plus the last three tokens as the convolution tail. The
//! target's logits at `q` mix the KV the query sees with the state at `q + 1`, so a
//! skipped, doubled, reordered or rejected token changes every later output.
//!
//! A stored image is five words: the state, the tail's three tokens (each plus one, zero
//! before position 0) and a tag, the stored position plus one. The executor fails a step
//! that reads a base whose tag is not the row's `base_len`, a base that is zero or
//! encrypted, a write into a buffer that holds a state (unless it is the row's own base,
//! for its last write), a fold over positions the slot never kept, a tail past
//! `tail_positions`, a draft reaching a stored state, and every crypt and host-tier misuse
//! it fails for KV blocks.

use super::{MockConfig, fingerprint};
use crate::executor::{ExecutorError, StateRow, StateWrite};
use crate::kv_cipher::{KvKeystream, KvNonce};
use crate::sampling::mix64;
use crate::secret::KvKey;
use crate::spec::{ModelSpec, NULL_STATE};

/// Words in a stored image.
pub const IMAGE_WORDS: usize = 5;

/// A stored image: `[state, tail token 0 + 1, tail token 1 + 1, tail token 2 + 1, len + 1]`.
pub type Image = [u64; IMAGE_WORDS];

/// The state at some position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecState {
    /// The fold.
    pub word: u64,
    /// The last three tokens, oldest first, each plus one (0 before position 0).
    pub conv: [u64; 3],
}

impl RecState {
    /// The state at position 0.
    pub fn start(cfg: &MockConfig) -> Self {
        Self {
            word: mix64(cfg.model_seed ^ 0x5157_454e_5354_4154) | 1,
            conv: [0; 3],
        }
    }

    /// The state after `token` at `pos` runs.
    pub fn step(self, cfg: &MockConfig, token: u32, pos: u32) -> Self {
        let input = mix64(cfg.model_seed ^ (u64::from(token) << 32) ^ u64::from(pos));
        let conv_in = self.conv[0] ^ (self.conv[1] << 21) ^ (self.conv[2] << 42);
        Self {
            word: mix64(self.word ^ input ^ mix64(conv_in)) | 1,
            conv: [self.conv[1], self.conv[2], u64::from(token) + 1],
        }
    }

    /// What the state contributes to the logits.
    pub fn output(&self) -> u64 {
        mix64(self.word ^ self.conv[2].rotate_left(17))
    }

    /// The image a buffer holding this state at `len` stores.
    pub fn image(&self, len: u32) -> Image {
        [
            self.word,
            self.conv[0],
            self.conv[1],
            self.conv[2],
            u64::from(len) + 1,
        ]
    }
}

/// The state at `tokens.len()`: every token folded from position 0.
pub fn recurrent_state(cfg: &MockConfig, tokens: &[u32]) -> RecState {
    tokens
        .iter()
        .enumerate()
        .fold(RecState::start(cfg), |s, (p, &t)| s.step(cfg, t, p as u32))
}

/// One slot's state between steps: its base and the tokens of the positions after it.
#[derive(Clone, Debug, Default)]
struct Slot {
    /// `(buffer, base_len)`, once a row named it.
    base: Option<(u32, u32)>,
    /// Tokens of positions `base_len ..`, drafts of the last row included.
    tail: Vec<u32>,
}

/// The state pool, its host tier and the per-slot tails.
#[derive(Debug)]
pub(super) struct Pool {
    /// The index maintenance names the pool by.
    pub(super) index: u32,
    tail_cap: u32,
    pub(super) buffers: Vec<Image>,
    pub(super) encrypted: Vec<Option<u64>>,
    pub(super) host: Vec<Image>,
    pub(super) host_encrypted: Vec<Option<u64>>,
    slots: Vec<Slot>,
    /// Every image stored by a row or by `SaveState`, in order.
    pub(super) stored: Vec<Image>,
}

/// One row's state work, prepared before its forward.
pub(super) struct RowState {
    /// The state at `pos`: after the fold and the host tokens run so far.
    pub(super) state: RecState,
    /// The row's writes, ascending.
    writes: Vec<StateWrite>,
    /// The next position to run.
    pos: u32,
    slot: u32,
    /// The base the row started from, `(buffer, base_len)`.
    base: (u32, u32),
    /// The tokens of positions `base_len .. pos`.
    tokens: Vec<u32>,
}

impl Pool {
    pub(super) fn new(spec: &ModelSpec) -> Option<Self> {
        let state = spec.state.as_ref()?;
        Some(Self {
            index: spec.state_pool().expect("a state pool"),
            tail_cap: state.tail_positions,
            buffers: vec![[0; IMAGE_WORDS]; state.num_buffers as usize],
            encrypted: vec![None; state.num_buffers as usize],
            host: vec![[0; IMAGE_WORDS]; state.host_buffers as usize],
            host_encrypted: vec![None; state.host_buffers as usize],
            slots: vec![Slot::default(); spec.num_state_slots as usize],
            stored: Vec::new(),
        })
    }

    pub(super) fn is_zero(&self, buffer: u32) -> bool {
        self.buffers[buffer as usize] == [0; IMAGE_WORDS]
    }

    pub(super) fn host_is_zero(&self, host: u32) -> bool {
        self.host[host as usize] == [0; IMAGE_WORDS]
    }

    /// Whether `slot` holds no base and no tail.
    pub(super) fn slot_is_clear(&self, slot: u32) -> bool {
        let s = &self.slots[slot as usize];
        s.base.is_none() && s.tail.is_empty()
    }

    pub(super) fn reset(&mut self, slot: u32) {
        self.slots[slot as usize] = Slot::default();
    }

    pub(super) fn zero(&mut self, buffer: u32) {
        self.buffers[buffer as usize] = [0; IMAGE_WORDS];
        self.encrypted[buffer as usize] = None;
    }

    pub(super) fn zero_host(&mut self, host: u32) {
        self.host[host as usize] = [0; IMAGE_WORDS];
        self.host_encrypted[host as usize] = None;
    }

    pub(super) fn crypt(
        &mut self,
        buffer: u32,
        key: &KvKey,
        nonce: &KvNonce,
        encrypt: bool,
    ) -> Result<(), ExecutorError> {
        let print = fingerprint(key);
        let state = &mut self.encrypted[buffer as usize];
        match (encrypt, *state) {
            (true, None) => *state = Some(print),
            (false, Some(p)) if p == print => *state = None,
            (false, Some(_)) => {
                return Err(ExecutorError(format!(
                    "state buffer {buffer}: decrypt under another key"
                )));
            }
            (true, Some(_)) => {
                return Err(ExecutorError(format!(
                    "state buffer {buffer}: encrypted twice"
                )));
            }
            (false, None) => {
                return Err(ExecutorError(format!(
                    "state buffer {buffer}: decrypt of a plaintext buffer"
                )));
            }
        }
        let image = &mut self.buffers[buffer as usize];
        KvKeystream::new(key, nonce).range(2 * IMAGE_WORDS, |i, k| {
            image[i / 2] ^= u64::from(k) << (32 * (i % 2));
        });
        Ok(())
    }

    pub(super) fn demote(&mut self, buffer: u32, host: u32) -> Result<(), ExecutorError> {
        let Some(print) = self.encrypted[buffer as usize] else {
            return Err(ExecutorError(format!(
                "state buffer {buffer}: demote of a plaintext buffer"
            )));
        };
        self.host[host as usize] = self.buffers[buffer as usize];
        self.host_encrypted[host as usize] = Some(print);
        Ok(())
    }

    pub(super) fn promote(&mut self, host: u32, buffer: u32) -> Result<(), ExecutorError> {
        let Some(print) = self.host_encrypted[host as usize] else {
            return Err(ExecutorError(format!(
                "state host buffer {host}: promote of a buffer holding no ciphertext"
            )));
        };
        if !self.is_zero(buffer) || self.encrypted[buffer as usize].is_some() {
            return Err(ExecutorError(format!(
                "state buffer {buffer}: promote into a buffer that is not zero"
            )));
        }
        self.buffers[buffer as usize] = self.host[host as usize];
        self.encrypted[buffer as usize] = Some(print);
        Ok(())
    }

    /// The state at `base_len` held by `base`, checked.
    fn read_base(
        &self,
        cfg: &MockConfig,
        base: u32,
        base_len: u32,
    ) -> Result<RecState, ExecutorError> {
        if base == NULL_STATE {
            if base_len != 0 {
                return Err(ExecutorError(format!(
                    "the zero state named as the state at {base_len}"
                )));
            }
            return Ok(RecState::start(cfg));
        }
        if self.encrypted[base as usize].is_some() {
            return Err(ExecutorError(format!(
                "state buffer {base}: read while encrypted"
            )));
        }
        let image = self.buffers[base as usize];
        if image[4] != u64::from(base_len) + 1 {
            return Err(ExecutorError(format!(
                "state buffer {base}: holds the state at {} (or none), read as the state at \
                 {base_len}",
                image[4].wrapping_sub(1)
            )));
        }
        Ok(RecState {
            word: image[0],
            conv: [image[1], image[2], image[3]],
        })
    }

    /// Checks a write target: a fresh (zero) buffer, or the row's own base.
    fn check_target(&self, buffer: u32, own_base: u32) -> Result<(), ExecutorError> {
        if buffer == NULL_STATE || buffer as usize >= self.buffers.len() {
            return Err(ExecutorError(format!("state write into buffer {buffer}")));
        }
        if buffer != own_base
            && (!self.is_zero(buffer) || self.encrypted[buffer as usize].is_some())
        {
            return Err(ExecutorError(format!(
                "state buffer {buffer}: written while it holds a state"
            )));
        }
        Ok(())
    }

    /// Prepares a row: checks its base against the slot, checks its writes, folds the
    /// tail up to `context_len` (storing what the writes name on the way) and returns the
    /// state there.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_row(
        &mut self,
        cfg: &MockConfig,
        block_size: u32,
        slot: u32,
        row: usize,
        context_len: u32,
        num_tokens: u32,
        state_row: StateRow,
        writes: Vec<StateWrite>,
        other_bases: &[u32],
    ) -> Result<RowState, ExecutorError> {
        let StateRow { base, base_len } = state_row;
        let s = &self.slots[slot as usize];
        match s.base {
            Some(b) if b != (base, base_len) => {
                return Err(ExecutorError(format!(
                    "slot {slot}: row names base {base}@{base_len}, the slot's is {}@{}",
                    b.0, b.1
                )));
            }
            None if context_len != base_len => {
                return Err(ExecutorError(format!(
                    "slot {slot}: a fresh slot starts at its base, not {context_len}"
                )));
            }
            _ => {}
        }
        if !base_len.is_multiple_of(block_size) || base_len > context_len {
            return Err(ExecutorError(format!(
                "row {row}: base at {base_len} is not a boundary at most {context_len}"
            )));
        }
        let kept = base_len + s.tail.len() as u32;
        if context_len > kept {
            return Err(ExecutorError(format!(
                "slot {slot}: fold to {context_len} over positions the slot never kept \
                 (it kept up to {kept})"
            )));
        }
        let end = context_len + num_tokens;
        for (i, w) in writes.iter().enumerate() {
            if w.len <= base_len || w.len > end || !w.len.is_multiple_of(block_size) {
                return Err(ExecutorError(format!(
                    "row {row}: state write at {} outside its committed range ({base_len}, {end}]",
                    w.len
                )));
            }
            if i > 0 && writes[i - 1].len >= w.len {
                return Err(ExecutorError(format!(
                    "row {row}: state writes out of order"
                )));
            }
            if other_bases.contains(&w.buffer) {
                return Err(ExecutorError(format!(
                    "state buffer {}: written while another row reads it",
                    w.buffer
                )));
            }
            if w.buffer == base && i + 1 != writes.len() {
                return Err(ExecutorError(format!(
                    "row {row}: its base overwritten before its last write"
                )));
            }
            self.check_target(w.buffer, base)?;
        }
        let state = self.read_base(cfg, base, base_len)?;
        let tail = s.tail[..(context_len - base_len) as usize].to_vec();
        let mut rs = RowState {
            state,
            writes,
            pos: base_len,
            slot,
            base: (base, base_len),
            tokens: Vec::new(),
        };
        for t in tail {
            self.advance(cfg, &mut rs, t);
        }
        Ok(rs)
    }

    /// Runs `token` at the row's next position, storing the state a write names there.
    /// Only host tokens and the fold come through here; drafts never do.
    pub(super) fn advance(&mut self, cfg: &MockConfig, rs: &mut RowState, token: u32) {
        rs.state = rs.state.step(cfg, token, rs.pos);
        rs.pos += 1;
        rs.tokens.push(token);
        if let Some(w) = rs.writes.iter().find(|w| w.len == rs.pos) {
            let image = rs.state.image(w.len);
            self.buffers[w.buffer as usize] = image;
            self.stored.push(image);
        }
    }

    /// Ends a row: the slot's base moves to its last write, and its tail keeps the tokens
    /// of every position after it, `drafts` included.
    pub(super) fn end_row(&mut self, rs: RowState, drafts: &[u32]) -> Result<(), ExecutorError> {
        let (mut base, mut base_len) = rs.base;
        let mut tokens = rs.tokens;
        tokens.extend_from_slice(drafts);
        if let Some(w) = rs.writes.last() {
            tokens.drain(..(w.len - base_len) as usize);
            base = w.buffer;
            base_len = w.len;
        }
        if tokens.len() as u32 > self.tail_cap {
            return Err(ExecutorError(format!(
                "slot {}: a tail of {} positions past {base_len}, more than {}",
                rs.slot,
                tokens.len(),
                self.tail_cap
            )));
        }
        self.slots[rs.slot as usize] = Slot {
            base: Some((base, base_len)),
            tail: tokens,
        };
        Ok(())
    }

    /// `Maintenance::SaveState`.
    pub(super) fn save(
        &mut self,
        cfg: &MockConfig,
        block_size: u32,
        slot: u32,
        len: u32,
        buffer: u32,
    ) -> Result<(), ExecutorError> {
        let s = self.slots[slot as usize].clone();
        let Some((base, base_len)) = s.base else {
            return Err(ExecutorError(format!(
                "slot {slot}: state saved from a slot that holds none"
            )));
        };
        if !len.is_multiple_of(block_size) || len < base_len || len > base_len + s.tail.len() as u32
        {
            return Err(ExecutorError(format!(
                "slot {slot}: state saved at {len}, outside {base_len} ..= {}",
                base_len + s.tail.len() as u32
            )));
        }
        self.check_target(buffer, base)?;
        let state = s.tail[..(len - base_len) as usize]
            .iter()
            .enumerate()
            .fold(self.read_base(cfg, base, base_len)?, |st, (i, &t)| {
                st.step(cfg, t, base_len + i as u32)
            });
        let image = state.image(len);
        self.buffers[buffer as usize] = image;
        self.stored.push(image);
        Ok(())
    }
}
