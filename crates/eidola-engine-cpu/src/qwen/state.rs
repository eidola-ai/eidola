//! The recurrent state's memory: the pool of state buffers (and its host tier), and each
//! slot's tail of inputs.
//!
//! **A buffer** holds the state at one block boundary for every DeltaNet layer: per layer,
//! in layer order, its convolution history (the last `conv_kernel - 1` raw `mixed` rows,
//! oldest first) and then its recurrent state (`S[h][i][j]`, the model crate's layout),
//! all `f32`; beside them a tag, the boundary plus one (0 when the buffer holds nothing).
//! A read checks the tag against the position the row expects, so a stale, zeroed or
//! encrypted buffer panics instead of resuming from the wrong state.
//!
//! **A buffer's image** (what a zero clears, a transfer moves and the idle cipher XORs) is
//! two ranges, each from a fresh keystream block: the data's `f32` bits, then the tag.
//!
//! **A slot's tail** holds, for each position after the slot's base, the inputs the
//! DeltaNet layers' recurrence reads (per layer the raw `mixed` row and the `a` and `b`
//! rows), with a position tag per entry. It is allocated once at full size and only
//! overwritten in place, like device-resident slot state, and zeroed by `ResetSlot`.

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::secret::KvKey;
use eidola_engine_model::qwen::DeltaNetState;
use eidola_engine_model::qwen::config::DeltaNetSpec;

use crate::page::PageBuf;

/// Floats one DeltaNet layer takes in a buffer.
pub(crate) fn layer_floats(spec: &DeltaNetSpec) -> usize {
    (spec.conv_kernel - 1) * spec.conv_dim() + spec.state_len()
}

/// Floats one position takes in a tail: per layer, `mixed`, `a` and `b`.
pub(crate) fn tail_floats_per_layer(spec: &DeltaNetSpec) -> usize {
    spec.conv_dim() + 2 * spec.num_v_heads
}

/// State buffers: a device pool or its host tier.
#[derive(Debug)]
pub(crate) struct StatePool {
    spec: DeltaNetSpec,
    layers: usize,
    data: PageBuf<f32>,
    tags: PageBuf<u32>,
    /// Per buffer: whether it is held encrypted (instrumentation).
    encrypted: Vec<bool>,
}

impl StatePool {
    pub fn new(spec: &DeltaNetSpec, layers: usize, buffers: usize) -> Self {
        Self {
            spec: spec.clone(),
            layers,
            data: PageBuf::zeroed(buffers * layers * layer_floats(spec)),
            tags: PageBuf::zeroed(buffers),
            encrypted: vec![false; buffers],
        }
    }

    pub fn lock(&mut self) -> std::io::Result<()> {
        let locked = self.data.lock().and_then(|()| self.tags.lock());
        if locked.is_err() {
            self.data.unlock();
            self.tags.unlock();
        }
        locked
    }

    fn floats(&self) -> usize {
        self.layers * layer_floats(&self.spec)
    }

    fn range(&self, b: u32) -> std::ops::Range<usize> {
        let n = self.floats();
        b as usize * n..(b as usize + 1) * n
    }

    /// Bytes one buffer takes: its data, its tag and its encrypted flag.
    pub fn buffer_bytes(&self) -> usize {
        self.floats() * 4 + 4 + 1
    }

    /// The states at `len` held by buffer `b`. Panics unless the buffer is plaintext
    /// holding exactly that boundary's state.
    pub fn read(&self, b: u32, len: u32) -> Vec<DeltaNetState> {
        assert!(
            !self.encrypted[b as usize],
            "state buffer {b}: read while encrypted"
        );
        assert_eq!(
            self.tags[b as usize],
            len + 1,
            "state buffer {b}: holds the state at {} (or none), read as the state at {len}",
            self.tags[b as usize].wrapping_sub(1)
        );
        let data = &self.data[self.range(b)];
        let (conv, lf) = (
            (self.spec.conv_kernel - 1) * self.spec.conv_dim(),
            layer_floats(&self.spec),
        );
        (0..self.layers)
            .map(|l| {
                let d = &data[l * lf..(l + 1) * lf];
                DeltaNetState {
                    conv: d[..conv].to_vec(),
                    recurrent: d[conv..].to_vec(),
                }
            })
            .collect()
    }

    /// Stores `states`, the states at `len`, into buffer `b`.
    pub fn write(&mut self, b: u32, len: u32, states: &[DeltaNetState]) {
        assert_eq!(states.len(), self.layers);
        let lf = layer_floats(&self.spec);
        let r = self.range(b);
        let data = &mut self.data[r];
        for (l, s) in states.iter().enumerate() {
            let d = &mut data[l * lf..(l + 1) * lf];
            d[..s.conv.len()].copy_from_slice(&s.conv);
            d[s.conv.len()..].copy_from_slice(&s.recurrent);
        }
        self.tags[b as usize] = len + 1;
    }

    /// Whether buffer `b` holds nothing: zero bytes, and not encrypted.
    pub fn is_zero(&self, b: u32) -> bool {
        !self.encrypted[b as usize]
            && self.tags[b as usize] == 0
            && self.data[self.range(b)].iter().all(|x| x.to_bits() == 0)
    }

    pub fn is_encrypted(&self, b: u32) -> bool {
        self.encrypted[b as usize]
    }

    pub fn zero(&mut self, b: u32) {
        let r = self.range(b);
        self.data[r].fill(0.0);
        self.tags[b as usize] = 0;
        self.encrypted[b as usize] = false;
    }

    /// Encrypts (`encrypt`) or decrypts buffer `b` in place: one keystream XOR over its
    /// image. Panics on encrypting an encrypted buffer or decrypting a plaintext one.
    pub fn crypt(&mut self, b: u32, key: &KvKey, nonce: &KvNonce, encrypt: bool) {
        assert_ne!(
            self.encrypted[b as usize],
            encrypt,
            "state buffer {b}: {} twice",
            if encrypt { "encrypted" } else { "decrypted" }
        );
        self.encrypted[b as usize] = encrypt;
        let mut ks = KvKeystream::new(key, nonce);
        let r = self.range(b);
        let data = &mut self.data[r];
        ks.range(data.len(), |i, k| {
            data[i] = f32::from_bits(data[i].to_bits() ^ k)
        });
        let tag = &mut self.tags[b as usize];
        ks.range(1, |_, k| *tag ^= k);
    }

    /// Copies buffer `src` of `from` over buffer `dst` of `to`: data, tag and the
    /// encrypted flag.
    pub fn transfer(from: &StatePool, src: u32, to: &mut StatePool, dst: u32) {
        assert_eq!(
            from.floats(),
            to.floats(),
            "transfer between pools of one layout"
        );
        let (s, d) = (from.range(src), to.range(dst));
        to.data[d].copy_from_slice(&from.data[s]);
        to.tags[dst as usize] = from.tags[src as usize];
        to.encrypted[dst as usize] = from.encrypted[src as usize];
    }

    /// The buffer's image as words, in [`StatePool::crypt`]'s order (tests).
    pub fn image(&self, b: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self.data[self.range(b)]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        out.push(self.tags[b as usize]);
        out
    }
}

/// One position's recurrence inputs, per DeltaNet layer: `(mixed, a, b)`.
pub(crate) type TailEntry = Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>;

/// One slot's state between steps: its base and its tail.
#[derive(Debug)]
pub(crate) struct SlotTail {
    spec: DeltaNetSpec,
    layers: usize,
    capacity: usize,
    /// `(buffer, base_len)`, once a row named it.
    pub base: Option<(u32, u32)>,
    /// Positions held, from the base on.
    len: usize,
    data: PageBuf<f32>,
    /// Per entry, its position plus one.
    tags: PageBuf<u32>,
}

impl SlotTail {
    pub fn new(spec: &DeltaNetSpec, layers: usize, capacity: usize) -> Self {
        let per = layers * tail_floats_per_layer(spec);
        Self {
            spec: spec.clone(),
            layers,
            capacity,
            base: None,
            len: 0,
            data: PageBuf::zeroed(capacity * per),
            tags: PageBuf::zeroed(capacity),
        }
    }

    fn per(&self) -> usize {
        self.layers * tail_floats_per_layer(&self.spec)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// The entry for `pos`, which must be held.
    pub fn entry(&self, pos: u32) -> TailEntry {
        let (_, base_len) = self.base.expect("a slot with a base");
        let i = (pos - base_len) as usize;
        assert!(i < self.len, "the tail does not reach position {pos}");
        assert_eq!(
            self.tags[i],
            pos + 1,
            "tail entry {i} holds another position"
        );
        let per = self.per();
        let d = &self.data[i * per..(i + 1) * per];
        let (cd, vh) = (self.spec.conv_dim(), self.spec.num_v_heads);
        let w = tail_floats_per_layer(&self.spec);
        (0..self.layers)
            .map(|l| {
                let x = &d[l * w..(l + 1) * w];
                (
                    x[..cd].to_vec(),
                    x[cd..cd + vh].to_vec(),
                    x[cd + vh..].to_vec(),
                )
            })
            .collect()
    }

    /// Replaces the slot's base and tail: `entries` are the positions from `base_len` on.
    /// Panics past the capacity: the host keeps a slot's tail within it.
    pub fn set(&mut self, base: (u32, u32), entries: &[TailEntry]) {
        assert!(
            entries.len() <= self.capacity,
            "a tail of {} positions past {}, more than {}",
            entries.len(),
            base.1,
            self.capacity
        );
        let per = self.per();
        let w = tail_floats_per_layer(&self.spec);
        self.data.fill(0.0);
        self.tags.fill(0);
        for (i, e) in entries.iter().enumerate() {
            let d = &mut self.data[i * per..(i + 1) * per];
            for (l, (mixed, a, b)) in e.iter().enumerate() {
                let x = &mut d[l * w..(l + 1) * w];
                x[..mixed.len()].copy_from_slice(mixed);
                x[mixed.len()..mixed.len() + a.len()].copy_from_slice(a);
                x[mixed.len() + a.len()..].copy_from_slice(b);
            }
            self.tags[i] = base.1 + i as u32 + 1;
        }
        self.base = Some(base);
        self.len = entries.len();
    }

    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.tags.fill(0);
        self.base = None;
        self.len = 0;
    }

    pub fn is_zero(&self) -> bool {
        self.base.is_none()
            && self.len == 0
            && self.data.iter().all(|x| x.to_bits() == 0)
            && self.tags.iter().all(|&t| t == 0)
    }
}
