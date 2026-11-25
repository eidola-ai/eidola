//! A prompt's words leave nothing behind in the tokenizer: after a prompt is encoded and
//! its ids dropped, no allocation the encoding made is still live. `tokenizers`' BPE model
//! keeps a per-thread cache of the words it has merged, which would hold every word of
//! every past prompt in memory between requests (out of reach of any zeroing of freed
//! memory); the crate's tokenizers are built without it (`tokenizer.rs`,
//! `without_word_cache`). Counted by this binary's own allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};

use eidola_engine_chat::{ChatFormat, Tokenizer};

struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as i64, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as i64, Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size as i64 - layout.size() as i64, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn qwen35() -> Tokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35");
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(dir.join("tokenizer.json.gz")).unwrap())
        .read_to_end(&mut json)
        .unwrap();
    let config = std::fs::read(dir.join("config.json")).unwrap();
    Tokenizer::for_format(ChatFormat::Qwen3_5, &json, &config).unwrap()
}

/// `n` distinct made-up words from `seed`, none of them a single token.
fn words(seed: u64, n: usize) -> String {
    let mut state = seed;
    let mut out = String::new();
    for _ in 0..n {
        out.push(' ');
        for _ in 0..12 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push((b'a' + (state >> 59) as u8 % 26) as char);
        }
    }
    out
}

#[test]
fn an_encoded_prompt_leaves_no_word_in_memory() {
    let tokenizer = qwen35();
    // Every lazily built structure the encoder keeps for itself (the pre-tokenizer's
    // regex caches among them) is built by a first prompt of the same kind.
    for seed in 1..4 {
        drop(tokenizer.encode(&words(seed, 500)).unwrap());
    }
    let text = words(99, 4000);
    let before = LIVE.load(Ordering::SeqCst);
    let ids = tokenizer.encode(&text).unwrap();
    assert!(ids.len() > 4000);
    drop(ids);
    let retained = LIVE.load(Ordering::SeqCst) - before;
    // A word cache would keep each of the 4,000 words (its text and its merged
    // symbols): hundreds of kilobytes.
    assert!(
        retained <= 0,
        "{retained} bytes the encoding allocated are still live"
    );
}
