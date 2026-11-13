//! A declared cached prefix (`cached_prefix`) end to end, in process: real HTTP, the
//! pinned MiMo template and tokenizer, the engine core over the CPU executor, on the
//! synthetic dev model. A client declares what an earlier request under the same key
//! left in the engine by the shared rule (`eidola_common::engine_protocol::declared_prompt`);
//! these tests hold the node and its engine to serving every such declaration, and to
//! refusing every other one with `409` before anything of it is computed.

mod common;

use base64::Engine as _;
use common::*;
use eidola_common::engine_protocol::{CachedPrefix, declared_prompt};
use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_chat::{ChatInput, RenderOptions};
use eidola_engine_cpu::{CpuExecutor, CpuExecutorConfig, MtpHidden};
use eidola_server_engine::config::env;
use serde_json::{Value, json};

/// A user message long enough that its prompt spans more than one declared quantum
/// (256 tokens) and leaves room in the dev model's 1,024-position window.
fn long_message() -> String {
    [
        "The quick brown fox jumps over the lazy dog while the five boxing wizards jump",
        "quickly, and a wizard's job is to vex chumps quickly in fog. Pack my box with",
        "five dozen liquor jugs; how vexingly quick daft zebras jump!",
    ]
    .join(" ")
    .repeat(6)
}

/// The dev model's window in tokens (`common`'s `MAX_MODEL_LEN`), which bounds what a
/// declaration budgets.
const WINDOW: u64 = 1_024;

fn key(byte: u8) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32])
}

/// A greedy request for `messages` under `key`.
fn body(messages: Value, max_tokens: u32, key: Option<&str>) -> Value {
    let mut b = json!({
        "model": MODEL_ID,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    });
    if let Some(k) = key {
        b["cache_key"] = k.into();
    }
    b
}

fn declaring(mut b: Value, prefix: CachedPrefix) -> Value {
    b["cached_prefix"] = prefix.to_value();
    b
}

fn usage(v: &Value, member: &str) -> u64 {
    v["usage"][member].as_u64().unwrap()
}

fn cached(v: &Value) -> u64 {
    v["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap()
}

fn message_text(v: &Value, member: &str) -> String {
    v["choices"][0]["message"][member]
        .as_str()
        .unwrap_or("")
        .to_string()
}

/// Asserts a `409` with `kind`, a plain JSON error rather than a stream.
fn assert_refused(status: u16, v: &Value, kind: &str) {
    assert_eq!(status, 409, "{v}");
    assert_eq!(v["error"]["type"], kind, "{v}");
}

/// The first answer of a conversation under `key`: its body and answer.
async fn first_turn(node: &TestNode, key: &str, max_tokens: u32) -> (Value, Value) {
    let first = body(
        json!([{"role": "user", "content": long_message()}]),
        max_tokens,
        Some(key),
    );
    let (status, answer) = node.chat_json(&first).await;
    assert_eq!(status, 200, "{answer}");
    assert!(usage(&answer, "prompt_tokens") > 256, "{answer}");
    (first, answer)
}

/// The conversation `first` began, extended by its answer and a new user message: the
/// next turn's request, and the bytes and messages it adds as the pricing walk counts
/// them (`eidola_common::PromptCharge`: each message, and its `content` and
/// `reasoning_content` strings).
fn next_turn(first: &Value, answer: &Value, key: Option<&str>) -> (Value, u64, u64) {
    let reasoning = message_text(answer, "reasoning_content");
    let content = message_text(answer, "content");
    let question = "And what would a second, shorter answer say?";
    let mut messages = first["messages"].as_array().unwrap().clone();
    messages.push(json!({"role": "assistant", "reasoning_content": reasoning, "content": content}));
    messages.push(json!({"role": "user", "content": question}));
    let added_bytes = (reasoning.len() + content.len() + question.len()) as u64;
    (body(Value::Array(messages), 6, key), added_bytes, 2)
}

/// A new turn declaring what the previous one left: its prompt, which the next turn's
/// begins with, and the bytes it adds.
#[tokio::test(flavor = "multi_thread")]
async fn a_new_turn_declaring_the_last_ones_prompt_is_served() {
    let node = TestNode::start(&[]).await;
    let k = key(1);
    let (first, answer) = first_turn(&node, &k, 6).await;
    let (next, bytes, messages) = next_turn(&first, &answer, Some(&k));
    let prefix = declared_prompt(usage(&answer, "prompt_tokens"), bytes, messages, WINDOW).unwrap();
    let (status, served) = node.chat_json(&declaring(next.clone(), prefix)).await;
    assert_eq!(status, 200, "{served}");
    assert!(
        cached(&served) >= u64::from(prefix.cached_tokens),
        "{served} {prefix:?}"
    );
    assert!(usage(&served, "prompt_tokens") <= prefix.prompt_limit());
    // The declaration changes nothing about the answer.
    let (_, undeclared) = node.chat_json(&next).await;
    assert_eq!(served["choices"], undeclared["choices"]);
}

/// A regeneration of the same request declaring its own prompt.
#[tokio::test(flavor = "multi_thread")]
async fn a_regeneration_declaring_its_prompt_is_served() {
    let node = TestNode::start(&[]).await;
    let k = key(2);
    let (first, answer) = first_turn(&node, &k, 6).await;
    let prefix = declared_prompt(usage(&answer, "prompt_tokens"), 0, 0, WINDOW).unwrap();
    let (status, again) = node.chat_json(&declaring(first.clone(), prefix)).await;
    assert_eq!(status, 200, "{again}");
    assert!(cached(&again) >= u64::from(prefix.cached_tokens), "{again}");
    assert_eq!(again["choices"], answer["choices"]);
    // Admitted and finished within one step (the uncached tail and one token): the
    // admission is announced before the tokens all the same.
    let mut one = declaring(first, prefix);
    one["max_completion_tokens"] = 1.into();
    let (status, v) = node.chat_json(&one).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(usage(&v, "completion_tokens"), 1);
}

/// A user message whose continued prompt is just under two quanta (474 tokens), so a
/// continuation of a few dozen tokens declares a quantum more than its prompt alone.
fn longer_message() -> String {
    let m = long_message();
    m.repeat(2)[..m.len() * 3 / 2].to_string()
}

/// A request continuing the answer `content` to `message` under `key`.
fn continuing(message: &str, content: &str, max_tokens: u32, key: &str) -> Value {
    let mut b = body(
        json!([
            {"role": "user", "content": message},
            {"role": "assistant", "reasoning_content": "a thought", "content": content},
        ]),
        max_tokens,
        Some(key),
    );
    b["add_generation_prompt"] = false.into();
    b["continue_final_message"] = "content".into();
    b
}

/// The prompt tokens the node renders and encodes for a [`continuing`] body.
fn continued_tokens(body: &Value) -> Vec<u32> {
    let model = loaded();
    let input = ChatInput::from_json(&body["messages"].to_string(), None).unwrap();
    let options = RenderOptions {
        add_generation_prompt: false,
        enable_thinking: None,
        continue_final_message: Some("content"),
    };
    let prompt = model.template().render_prompt(&input, options).unwrap();
    model.tokenizer().encode_prompt(&prompt).unwrap()
}

/// The tokens the dev model generates greedily after `prompt`: the engine driven
/// directly, without the node (greedy output is the same at every draft width).
fn greedy_tokens(prompt: Vec<u32>, max_tokens: u32) -> Vec<u32> {
    let model = loaded();
    let exec = CpuExecutor::new(
        model.reference().unwrap().clone(),
        CpuExecutorConfig {
            block_size: 16,
            num_blocks: 256,
            host_blocks: 0,
            num_state_slots: 1,
            max_model_len: 1024,
            buckets: vec![Bucket {
                max_seqs: 1,
                max_tokens: 256,
            }],
            mtp_depths: Vec::new(),
            mtp_hidden: MtpHidden::Normed,
            sampleable_vocab_size: model.tokenizer().vocab_size() as u32,
            pad_batches: false,
            record: false,
        },
    );
    let mut engine = Engine::new(
        exec,
        SchedulerConfig {
            max_batched_tokens: 256,
            max_seqs: 1,
            max_prefill_chunk: 256,
            eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
            speculative: false,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        },
    )
    .unwrap();
    engine
        .submit(Request {
            id: 1,
            prompt,
            sampling: SamplingParams::greedy(),
            max_tokens,
            required_cached_tokens: 0,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
        })
        .unwrap();
    let mut out = Vec::new();
    for now in 0.. {
        for ev in engine.step(now).unwrap() {
            out.extend(ev.tokens);
            if ev.finish.is_some() {
                return out;
            }
        }
    }
    unreachable!()
}

/// A continuation of an answer that ended at its ceiling, continuable, declaring its
/// prompt and completion together (`declared_prompt(prompt + completion, 0, 0)`), is
/// served exactly where its prompt re-encodes to the earlier prompt followed by the
/// generated tokens through the declared boundary, and refused as missing where it does
/// not: a `continuable` cut says the text resumes exactly, not that the model sampled
/// the canonical encoding of that text, and the cache is keyed by tokens. Both happen on
/// the dev model, whose gibberish is often not canonical. Declaring the earlier prompt
/// alone, with the completion counted as added bytes, is served in every case.
///
/// The answer continued is itself a continuation of an answer in progress, so the dev
/// model's output is content with no opening `<think>`, which it can cut exactly.
#[tokio::test(flavor = "multi_thread")]
async fn a_continuation_declaring_prompt_and_completion_needs_canonical_tokens() {
    let node = TestNode::start(&[]).await;
    let message = longer_message();
    let so_far = "The answer so far";
    let (mut exact, mut inexact) = (0, 0);
    for (byte, max) in (20u8..).step_by(2).zip([8u32, 16, 34, 40, 45, 50, 60]) {
        let (full_key, safe_key) = (key(byte), key(byte + 1));
        let first = continuing(&message, so_far, max, &full_key);
        let (status, answer) = node.chat_json(&first).await;
        assert_eq!(status, 200, "{answer}");
        let choice = &answer["choices"][0];
        if choice["finish_reason"] != "length" || choice["continuable"] != true {
            continue;
        }
        let (prompt, completion) = (
            usage(&answer, "prompt_tokens"),
            usage(&answer, "completion_tokens"),
        );
        let content = message_text(&answer, "content");
        let next = |k: &str| continuing(&message, &format!("{so_far}{content}"), 6, k);

        // Whether the continuation's tokens are the earlier ones through the boundary.
        let full = declared_prompt(prompt + completion, 0, 0, WINDOW).unwrap();
        let first_tokens = continued_tokens(&first);
        assert_eq!(first_tokens.len() as u64, prompt);
        let mut resumed = first_tokens.clone();
        resumed.extend(greedy_tokens(first_tokens, max));
        assert_eq!(resumed.len() as u64, prompt + completion);
        let boundary = full.cached_tokens as usize;
        let canonical = continued_tokens(&next(&full_key))[..boundary] == resumed[..boundary];

        let (status, v) = node.chat_json(&declaring(next(&full_key), full)).await;
        if canonical {
            assert_eq!(status, 200, "max {max}: {v} {full:?}");
            assert!(
                cached(&v) >= u64::from(full.cached_tokens),
                "max {max}: {v}"
            );
            exact += 1;
        } else {
            assert_refused(status, &v, "cached_prefix_missing");
            inexact += 1;
        }

        // The earlier prompt alone, under a key whose cache holds nothing else.
        let (status, again) = node
            .chat_json(&continuing(&message, so_far, max, &safe_key))
            .await;
        assert_eq!(status, 200);
        assert_eq!(again["choices"], answer["choices"]);
        let safe = declared_prompt(prompt, content.len() as u64, 0, WINDOW).unwrap();
        let (status, v) = node.chat_json(&declaring(next(&safe_key), safe)).await;
        assert_eq!(status, 200, "max {max}: {v} {safe:?}");
        assert!(
            cached(&v) >= u64::from(safe.cached_tokens),
            "max {max}: {v}"
        );
    }
    assert!(exact > 0, "no continuable cut re-encoded canonically");
    assert!(inexact > 0, "every continuable cut re-encoded canonically");
}

/// Under another key, or once the entry has expired, the same declarations are refused
/// before anything is computed: no engine step runs for them.
#[tokio::test(flavor = "multi_thread")]
async fn a_declaration_under_another_key_is_refused_before_compute() {
    let node = TestNode::start(&[]).await;
    let bodies = declarations(&node, &key(9), &key(10)).await;
    refuse_without_a_step(&node, bodies).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_declaration_whose_entry_expired_is_refused_before_compute() {
    let node = TestNode::start(&[(env::CACHE_IDLE_TTL_SECS, "1")]).await;
    let bodies = declarations(&node, &key(11), &key(11)).await;
    // Past the idle TTL, and past the idle sweep that zeroes what expired: wait until a
    // sweep interval passes with no step.
    let mut steps = node.stats().engine_steps;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let now = node.stats().engine_steps;
        if now == steps {
            break;
        }
        steps = now;
    }
    refuse_without_a_step(&node, bodies).await;
}

/// After a first turn under `first_key`, the extension and the regeneration declaring
/// what it left, sent under `key`.
async fn declarations(node: &TestNode, first_key: &str, key: &str) -> [Value; 2] {
    let (first, answer) = first_turn(node, first_key, 6).await;
    let prompt = usage(&answer, "prompt_tokens");
    let (next, bytes, messages) = next_turn(&first, &answer, Some(key));
    let mut regen = declaring(first, declared_prompt(prompt, 0, 0, WINDOW).unwrap());
    regen["cache_key"] = key.into();
    [
        declaring(
            next,
            declared_prompt(prompt, bytes, messages, WINDOW).unwrap(),
        ),
        regen,
    ]
}

async fn refuse_without_a_step(node: &TestNode, bodies: [Value; 2]) {
    for b in bodies {
        // The earlier requests' last maintenance has run (their keys are dropped with it).
        let before = node
            .wait_for(std::time::Duration::from_secs(5), |s| {
                s.in_engine == 0 && s.kv_keys_held == 0
            })
            .await;
        let (status, v) = node.chat_json(&b).await;
        assert_refused(status, &v, "cached_prefix_missing");
        let after = node
            .wait_for(std::time::Duration::from_secs(5), |s| s.in_engine == 0)
            .await;
        assert_eq!(after.engine_steps, before.engine_steps, "a step ran for it");
        assert_eq!(after.prefix_refused, before.prefix_refused + 1);
        assert_eq!(after.submitted, before.submitted + 1);
        assert_eq!(after.kv_keys_held, 0);
    }
}

/// A declaration refused while it waits behind another request is still a plain HTTP
/// error: the response opens only once the engine has admitted the request.
#[tokio::test(flavor = "multi_thread")]
async fn a_refusal_after_waiting_for_admission_is_still_a_plain_error() {
    let node = TestNode::start(&[(env::MAX_SEQS, "1")]).await;
    let busy = body(json!([{"role": "user", "content": "hi"}]), 200, None);
    let running = {
        let req = node.chat(&busy);
        tokio::spawn(async move { req.send().await.unwrap().bytes().await.unwrap() })
    };
    node.wait_for(std::time::Duration::from_secs(10), |s| s.in_engine == 1)
        .await;
    let declared = declaring(
        body(
            json!([{"role": "user", "content": long_message()}]),
            4,
            Some(&key(11)),
        ),
        CachedPrefix {
            cached_tokens: 256,
            new_tokens: 1024,
        },
    );
    let (status, v) = node.chat_json(&declared).await;
    assert_refused(status, &v, "cached_prefix_missing");
    // The engine thread publishes its counters after it has answered.
    node.wait_for(std::time::Duration::from_secs(10), |s| {
        s.prefix_refused == 1
    })
    .await;
    running.await.unwrap();
}

/// A prompt longer than its declaration allows is refused after tokenizing, before the
/// engine sees it.
#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_past_its_budget_is_refused_before_submission() {
    let node = TestNode::start(&[]).await;
    let k = key(12);
    let (first, answer) = first_turn(&node, &k, 6).await;
    let prompt = usage(&answer, "prompt_tokens");
    let (next, bytes, messages) = next_turn(&first, &answer, Some(&k));
    let fair = declared_prompt(prompt, bytes, messages, WINDOW).unwrap();
    let before = node
        .wait_for(std::time::Duration::from_secs(5), |s| {
            s.in_engine == 0 && s.kv_keys_held == 0
        })
        .await;
    for prefix in [
        CachedPrefix {
            new_tokens: 0,
            ..fair
        },
        CachedPrefix {
            cached_tokens: 256,
            new_tokens: (prompt - 257) as u32,
        },
    ] {
        let (status, v) = node.chat_json(&declaring(first.clone(), prefix)).await;
        assert_refused(status, &v, "prompt_budget_exceeded");
        let (status, v) = node.chat_json(&declaring(next.clone(), prefix)).await;
        assert_refused(status, &v, "prompt_budget_exceeded");
    }
    let after = node.stats();
    assert_eq!(
        after.submitted, before.submitted,
        "nothing reached the engine"
    );
    assert_eq!(after.engine_steps, before.engine_steps);
    // Exactly at the limit is within it.
    let exact = CachedPrefix {
        cached_tokens: 256,
        new_tokens: (prompt - 256) as u32,
    };
    let (status, v) = node.chat_json(&declaring(first.clone(), exact)).await;
    assert_eq!(status, 200, "{v}");
    assert!(cached(&v) >= 256);
    // Counted once it was served, with every refusal before it: none was the engine's.
    let s = node.stats();
    assert_eq!(s.submitted, before.submitted + 1);
    assert_eq!(s.rejected, before.rejected);
}

/// A declared prompt longer than the window is refused for its length, not for its
/// budget. The rule a client declares by budgets what a request adds at no more than
/// the window leaves, so a prompt past the window is past that budget too; told its
/// budget was exceeded, the client would send the request again without a declaration
/// and be refused, and charged, a second time for the length.
#[tokio::test(flavor = "multi_thread")]
async fn a_declared_prompt_past_the_window_is_refused_for_its_length() {
    let node = TestNode::start(&[]).await;
    let k = key(14);
    let (first, answer) = first_turn(&node, &k, 6).await;
    let prompt = usage(&answer, "prompt_tokens");
    // The conversation, extended by a message that alone fills the window.
    let addition = long_message().repeat(4);
    let mut messages = first["messages"].as_array().unwrap().clone();
    messages.push(json!({"role": "assistant", "content": message_text(&answer, "content")}));
    messages.push(json!({"role": "user", "content": addition}));
    let added_bytes = (message_text(&answer, "content").len() + addition.len()) as u64;
    let next = body(Value::Array(messages), 6, Some(&k));
    let prefix = declared_prompt(prompt, added_bytes, 2, WINDOW).unwrap();
    assert_eq!(
        prefix.prompt_limit(),
        WINDOW,
        "the budget stops at the window"
    );
    let before = node
        .wait_for(std::time::Duration::from_secs(5), |s| {
            s.in_engine == 0 && s.kv_keys_held == 0
        })
        .await;

    let (status, v) = node.chat_json(&declaring(next.clone(), prefix)).await;
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["type"], "context_length_exceeded", "{v}");
    // Undeclared, it is refused the same way: the declaration changed nothing.
    let (status, v) = node.chat_json(&next).await;
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["type"], "context_length_exceeded", "{v}");
    let after = node.stats();
    assert_eq!(
        after.submitted, before.submitted,
        "nothing reached the engine"
    );
    // The engine refuses a prompt past its length with the same error, without
    // counting a submission: served after them, one more request shows it refused none.
    let s = node.stats_after_one_more().await;
    assert_eq!(s.submitted, before.submitted + 1);
    assert_eq!(s.rejected, before.rejected);
}

/// A declaration reaching the prompt's final token can never hold (that token is always
/// recomputed): refused as missing.
#[tokio::test(flavor = "multi_thread")]
async fn a_declaration_covering_the_final_token_is_missing() {
    let node = TestNode::start(&[]).await;
    let k = key(13);
    let (first, answer) = first_turn(&node, &k, 6).await;
    let prompt = usage(&answer, "prompt_tokens") as u32;
    let (status, v) = node
        .chat_json(&declaring(
            first,
            CachedPrefix {
                cached_tokens: prompt,
                new_tokens: 0,
            },
        ))
        .await;
    assert_refused(status, &v, "cached_prefix_missing");
}

/// The member's shape and the subset's rule: exactly the two counts, a key, and at least
/// one cached token; anything else is `400` like the subset's other refusals.
#[tokio::test(flavor = "multi_thread")]
async fn a_declaration_outside_the_shape_or_without_a_key_is_invalid() {
    let node = TestNode::start(&[]).await;
    let base = body(
        json!([{"role": "user", "content": "hi"}]),
        4,
        Some(&key(14)),
    );
    let with = |prefix: Value, keyed: bool| {
        let mut b = base.clone();
        b["cached_prefix"] = prefix;
        if !keyed {
            b.as_object_mut().unwrap().remove("cache_key");
        }
        b
    };
    for b in [
        with(json!({"cached_tokens": 256, "new_tokens": 10}), false),
        with(json!({"cached_tokens": 0, "new_tokens": 10}), true),
        with(json!({"cached_tokens": 256}), true),
        with(json!({"new_tokens": 10}), true),
        with(
            json!({"cached_tokens": 256, "new_tokens": 10, "x": 1}),
            true,
        ),
        with(json!({"cached_tokens": -1, "new_tokens": 10}), true),
        with(json!({"cached_tokens": 1.5, "new_tokens": 10}), true),
        with(json!({"cached_tokens": "256", "new_tokens": 10}), true),
        with(
            json!({"cached_tokens": 4_294_967_296u64, "new_tokens": 10}),
            true,
        ),
        with(json!([256, 10]), true),
        with(json!(256), true),
    ] {
        let (status, v) = node.chat_json(&b).await;
        assert_eq!(status, 400, "{b}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{b}: {v}");
    }
    assert_eq!(node.stats().submitted, 0);
    let s = node.stats_after_one_more().await;
    assert_eq!(
        (s.submitted, s.rejected),
        (1, 0),
        "nothing before reached the engine"
    );
}

/// A request that declares nothing is answered exactly as before: `cached_prefix: null`
/// is no declaration.
#[tokio::test(flavor = "multi_thread")]
async fn a_null_declaration_is_none() {
    let node = TestNode::start(&[]).await;
    let mut b = body(json!([{"role": "user", "content": "hi"}]), 4, None);
    let (status, plain) = node.chat_json(&b).await;
    assert_eq!(status, 200);
    b["cached_prefix"] = Value::Null;
    let (status, v) = node.chat_json(&b).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["choices"], plain["choices"]);
}

/// With the prefix cache on, the block size must divide the declared quantum, or the
/// configuration does not parse (and so the node does not boot); with it off, any block
/// size boots.
#[test]
fn a_block_size_the_quantum_does_not_land_on_is_refused() {
    let mut map = env_map();
    map.insert(env::KV_BLOCK_SIZE, "24".into());
    let e = config_from(&map).unwrap_err();
    assert!(e.contains(env::KV_BLOCK_SIZE), "{e}");
    map.insert(env::PREFIX_CACHE, "false".into());
    map.insert(env::KV_HOST_BYTES, "0".into());
    let config = config_from(&map).unwrap();
    assert!(eidola_server_engine::start(config, loaded()).is_ok());
}
