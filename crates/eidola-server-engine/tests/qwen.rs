//! The node serving Qwen3.8, end to end and in process: real HTTP, the model's pinned
//! template and tokenizer, the engine core over the Qwen3.8 CPU executor (its recurrent
//! state through the serving core), on the Qwen dev model (`common::fixture`). The
//! model's output is gibberish; these tests assert structure, refusals, and token-level
//! equality with the engine driven directly.

mod common;

use std::sync::Arc;

use base64::Engine as _;
use common::*;
use eidola_common::engine_protocol::{CachedPrefix, declared_prompt};
use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_chat::{
    ChatDelta, ChatFormat, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, StopCause,
    ToolSchemas,
};
use eidola_engine_cpu::qwen::{QwenCpuExecutor, QwenCpuExecutorConfig};
use eidola_engine_model::qwen::QwenModel;
use eidola_server_engine::config::env;
use eidola_server_engine::model::{LoadedModel, Served};
use eidola_server_engine::pipeline::{Decoder, qwen_options};
use serde_json::{Value, json};

const LONG_TEXT: &str = "The quick brown fox jumps over the lazy dog while the five boxing \
    wizards jump quickly, and a wizard's job is to vex chumps quickly in fog. Pack my box \
    with five dozen liquor jugs; how vexingly quick daft zebras jump!";

/// The dev model's window in tokens (`common`'s `MAX_MODEL_LEN`).
const WINDOW: u64 = 1_024;

fn reference(model: &LoadedModel) -> Arc<QwenModel> {
    match model.served() {
        Served::Qwen { model, .. } => model.clone(),
        _ => panic!("the Qwen dev model"),
    }
}

/// A greedy request for `messages`.
fn body(messages: Value, max_tokens: u32) -> Value {
    json!({
        "model": QWEN_MODEL_ID,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    })
}

fn user(text: &str, max_tokens: u32) -> Value {
    body(json!([{"role": "user", "content": text}]), max_tokens)
}

/// The prompt text the template renders for `messages` and `tools` with the node's
/// options, rendered here (not by the node).
fn prompt_text(messages: &Value, tools: Option<&Value>) -> String {
    let model = qwen_loaded();
    let Served::Qwen { template, .. } = model.served() else {
        panic!("the Qwen dev model")
    };
    let input = ChatInput::from_json(
        &messages.to_string(),
        tools.map(|t| t.to_string()).as_deref(),
    )
    .unwrap();
    template.render(&input, &qwen_options()).unwrap()
}

/// The prompt tokens for `messages` and `tools`, request data never a control token.
fn prompt_tokens(messages: &Value, tools: Option<&Value>) -> Vec<u32> {
    let model = qwen_loaded();
    let Served::Qwen { template, .. } = model.served() else {
        panic!("the Qwen dev model")
    };
    let input = ChatInput::from_json(
        &messages.to_string(),
        tools.map(|t| t.to_string()).as_deref(),
    )
    .unwrap();
    let prompt = template.render_prompt(&input, &qwen_options()).unwrap();
    model.tokenizer().encode_prompt(&prompt).unwrap()
}

/// Greedy generation through the engine core and the Qwen3.8 CPU executor, without
/// speculation.
fn direct_greedy(prompt: Vec<u32>, max_tokens: u32) -> (Vec<u32>, FinishReason) {
    let model = qwen_loaded();
    let exec = QwenCpuExecutor::new(
        reference(&model),
        QwenCpuExecutorConfig {
            block_size: 16,
            num_blocks: 256,
            host_blocks: 0,
            state_buffers: 21,
            host_state_buffers: 0,
            num_state_slots: 4,
            max_model_len: 1024,
            buckets: vec![Bucket {
                max_seqs: 4,
                max_tokens: 256,
            }],
            draft_tokens: 0,
            sampleable_vocab_size: model.sampleable_vocab_size(),
            record: false,
        },
    );
    let mut engine = Engine::new(
        exec,
        SchedulerConfig {
            max_batched_tokens: 256,
            max_seqs: 4,
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
            if let Some(f) = ev.finish {
                return (out, f);
            }
        }
    }
    unreachable!()
}

/// What the format's output parser makes of `tokens` with thinking on.
fn expected_output(tokens: &[u32], finish: FinishReason) -> (ChatDelta, &'static str) {
    let model = qwen_loaded();
    let mut parser = OutputParser::new(
        OutputConfig::qwen3_8(None, None),
        PrefixedCallIds("c".into()),
    );
    let mut all = ChatDelta::default();
    let mut text_tokens = tokens;
    if finish == FinishReason::Stop {
        assert!(model.tokenizer().is_eos(*tokens.last().unwrap()));
        text_tokens = &tokens[..tokens.len() - 1];
    }
    for &t in text_tokens {
        let d = parser.push_token(model.tokenizer(), t).unwrap();
        all.reasoning_content.push_str(&d.reasoning_content);
        all.content.push_str(&d.content);
    }
    let cause = match finish {
        FinishReason::Stop => StopCause::EndOfSequence,
        _ => StopCause::MaxTokens,
    };
    let (d, reason) = parser.finish(cause);
    all.reasoning_content.push_str(&d.reasoning_content);
    all.content.push_str(&d.content);
    (all, reason.as_str())
}

fn text_of(message: &Value, key: &str) -> String {
    message[key].as_str().unwrap_or("").to_string()
}

fn usage(v: &Value, member: &str) -> u64 {
    v["usage"][member].as_u64().unwrap()
}

fn assert_invalid(status: u16, v: &Value, needle: &str) {
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains(needle),
        "{v}"
    );
}

// ---------------------------------------------------------------------------------------
// The served model.
// ---------------------------------------------------------------------------------------

/// The node serves the architecture its verified `config.json` names, through the
/// Qwen3.8 template and tokenizer; every tokenizer id is sampleable, and the sampling
/// defaults and end-of-sequence ids are `generation_config.json`'s.
#[test]
fn the_dev_model_is_qwen() {
    let model = qwen_loaded();
    assert_eq!(
        model.architecture(),
        eidola_common::architecture::Architecture::Qwen38Dense27B
    );
    assert_eq!(model.format(), ChatFormat::Qwen3_8);
    assert_eq!(model.tokenizer().vocab_size(), 248_077);
    assert_eq!(model.sampleable_vocab_size(), 248_077);
    assert_eq!(model.max_positions(), 4096);
    assert_eq!(model.qwen_config().unwrap().vocab_size, fixture::QWEN_VOCAB);
    let d = model.defaults();
    assert_eq!((d.temperature, d.top_p, d.top_k), (1.0, 0.95, 20));
    assert_eq!(model.tokenizer().eos_token_ids(), [248_046, 248_044]);
}

/// A plain chat: the answer, joined from the stream, is the engine core's greedy
/// generation over the same prompt tokens (without speculation, while the node drafts
/// two tokens a step from the model's one MTP layer), parsed by the format's output
/// parser, with the prompt and completion counted.
#[tokio::test(flavor = "multi_thread")]
async fn a_plain_chat_is_what_the_engine_computes() {
    let node = TestNode::start_qwen(&[]).await;
    let messages = json!([{"role": "user", "content": LONG_TEXT}]);
    let prompt = prompt_tokens(&messages, None);
    let (tokens, finish) = direct_greedy(prompt.clone(), 12);
    let (expected, reason) = expected_output(&tokens, finish);

    let (status, v) = node.chat_json(&body(messages, 12)).await;
    assert_eq!(status, 200, "{v}");
    let message = &v["choices"][0]["message"];
    assert_eq!(
        text_of(message, "reasoning_content"),
        expected.reasoning_content
    );
    assert_eq!(text_of(message, "content"), expected.content);
    assert_eq!(v["choices"][0]["finish_reason"], reason);
    assert_eq!(usage(&v, "prompt_tokens"), prompt.len() as u64);
    assert_eq!(usage(&v, "completion_tokens"), tokens.len() as u64);
    assert_eq!(v["model"], QWEN_MODEL_ID);
    assert!(node.stats().drafted > 0, "the node drafts");
}

/// The node's request carries no thinking switch, effort or `preserve_thinking`, so the
/// template's defaults render every prompt: thinking on (the generation prompt opens the
/// reasoning) at the default effort, whose instructions the template writes.
#[tokio::test(flavor = "multi_thread")]
async fn every_prompt_thinks_at_the_templates_defaults() {
    let options = qwen_options();
    assert!(options.add_generation_prompt);
    assert_eq!(
        (
            options.enable_thinking,
            options.preserve_thinking,
            options.reasoning_effort.as_deref()
        ),
        (None, None, None)
    );
    let messages = json!([{"role": "user", "content": "hi"}]);
    let text = prompt_text(&messages, None);
    assert!(
        text.ends_with("<|im_start|>assistant\n<think>\n"),
        "{text:?}"
    );

    let node = TestNode::start_qwen(&[]).await;
    let (status, v) = node.chat_json(&body(messages.clone(), 2)).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(
        usage(&v, "prompt_tokens"),
        prompt_tokens(&messages, None).len() as u64
    );
}

/// The reasoning of every past turn is kept (the template's `preserve_thinking`
/// default), so a new turn's prompt begins with the earlier turn's whole prompt: the
/// token prefix a client declares cached.
#[test]
fn a_new_turns_prompt_begins_with_the_last_ones() {
    let first = json!([{"role": "user", "content": "First question?"}]);
    let next = json!([
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "reasoning_content": "Some thought.", "content": "An answer."},
        {"role": "user", "content": "A second question?"},
        {"role": "assistant", "reasoning_content": "More thought.", "content": "Another."},
        {"role": "user", "content": "A third?"},
    ]);
    let earlier = prompt_tokens(&first, None);
    let later = prompt_tokens(&next, None);
    assert_eq!(&later[..earlier.len()], &earlier[..]);
    // Both earlier turns' reasoning is in the prompt.
    let text = prompt_text(&next, None);
    assert!(
        text.contains("Some thought.") && text.contains("More thought."),
        "{text}"
    );
}

/// The stream has the node's shape whatever the model: a role chunk, deltas carrying the
/// running usage, a finish chunk, the usage chunk with no choices, `[DONE]`, in frames.
/// A length ending says it is not continuable: the template trims the field a
/// continuation would continue, so no Qwen3.8 answer is.
#[tokio::test(flavor = "multi_thread")]
async fn the_stream_is_the_nodes_and_a_cut_answer_is_never_continuable() {
    let node = TestNode::start_qwen(&[]).await;
    let events = node.chat_stream(&user(LONG_TEXT, 6)).await;
    assert_eq!(events.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let usage_chunk = chunks.last().unwrap();
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 6);
    let finish = &chunks[chunks.len() - 2]["choices"][0];
    assert_eq!(finish["finish_reason"], "length");
    assert_eq!(finish["continuable"], false);
    let mut last = 0;
    for chunk in &chunks[..chunks.len() - 2] {
        let n = chunk["usage"]["completion_tokens"].as_u64().unwrap();
        assert!(n >= last && n <= 6, "{chunk}");
        last = n;
    }
}

// ---------------------------------------------------------------------------------------
// Tools.
// ---------------------------------------------------------------------------------------

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"]
            }
        }
    }])
}

/// The request's tools are rendered by the template, and `tool_choice: none` renders the
/// prompt without them.
#[tokio::test(flavor = "multi_thread")]
async fn tools_are_offered_unless_none_is_chosen() {
    let node = TestNode::start_qwen(&[]).await;
    let messages = json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Find the answer."},
    ]);
    let with = prompt_tokens(&messages, Some(&tools()));
    let without = prompt_tokens(&messages, None);
    assert!(with.len() > without.len());

    let mut b = body(messages.clone(), 2);
    b["tools"] = tools();
    let (status, v) = node.chat_json(&b).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(usage(&v, "prompt_tokens"), with.len() as u64);
    b["tool_choice"] = "none".into();
    let (status, v) = node.chat_json(&b).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(usage(&v, "prompt_tokens"), without.len() as u64);
}

/// A tool round: the assistant's call (arguments as the client sent them, a JSON
/// string, which the template's rule decodes) and its result render as the template
/// renders them, and the turn is answered; the model's own call markup comes back as a
/// tool call of the declared function.
#[tokio::test(flavor = "multi_thread")]
async fn a_tool_round_renders_and_calls_come_back_as_tool_calls() {
    let node = TestNode::start_qwen(&[]).await;
    let messages = json!([
        {"role": "user", "content": "Find the answer."},
        {"role": "assistant", "reasoning_content": "I should look.", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{\"query\": \"answer\", \"limit\": 2}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]);
    let expected = prompt_tokens(&messages, Some(&tools()));
    assert!(prompt_text(&messages, Some(&tools())).contains("<parameter=limit>\n2\n</parameter>"));
    let mut b = body(messages, 4);
    b["tools"] = tools();
    let (status, v) = node.chat_json(&b).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(usage(&v, "prompt_tokens"), expected.len() as u64);

    let text = "reason\n</think>\n\nsummary\n\n<tool_call>\n<function=lookup>\n\
                <parameter=query>\nvalue\n</parameter>\n<parameter=limit>\n2\n</parameter>\n\
                </function>\n</tool_call><|im_end|>";
    let model = qwen_loaded();
    let tokens = model.tokenizer().encode(text).unwrap();
    let declared = eidola_engine_chat::json::parse(&tools().to_string()).unwrap();
    let mut decoder = Decoder::new(
        model.clone(),
        OutputConfig::qwen3_8(None, Some(ToolSchemas::from_tools(&declared))),
        Vec::new(),
        "call_".into(),
    );
    let mut delta = ChatDelta::default();
    let mut finish = None;
    for (i, &t) in tokens.iter().enumerate() {
        let last = i + 1 == tokens.len();
        let step = decoder
            .push(&[t], last.then_some(FinishReason::Stop))
            .unwrap();
        delta
            .reasoning_content
            .push_str(&step.delta.reasoning_content);
        delta.content.push_str(&step.delta.content);
        delta.tool_calls.extend(step.delta.tool_calls);
        finish = finish.or(step.finish);
    }
    assert_eq!(delta.reasoning_content.trim(), "reason");
    assert_eq!(delta.content.trim(), "summary");
    assert_eq!(delta.tool_calls.len(), 1);
    assert_eq!(delta.tool_calls[0].name, "lookup");
    assert_eq!(
        serde_json::from_str::<Value>(&delta.tool_calls[0].arguments).unwrap(),
        json!({"query": "value", "limit": 2})
    );
    assert_eq!(finish.unwrap().as_str(), "tool_calls");
}

// ---------------------------------------------------------------------------------------
// The prefix cache.
// ---------------------------------------------------------------------------------------

fn key(byte: u8) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32])
}

fn declaring(mut b: Value, prefix: CachedPrefix) -> Value {
    b["cached_prefix"] = prefix.to_value();
    b
}

fn cached(v: &Value) -> u64 {
    v["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap()
}

/// A regeneration declaring its own prompt is served from the cache (the recurrent
/// state resumed from a snapshot) with the same answer; under another key it is refused
/// before anything is computed.
#[tokio::test(flavor = "multi_thread")]
async fn a_regeneration_is_served_under_its_key_only() {
    let node = TestNode::start_qwen(&[]).await;
    let mut first = user(&LONG_TEXT.repeat(6), 6);
    first["cache_key"] = key(1).into();
    let (status, answer) = node.chat_json(&first).await;
    assert_eq!(status, 200, "{answer}");
    let prompt = usage(&answer, "prompt_tokens");
    assert!(prompt > 256, "{answer}");
    let prefix = declared_prompt(prompt, 0, 0, WINDOW).unwrap();
    let (status, again) = node.chat_json(&declaring(first.clone(), prefix)).await;
    assert_eq!(status, 200, "{again}");
    assert!(cached(&again) >= u64::from(prefix.cached_tokens), "{again}");
    assert_eq!(again["choices"], answer["choices"]);

    let mut other = declaring(first, prefix);
    other["cache_key"] = key(2).into();
    let wait = std::time::Duration::from_secs(5);
    let before = node
        .wait_for(wait, |s| s.in_engine == 0 && s.kv_keys_held == 0)
        .await;
    let (status, v) = node.chat_json(&other).await;
    assert_eq!(status, 409, "{v}");
    assert_eq!(v["error"]["type"], "cached_prefix_missing", "{v}");
    let after = node.wait_for(wait, |s| s.in_engine == 0).await;
    assert_eq!(after.engine_steps, before.engine_steps, "a step ran for it");
    assert_eq!(after.prefix_refused, before.prefix_refused + 1);
}

/// A new turn (the answer and a new user message) declaring the last turn's prompt, by
/// the shared rule, is served with at least the declared hit and the same answer as
/// undeclared.
#[tokio::test(flavor = "multi_thread")]
async fn a_new_turn_declaring_the_last_ones_prompt_is_served() {
    let node = TestNode::start_qwen(&[]).await;
    let mut first = user(&LONG_TEXT.repeat(6), 6);
    first["cache_key"] = key(3).into();
    let (status, answer) = node.chat_json(&first).await;
    assert_eq!(status, 200, "{answer}");
    let message = &answer["choices"][0]["message"];
    let (reasoning, content) = (
        text_of(message, "reasoning_content"),
        text_of(message, "content"),
    );
    let question = "And what would a second, shorter answer say?";
    let mut messages = first["messages"].as_array().unwrap().clone();
    messages.push(json!({"role": "assistant", "reasoning_content": reasoning, "content": content}));
    messages.push(json!({"role": "user", "content": question}));
    let mut next = body(Value::Array(messages), 6);
    next["cache_key"] = key(3).into();
    let added = (reasoning.len() + content.len() + question.len()) as u64;
    let prefix = declared_prompt(usage(&answer, "prompt_tokens"), added, 2, WINDOW).unwrap();
    let (status, served) = node.chat_json(&declaring(next.clone(), prefix)).await;
    assert_eq!(status, 200, "{served}");
    assert!(
        cached(&served) >= u64::from(prefix.cached_tokens),
        "{served} {prefix:?}"
    );
    let (_, undeclared) = node.chat_json(&next).await;
    assert_eq!(served["choices"], undeclared["choices"]);
}

// ---------------------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------------------

/// Image parts are refused: this model is served text only.
#[tokio::test(flavor = "multi_thread")]
async fn images_are_refused() {
    let node = TestNode::start_qwen(&[]).await;
    let part = body(
        json!([{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]),
        2,
    );
    let (status, v) = node.chat_json(&part).await;
    assert_eq!(status, 400, "{v}");
    let image = body(
        json!([{"role": "user", "content": [{"type": "image", "image": "x"}]}]),
        2,
    );
    let (status, v) = node.chat_json(&image).await;
    assert_eq!(status, 400, "{v}");
}

/// A wrong or missing weights hash is refused on the header alone, and another model's
/// id is not this node's.
#[tokio::test(flavor = "multi_thread")]
async fn the_weights_hash_is_checked() {
    let node = TestNode::start_qwen(&[]).await;
    let url = format!("{}/v1/chat/completions", node.base);
    let r = node
        .client
        .post(&url)
        .bearer_auth(TOKEN)
        .header("x-eidola-weights-sha256", weights_hash())
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 412);
    let r = node
        .client
        .post(&url)
        .bearer_auth(TOKEN)
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 428);
    let (status, v) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 404, "{v}");
}

/// Fields outside the strict subset (a reasoning effort, the template's switches) are
/// refused, and so is a continuation, which the format cannot do exactly.
#[tokio::test(flavor = "multi_thread")]
async fn what_the_subset_or_the_format_does_not_have_is_refused() {
    let node = TestNode::start_qwen(&[]).await;
    for (field, value) in [
        ("reasoning_effort", json!("low")),
        ("chat_template_kwargs", json!({"enable_thinking": false})),
        ("preserve_thinking", json!(false)),
    ] {
        let mut b = user("hi", 2);
        b[field] = value;
        let (status, v) = node.chat_json(&b).await;
        assert_eq!(status, 400, "{field}: {v}");
    }
    let mut cont = body(
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello"},
        ]),
        2,
    );
    cont["continue_final_message"] = true.into();
    cont["add_generation_prompt"] = false.into();
    let (status, v) = node.chat_json(&cont).await;
    assert_invalid(status, &v, "cannot continue");
}

/// What the Qwen3.8 CPU executor cannot run is refused at boot, never panicking on the
/// engine thread: a host tier, which it does not offer, and a length past the model's
/// positions.
#[test]
fn boot_refuses_what_the_executor_cannot_run() {
    let refused = |overrides: &[(&'static str, &str)]| {
        let mut map = qwen_env_map();
        for (k, v) in overrides {
            map.insert(k, v.to_string());
        }
        let config = config_from(&map).unwrap();
        eidola_server_engine::start(config, qwen_loaded())
            .expect_err("refused")
            .to_string()
    };
    let e = refused(&[(env::KV_HOST_BYTES, "1048576")]);
    assert!(e.contains("host tier"), "{e}");
    let e = refused(&[(env::MAX_MODEL_LEN, "4097")]);
    assert!(e.contains("max_position_embeddings"), "{e}");
}

/// A full boot from the directory (the weights hash, `config.json` read as Qwen3.8, the
/// pinned chat files), and a wrong expected hash refused before loading.
#[test]
fn boot_verifies_the_qwen_weights() {
    let config = config_from(&qwen_env_map()).unwrap();
    let node = boot_without_swap(config).unwrap();
    assert_eq!(node.weights_hash, qwen_weights_hash());
    let mut map = qwen_env_map();
    map.insert(env::WEIGHTS_SHA256, "00".repeat(32));
    let e = boot_without_swap(config_from(&map).unwrap()).unwrap_err();
    assert!(e.to_string().contains("weights hash"), "{e}");
}

/// A Qwen3.8 directory missing a file its node reads (the template, here) is refused
/// after the hash, naming the file.
#[test]
fn a_missing_template_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(qwen_model_dir()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name != "chat_template.jinja" && !name.to_string_lossy().starts_with('.') {
            std::fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
        }
    }
    let hash = fixture::weights_hash_of(tmp.path());
    let e = LoadedModel::load(
        tmp.path(),
        &hash,
        eidola_server_engine::config::WeightsStorage::DevWritable,
        eidola_server_engine::config::ExecutorKind::Cpu,
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("chat_template.jinja is missing"),
        "{e}"
    );
}
