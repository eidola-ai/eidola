//! The node serving DeepSeek-V4.1, end to end and in process: real HTTP, the format's
//! compiled-in encoder and pinned tokenizer, the engine core over the DeepSeek CPU
//! executor (two logical devices), on the DeepSeek dev model (`common::fixture`). The
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
use eidola_engine_chat::deepseek_v4_1::{self, EncodeOptions, ReasoningEffort};
use eidola_engine_chat::{
    ChatDelta, ChatFormat, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, StopCause,
};
use eidola_engine_cpu::deepseek::{DsCpuExecutor, DsCpuExecutorConfig};
use eidola_engine_model::deepseek::DsModel;
use eidola_server_engine::config::env;
use eidola_server_engine::model::{LoadedModel, Served};
use eidola_server_engine::pipeline::{Decoder, deepseek_options};
use serde_json::{Value, json};

const LONG_TEXT: &str = "The quick brown fox jumps over the lazy dog while the five boxing \
    wizards jump quickly, and a wizard's job is to vex chumps quickly in fog. Pack my box \
    with five dozen liquor jugs; how vexingly quick daft zebras jump!";

/// The dev model's window in tokens, which bounds what a declaration budgets.
const WINDOW: u64 = 1_024;

fn reference(model: &LoadedModel) -> Arc<DsModel> {
    match model.served() {
        Served::DeepSeek { model: Some(m), .. } => m.clone(),
        _ => panic!("the DeepSeek dev model, loaded for the cpu executor"),
    }
}

/// A greedy request for `messages`.
fn body(messages: Value, max_tokens: u32) -> Value {
    json!({
        "model": DEEPSEEK_MODEL_ID,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    })
}

fn user(text: &str, max_tokens: u32) -> Value {
    body(json!([{"role": "user", "content": text}]), max_tokens)
}

/// The prompt tokens the format encodes for `messages` and `tools` under `options`,
/// encoded here (not by the node).
fn prompt_tokens(messages: &Value, tools: Option<&Value>, options: EncodeOptions) -> Vec<u32> {
    let model = deepseek_loaded();
    let input = ChatInput::from_json(
        &messages.to_string(),
        tools.map(|t| t.to_string()).as_deref(),
    )
    .unwrap();
    let prompt = deepseek_v4_1::encode(&input, options).unwrap();
    model.tokenizer().encode_prompt(&prompt).unwrap()
}

/// Greedy generation through the engine core and the DeepSeek CPU executor, without
/// speculation.
fn direct_greedy(prompt: Vec<u32>, max_tokens: u32) -> (Vec<u32>, FinishReason) {
    let model = deepseek_loaded();
    let exec = DsCpuExecutor::new(
        reference(&model),
        DsCpuExecutorConfig {
            block_size: 16,
            num_blocks: 256,
            host_blocks: 0,
            num_state_slots: 4,
            max_model_len: 1024,
            buckets: vec![Bucket {
                max_seqs: 4,
                max_tokens: 256,
            }],
            draft_width: 0,
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

/// What the format's output parser makes of `tokens` in thinking mode.
fn expected_output(tokens: &[u32], finish: FinishReason) -> (ChatDelta, &'static str) {
    let model = deepseek_loaded();
    let mut parser = OutputParser::new(
        OutputConfig::deepseek_v4_1(true, None),
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
/// DeepSeek format and tokenizer; the image token and the vision ids after it are never
/// sampled.
#[test]
fn the_dev_model_is_deepseek() {
    let model = deepseek_loaded();
    assert_eq!(
        model.architecture(),
        eidola_common::architecture::Architecture::DeepSeekV41Flash
    );
    assert_eq!(model.format(), ChatFormat::DeepSeekV4_1);
    assert_eq!(model.tokenizer().vocab_size(), fixture::DEEPSEEK_VOCAB);
    assert_eq!(model.sampleable_vocab_size(), fixture::DEEPSEEK_IMAGE_TOKEN);
    assert_eq!(model.max_positions(), 1024);
    // No generation_config.json: the defaults.
    let d = model.defaults();
    assert_eq!((d.temperature, d.top_p, d.top_k), (1.0, 1.0, 0));
}

/// The Engram layers hash over the map derived from the tokenizer: the dev model's shards
/// carry none, as the released checkpoint's do not, and the rows the loaded model reads
/// are those of a hasher over the derived map.
#[test]
fn the_engram_map_is_derived_from_the_tokenizer() {
    use eidola_engine_model::deepseek::engram::EngramHasher;
    use eidola_server_engine::model::ENGRAM_TOKEN_MAP_TENSOR;

    let model = deepseek_loaded();
    assert!(model.store().get(ENGRAM_TOKEN_MAP_TENSOR).is_err());
    let reference = reference(&model);
    let spec = reference.config().engram.as_ref().expect("Engram layers");
    let derived = deepseek_v4_1::compressed_vocab(model.tokenizer()).unwrap();
    assert_eq!(derived.classes, fixture::DEEPSEEK_COMPRESSED_VOCAB);
    assert_eq!(spec.compressed_vocab, derived.classes);
    // " The", "the" and "THE" are one token each, of one class.
    let words: Vec<u32> = [" The", "the", "THE"]
        .iter()
        .map(|t| match model.tokenizer().encode(t).unwrap()[..] {
            [id] => id,
            ref ids => panic!("{t:?} is {ids:?}"),
        })
        .collect();
    assert!(
        words
            .iter()
            .all(|&id| derived.map[id as usize] == derived.map[words[0] as usize])
    );
    let hasher = EngramHasher::new(spec, reference.config().vocab_size, derived.map).unwrap();
    let history: Vec<u32> = words
        .into_iter()
        .chain([0, 1, 2, 77_666, 128_799, 129_279])
        .collect();
    for slot in 0..spec.layers.len() {
        for position in 0..history.len() {
            let seen = &history[..=position];
            assert_eq!(
                reference.engram_rows(slot, seen, position).unwrap(),
                hasher.rows(&spec.layers[slot], seen, position)
            );
        }
    }
}

/// A plain chat: the answer, joined from the stream, is the engine core's greedy
/// generation over the same prompt tokens (without speculation, while the node drafts
/// two tokens a step), parsed by the format's output parser, with the prompt and
/// completion counted.
#[tokio::test(flavor = "multi_thread")]
async fn a_plain_chat_is_what_the_engine_computes() {
    let node = TestNode::start_deepseek(&[]).await;
    let messages = json!([{"role": "user", "content": LONG_TEXT}]);
    let prompt = prompt_tokens(&messages, None, deepseek_options());
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
    assert_eq!(v["model"], DEEPSEEK_MODEL_ID);
}

/// The node's request carries neither a thinking switch nor a reasoning effort, so every
/// prompt is the format's request default: thinking on, effort `high`. It begins with
/// the BOS in its text, ends with the thinking-mode generation header, and is a
/// different prompt from chat mode's.
#[tokio::test(flavor = "multi_thread")]
async fn every_prompt_thinks_at_the_default_effort() {
    let options = deepseek_options();
    assert!(options.thinking);
    assert_eq!(options.reasoning_effort, ReasoningEffort::HIGH);
    assert_eq!(
        options,
        EncodeOptions::from_request(
            Some(true),
            Some(&eidola_engine_chat::json::parse("\"high\"").unwrap())
        )
        .unwrap()
    );

    let messages = json!([{"role": "user", "content": "hi"}]);
    let input = ChatInput::from_json(&messages.to_string(), None).unwrap();
    let thinking = deepseek_v4_1::encode(&input, options).unwrap();
    let chat = deepseek_v4_1::encode(&input, EncodeOptions::new(false)).unwrap();
    assert!(thinking.text().starts_with(deepseek_v4_1::BOS_TOKEN));
    assert!(
        thinking.text().ends_with("<｜Assistant｜><think>"),
        "{}",
        thinking.text()
    );
    assert!(chat.text().ends_with("<｜Assistant｜></think>"));

    let node = TestNode::start_deepseek(&[]).await;
    let (status, v) = node.chat_json(&body(messages.clone(), 2)).await;
    assert_eq!(status, 200, "{v}");
    let on = prompt_tokens(&messages, None, options);
    let off = prompt_tokens(&messages, None, EncodeOptions::new(false));
    assert_eq!(usage(&v, "prompt_tokens"), on.len() as u64);
    assert_ne!(on, off);
    // The BOS is the prompt's own text: one, at the start.
    let bos = deepseek_loaded()
        .tokenizer()
        .encode(deepseek_v4_1::BOS_TOKEN)
        .unwrap();
    assert_eq!(&on[..bos.len()], &bos[..]);
}

/// The stream has the node's shape whatever the model: a role chunk, deltas carrying the
/// running usage, a finish chunk, the usage chunk with no choices, `[DONE]`, in frames.
/// A length ending says it is not continuable: no DeepSeek-V4.1 answer is.
#[tokio::test(flavor = "multi_thread")]
async fn the_stream_is_the_nodes_and_a_cut_answer_is_never_continuable() {
    let node = TestNode::start_deepseek(&[]).await;
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

/// The request's tools go on the system message the format reads them from, and
/// `tool_choice: none` encodes the prompt without them.
#[tokio::test(flavor = "multi_thread")]
async fn tools_are_offered_on_the_system_message() {
    let node = TestNode::start_deepseek(&[]).await;
    let messages = json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Find the answer."},
    ]);
    let with = prompt_tokens(&messages, Some(&tools()), deepseek_options());
    let without = prompt_tokens(&messages, None, deepseek_options());
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
/// string, which the encoder decodes itself) and its result are encoded as the format
/// encodes them, and the turn is answered.
#[tokio::test(flavor = "multi_thread")]
async fn a_tool_round_is_encoded_as_the_format_does() {
    let node = TestNode::start_deepseek(&[]).await;
    let messages = json!([
        {"role": "user", "content": "Find the answer."},
        {"role": "assistant", "reasoning_content": "I should look.", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{\"query\": \"answer\", \"limit\": 2}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]);
    let expected = prompt_tokens(&messages, Some(&tools()), deepseek_options());
    let mut b = body(messages, 4);
    b["tools"] = tools();
    let (status, v) = node.chat_json(&b).await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(usage(&v, "prompt_tokens"), expected.len() as u64);
    // The model's own call markup comes back as a tool call.
    let text = format!(
        "reason</think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n\
         <｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n\
         <｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n\
         </｜DSML｜ invoke>\n</｜DSML｜ calls>{}",
        deepseek_v4_1::EOS_TOKEN
    );
    let model = deepseek_loaded();
    let tokens = model.tokenizer().encode(&text).unwrap();
    let declared = eidola_engine_chat::json::parse(&tools().to_string()).unwrap();
    let schemas = deepseek_v4_1::tool_schemas(&declared).unwrap();
    let mut decoder = Decoder::new(
        model.clone(),
        OutputConfig::deepseek_v4_1(true, Some(schemas)),
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
    assert_eq!(delta.reasoning_content, "reason");
    assert_eq!(delta.content, "summary");
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

/// A regeneration declaring its own prompt is served from the cache with the same
/// answer; under another key it is refused before anything is computed.
#[tokio::test(flavor = "multi_thread")]
async fn a_cached_prefix_is_served_under_its_key_only() {
    let node = TestNode::start_deepseek(&[]).await;
    let mut first = user(&LONG_TEXT.repeat(6), 6);
    first["cache_key"] = key(1).into();
    let (status, answer) = node.chat_json(&first).await;
    assert_eq!(status, 200, "{answer}");
    let prompt = usage(&answer, "prompt_tokens");
    assert!(prompt > 256, "{answer}");
    let prefix = declared_prompt(prompt, 0, 0, WINDOW).unwrap();
    let (status, again) = node.chat_json(&declaring(first.clone(), prefix)).await;
    assert_eq!(status, 200, "{again}");
    let cached = again["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert!(cached >= u64::from(prefix.cached_tokens), "{again}");
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

// ---------------------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------------------

/// An image part is refused by the strict subset, and text spelling the format's image
/// placeholder by the encoder: this model is text only.
#[tokio::test(flavor = "multi_thread")]
async fn images_are_refused() {
    let node = TestNode::start_deepseek(&[]).await;
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
    let spelled = user(&format!("look: {}", deepseek_v4_1::IMAGE_PLACEHOLDER), 2);
    let (status, v) = node.chat_json(&spelled).await;
    assert_invalid(status, &v, "unsupported content");
}

/// A wrong or missing weights hash is refused on the header alone.
#[tokio::test(flavor = "multi_thread")]
async fn the_weights_hash_is_checked() {
    let node = TestNode::start_deepseek(&[]).await;
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
    // The MiMo model's id is not this node's.
    let (status, v) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 404, "{v}");
}

/// Fields outside the strict subset (a reasoning effort, a thinking switch, a
/// `developer` message) are refused; so is a continuation, which the format cannot do,
/// and a request whose last message is the assistant's, which the format does not
/// answer.
#[tokio::test(flavor = "multi_thread")]
async fn what_the_subset_or_the_format_does_not_have_is_refused() {
    let node = TestNode::start_deepseek(&[]).await;
    for (field, value) in [
        ("reasoning_effort", json!("low")),
        ("chat_template_kwargs", json!({"thinking": false})),
        ("thinking", json!(false)),
    ] {
        let mut b = user("hi", 2);
        b[field] = value;
        let (status, v) = node.chat_json(&b).await;
        assert_eq!(status, 400, "{field}: {v}");
    }
    let developer = body(json!([{"role": "developer", "content": "hi"}]), 2);
    let (status, v) = node.chat_json(&developer).await;
    assert_eq!(status, 400, "{v}");

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

    let answered = body(
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello"},
        ]),
        2,
    );
    let (status, v) = node.chat_json(&answered).await;
    assert_invalid(status, &v, "last message");
}

/// What the DeepSeek executor cannot run is refused at boot, never panicking on the
/// engine thread: more drafts than the drafter's slots, a block size the pooling ratio
/// does not divide, a host tier.
#[test]
fn boot_refuses_what_the_executor_cannot_run() {
    let refused = |overrides: &[(&'static str, &str)]| {
        let mut map = deepseek_env_map();
        for (k, v) in overrides {
            map.insert(k, v.to_string());
        }
        let config = config_from(&map).unwrap();
        eidola_server_engine::start(config, deepseek_loaded())
            .expect_err("refused")
            .to_string()
    };
    let e = refused(&[(env::DRAFT_TOKENS, "5")]);
    assert!(e.contains("draft slots"), "{e}");
    let e = refused(&[(env::KV_BLOCK_SIZE, "5"), (env::PREFIX_CACHE, "false")]);
    assert!(e.contains("pooling ratio"), "{e}");
    let e = refused(&[(env::KV_HOST_BYTES, "1048576")]);
    assert!(e.contains("host tier"), "{e}");
    let e = refused(&[(env::MAX_MODEL_LEN, "1025")]);
    assert!(e.contains("max_position_embeddings"), "{e}");
}

/// A full boot from the directory, and a wrong expected hash refused before loading.
#[test]
fn boot_verifies_the_deepseek_weights() {
    let config = config_from(&deepseek_env_map()).unwrap();
    let node = boot_without_swap(config).unwrap();
    assert_eq!(node.weights_hash, deepseek_weights_hash());
    let mut map = deepseek_env_map();
    map.insert(env::WEIGHTS_SHA256, "00".repeat(32));
    let e = boot_without_swap(config_from(&map).unwrap()).unwrap_err();
    assert!(e.to_string().contains("weights hash"), "{e}");
}
