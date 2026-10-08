//! The node end to end, in process: real HTTP, the real chat pipeline (pinned MiMo
//! template and tokenizer), the engine core over the CPU executor, on the synthetic dev
//! model (`common::fixture`). The model's output is gibberish; these tests assert
//! structure, refusals, and token-level equality with the engine driven directly.

mod common;

use std::time::Duration;

use common::*;
use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_chat::{
    ChatDelta, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, RenderOptions, StopCause,
};
use eidola_engine_cpu::{CpuExecutor, CpuExecutorConfig, MtpHidden};
use eidola_server_engine::api::MAX_STOP_BYTES;
use eidola_server_engine::config::env;
use eidola_server_engine::worker::Stats;
use futures_util::StreamExt;
use serde_json::{Value, json};

const LONG_TEXT: &str = "The quick brown fox jumps over the lazy dog while the five boxing \
    wizards jump quickly, and a wizard's job is to vex chumps quickly in fog. Pack my box \
    with five dozen liquor jugs; how vexingly quick daft zebras jump!";

// ---------------------------------------------------------------------------------------
// The engine driven directly: the oracle for what the node must return.
// ---------------------------------------------------------------------------------------

/// The prompt tokens for one user message, rendered and encoded here (not by the node).
fn prompt_tokens(text: &str) -> Vec<u32> {
    let model = loaded();
    let messages = json!([{"role": "user", "content": text}]).to_string();
    let input = ChatInput::from_json(&messages, None).unwrap();
    let prompt = model
        .template()
        .render(
            &input,
            RenderOptions {
                add_generation_prompt: true,
                enable_thinking: None,
            },
        )
        .unwrap();
    model.tokenizer().encode(&prompt).unwrap()
}

/// Greedy generation through the engine core and CPU executor, without speculation.
fn direct_greedy(prompt: Vec<u32>, max_tokens: u32) -> (Vec<u32>, FinishReason) {
    let model = loaded();
    let exec = CpuExecutor::new(
        model.reference().unwrap().clone(),
        CpuExecutorConfig {
            block_size: 16,
            num_blocks: 256,
            num_state_slots: 4,
            max_model_len: 1024,
            buckets: vec![Bucket {
                max_seqs: 4,
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

/// What the chat layer makes of `tokens` (the EOS that ends a `Stop` is not text).
fn expected_output(tokens: &[u32], finish: FinishReason) -> (ChatDelta, &'static str) {
    let model = loaded();
    let mut parser = OutputParser::new(OutputConfig::new(None, None), PrefixedCallIds("c".into()));
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

// ---------------------------------------------------------------------------------------
// Authentication and the weights-hash check.
// ---------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn gateway_token_is_required() {
    let node = TestNode::start(&[]).await;
    let body = request("hi", 4);
    let url = format!("{}/v1/chat/completions", node.base);
    for auth in [
        None,
        Some("Bearer wrong"),
        Some(TOKEN),
        Some("Basic dGVzdA=="),
    ] {
        let mut r = node
            .client
            .post(&url)
            .header("x-eidola-weights-sha256", weights_hash())
            .json(&body);
        if let Some(a) = auth {
            r = r.header("authorization", a);
        }
        let r = r.send().await.unwrap();
        assert_eq!(r.status().as_u16(), 401, "{auth:?}");
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["error"]["type"], "authentication_error");
    }
    let r = node
        .client
        .get(format!("{}/v1/engine/info", node.base))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 401);
    // Authentication comes before the weights check: an unauthenticated caller learns
    // nothing about the weights.
    let r = node
        .client
        .post(&url)
        .header("x-eidola-weights-sha256", "00".repeat(32))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 401);
    // Health is unauthenticated and content-free.
    let r = node
        .client
        .get(format!("{}/healthz", node.base))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    // A dev-writable node says so, even in its health check.
    assert_eq!(r.text().await.unwrap(), "ok; weights-storage=dev-writable");
    assert_eq!(node.stats().submitted, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn info_reports_the_node_identity() {
    let node = TestNode::start(&[]).await;
    let r = node
        .client
        .get(format!("{}/v1/engine/info", node.base))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["model"], MODEL_ID);
    assert_eq!(v["weights_sha256"], weights_hash());
    assert_eq!(v["weights_storage"], "dev-writable");
    assert_eq!(v["executor"], "cpu");
    // The key is always present; the CPU executor has no device.
    assert_eq!(v.get("device"), Some(&Value::Null));
    assert_eq!(v["build"]["crate"], "eidola-server-engine");
    assert_eq!(v["build"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(v["build"].get("git_sha").is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn weights_hash_mismatch_is_refused_before_admission() {
    let node = TestNode::start(&[]).await;
    let url = format!("{}/v1/chat/completions", node.base);
    let mut other = weights_hash().to_string();
    other.replace_range(..1, if other.starts_with('0') { "1" } else { "0" });
    for body in [request("hi", 4).to_string(), "this is not json".to_string()] {
        let r = node
            .client
            .post(&url)
            .bearer_auth(TOKEN)
            .header("x-eidola-weights-sha256", &other)
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        // Refused on the header alone: the body (even an unparseable one) is never read.
        assert_eq!(r.status().as_u16(), 412);
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["error"]["type"], "weights_hash_mismatch");

        let r = node
            .client
            .post(&url)
            .bearer_auth(TOKEN)
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status().as_u16(), 428);
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["error"]["type"], "weights_hash_required");
    }
    // Nothing reached the engine: no submission, no step, no admission slot held.
    assert_eq!(node.stats(), Stats::default());

    // The same request with the right hash (any case) is served.
    let r = node
        .client
        .post(&url)
        .bearer_auth(TOKEN)
        .header(
            "x-eidola-weights-sha256",
            weights_hash().to_ascii_uppercase(),
        )
        .json(&request("hi", 2))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(node.stats().submitted, 1);
}

// ---------------------------------------------------------------------------------------
// The strict request subset.
// ---------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn wrong_model_is_refused() {
    let node = TestNode::start(&[]).await;
    let mut body = request("hi", 4);
    body["model"] = "some-other-model".into();
    let (status, v) = node.chat_json(&body).await;
    assert_eq!(status, 404);
    assert_eq!(v["error"]["type"], "model_not_found");
    assert_eq!(node.stats(), Stats::default());
}

#[tokio::test(flavor = "multi_thread")]
async fn strict_schema_refuses_what_it_does_not_name() {
    let node = TestNode::start(&[]).await;
    let base = request("hi", 4);
    let with = |k: &str, v: Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    let cases = [
        with("n", 2.into()),
        with("seed", 7.into()),
        with("logprobs", true.into()),
        with("chat_template_kwargs", json!({"enable_thinking": false})),
        with(
            "messages",
            json!([{"role": "user", "content": "hi", "extra": 1}]),
        ),
        with("messages", json!([{"role": "developer", "content": "hi"}])),
        with("messages", json!([])),
        with(
            "messages",
            json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]),
        ),
        with("stream_options", json!({"include_usage": true, "other": 1})),
        with("tool_choice", "required".into()),
        with(
            "tool_choice",
            json!({"type": "function", "function": {"name": "f"}}),
        ),
        with("stop", json!(["a", "b", "c", "d", "e"])),
        with("stop", "".into()),
        // One byte over the per-sequence cap, as one sequence or among four.
        with("stop", "x".repeat(MAX_STOP_BYTES + 1).into()),
        with(
            "stop",
            json!(["a", "b", "c", "é".repeat(MAX_STOP_BYTES / 2 + 1)]),
        ),
        with("max_completion_tokens", 0.into()),
        with("top_p", 0.0.into()),
        with("temperature", (-1.0).into()),
        with("cache_key", "too-short".into()),
        with("cache_key", "A".repeat(43).replace('A', "+").into()),
        with("cache_key", format!("{}=", "A".repeat(43)).into()),
        with("cache_key", 5.into()),
        // A valid key, then a field that fails to parse.
        {
            let mut b = with("cache_key", "A".repeat(43).into());
            b["seed"] = 7.into();
            b
        },
    ];
    for body in &cases {
        let (status, v) = node.chat_json(body).await;
        assert_eq!(status, 400, "{body}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{body}");
    }
    // A refused request never reaches the engine.
    assert_eq!(node.stats(), Stats::default());

    // The fields the gateway forwards today, explicit nulls included, are accepted.
    let ok = json!({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}], "name": "u"},
        ],
        "max_completion_tokens": 2,
        "temperature": null,
        "top_p": null,
        "stream": false,
        "stop": null,
        "cache_key": "A".repeat(43),
    });
    let (status, v) = node.chat_json(&ok).await;
    assert_eq!(status, 200, "{v}");

    // Four sequences of exactly the cap are accepted.
    let mut at_cap = request("hi", 2);
    at_cap["stop"] =
        json!(["\u{1}", "\u{2}", "\u{3}", "é"].map(|c: &str| c.repeat(MAX_STOP_BYTES / c.len())));
    let (status, v) = node.chat_json(&at_cap).await;
    assert_eq!(status, 200, "{v}");
}

// ---------------------------------------------------------------------------------------
// Responses.
// ---------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_matches_the_engine_token_for_token() {
    let node = TestNode::start(&[]).await;
    for (text, max) in [("Hello there", 12), (LONG_TEXT, 20), ("x", 1)] {
        let prompt = prompt_tokens(text);
        let n_prompt = prompt.len();
        let (tokens, finish) = direct_greedy(prompt, max);
        let (expected, reason) = expected_output(&tokens, finish);

        let (status, v) = node.chat_json(&request(text, max)).await;
        assert_eq!(status, 200, "{v}");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], MODEL_ID);
        assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert!(v["created"].as_u64().unwrap() > 0);
        let choice = &v["choices"][0];
        assert_eq!(v["choices"].as_array().unwrap().len(), 1);
        assert_eq!(choice["index"], 0);
        assert_eq!(choice["finish_reason"], reason);
        let message = &choice["message"];
        assert_eq!(message["role"], "assistant");
        assert_eq!(
            text_of(message, "reasoning_content"),
            expected.reasoning_content
        );
        assert_eq!(text_of(message, "content"), expected.content);
        assert!(message.get("tool_calls").is_none());

        let usage = &v["usage"];
        assert_eq!(usage["prompt_tokens"], n_prompt);
        assert_eq!(usage["completion_tokens"], tokens.len());
        assert_eq!(usage["total_tokens"], n_prompt + tokens.len());
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0);
    }
    let s = node
        .wait_for(Duration::from_secs(5), |s| s.finished == 3)
        .await;
    assert_eq!((s.submitted, s.cancelled, s.in_engine), (3, 0, 0));
}

fn parse_chunks(events: &[String]) -> (Vec<Value>, bool) {
    let done = events.last().map(String::as_str) == Some("[DONE]");
    let chunks = events
        .iter()
        .filter(|e| e.as_str() != "[DONE]")
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    (chunks, done)
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_shapes_text_and_usage() {
    let node = TestNode::start(&[]).await;
    let mut body = request(LONG_TEXT, 16);
    let (_, plain) = node.chat_json(&body).await;

    body["stream"] = true.into();
    body["stream_options"] = json!({"include_usage": true});
    let events = node.chat_stream(&body).await;
    let (chunks, done) = parse_chunks(&events);
    assert!(done, "the stream ends with [DONE]");
    let id = chunks[0]["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-"));
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], id);
        assert_eq!(c["model"], MODEL_ID);
    }
    // Role first.
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    // Then deltas; exactly one finish; then the usage chunk with no choices.
    let usage_chunk = chunks.last().unwrap();
    assert_eq!(usage_chunk["choices"], json!([]));
    let finish_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|c| {
            c["choices"]
                .get(0)
                .is_some_and(|ch| !ch["finish_reason"].is_null())
        })
        .collect();
    assert_eq!(finish_chunks.len(), 1);
    let finish = &finish_chunks[0]["choices"][0];
    assert_eq!(
        finish["finish_reason"],
        plain["choices"][0]["finish_reason"]
    );
    assert_eq!(finish["delta"], json!({}));
    let (mut reasoning, mut content) = (String::new(), String::new());
    for c in &chunks[1..chunks.len() - 2] {
        let delta = &c["choices"][0]["delta"];
        assert!(delta.get("role").is_none());
        let keys: Vec<&String> = delta.as_object().unwrap().keys().collect();
        assert!(!keys.is_empty(), "no empty deltas");
        assert!(
            keys.iter()
                .all(|k| ["reasoning_content", "content", "tool_calls"].contains(&k.as_str()))
        );
        reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
        content.push_str(delta["content"].as_str().unwrap_or(""));
    }
    let message = &plain["choices"][0]["message"];
    assert_eq!(reasoning, text_of(message, "reasoning_content"));
    assert_eq!(content, text_of(message, "content"));
    assert_eq!(usage_chunk["usage"], plain["usage"]);

    // Without include_usage there is no usage chunk.
    body["stream_options"] = Value::Null;
    let (chunks, done) = parse_chunks(&node.chat_stream(&body).await);
    assert!(done);
    assert!(chunks.iter().all(|c| c.get("usage").is_none()));
    assert!(!chunks.last().unwrap()["choices"][0]["finish_reason"].is_null());
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_sequences_truncate_the_output_and_end_the_request() {
    let node = TestNode::start(&[]).await;
    let max = 32;
    let (_, full) = node.chat_json(&request(LONG_TEXT, max)).await;
    let message = &full["choices"][0]["message"];
    let full_text = text_of(message, "reasoning_content") + &text_of(message, "content");
    // A stop sequence taken from the middle of what the model says.
    let chars: Vec<char> = full_text.chars().collect();
    assert!(chars.len() >= 8, "{full_text:?}");
    let stop: String = chars[chars.len() / 2..chars.len() / 2 + 3].iter().collect();
    let cut = full_text.find(&stop).unwrap();

    let mut body = request(LONG_TEXT, max);
    body["stop"] = json!([stop, "\u{1}never\u{1}"]);
    for stream in [false, true] {
        let (text, finish, usage) = if stream {
            body["stream"] = true.into();
            body["stream_options"] = json!({"include_usage": true});
            let (chunks, _) = parse_chunks(&node.chat_stream(&body).await);
            let mut text = String::new();
            let mut finish = Value::Null;
            for c in &chunks {
                if let Some(ch) = c["choices"].get(0) {
                    text.push_str(ch["delta"]["reasoning_content"].as_str().unwrap_or(""));
                    text.push_str(ch["delta"]["content"].as_str().unwrap_or(""));
                    if !ch["finish_reason"].is_null() {
                        finish = ch["finish_reason"].clone();
                    }
                }
            }
            (text, finish, chunks.last().unwrap()["usage"].clone())
        } else {
            let (status, v) = node.chat_json(&body).await;
            assert_eq!(status, 200);
            let m = &v["choices"][0]["message"];
            (
                text_of(m, "reasoning_content") + &text_of(m, "content"),
                v["choices"][0]["finish_reason"].clone(),
                v["usage"].clone(),
            )
        };
        assert_eq!(text, full_text[..cut], "stream={stream}");
        assert_eq!(finish, "stop");
        let used = usage["completion_tokens"].as_u64().unwrap();
        assert!(used >= 1 && used < max as u64, "{used}");
    }
    // Both stopped requests were cancelled in the engine, not run to max_tokens.
    let s = node
        .wait_for(Duration::from_secs(5), |s| s.in_engine == 0)
        .await;
    assert_eq!(s.cancelled, 2, "{s:?}");
}

/// A long stop sequence that the output never begins does not delay the stream: text is
/// released as it is generated.
#[tokio::test(flavor = "multi_thread")]
async fn a_long_stop_sequence_does_not_hold_back_the_stream() {
    let node = TestNode::start(&[]).await;
    let mut body = request(LONG_TEXT, 24);
    let (_, plain) = node.chat_json(&body).await;
    body["stop"] = json!(["\u{1}".repeat(eidola_server_engine::api::MAX_STOP_BYTES)]);
    body["stream"] = true.into();
    let (chunks, _) = parse_chunks(&node.chat_stream(&body).await);
    let deltas: Vec<&Value> = chunks
        .iter()
        .filter_map(|c| c["choices"].get(0))
        .filter(|ch| ch["finish_reason"].is_null() && ch["delta"].get("role").is_none())
        .collect();
    assert!(deltas.len() > 4, "text streamed in {} deltas", deltas.len());
    let text: String = deltas
        .iter()
        .map(|d| {
            d["delta"]["reasoning_content"]
                .as_str()
                .unwrap_or("")
                .to_string()
                + d["delta"]["content"].as_str().unwrap_or("")
        })
        .collect();
    let m = &plain["choices"][0]["message"];
    assert_eq!(
        text,
        text_of(m, "reasoning_content") + &text_of(m, "content")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn max_tokens_ends_with_length() {
    let node = TestNode::start(&[]).await;
    let (tokens, finish) = direct_greedy(prompt_tokens(LONG_TEXT), 5);
    let (_, v) = node.chat_json(&request(LONG_TEXT, 5)).await;
    assert_eq!(v["usage"]["completion_tokens"], tokens.len());
    let expected = if finish == FinishReason::Length {
        "length"
    } else {
        "stop"
    };
    assert_eq!(v["choices"][0]["finish_reason"], expected);
}

#[tokio::test(flavor = "multi_thread")]
async fn client_disconnect_cancels_in_the_engine() {
    let node = TestNode::start(&[]).await;
    let mut body = request("Tell me a very long story.", 1000);
    body["stream"] = true.into();
    let response = node.chat(&body).send().await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let mut stream = response.bytes_stream();
    // Read until generated text arrives, so the request is running in the engine.
    let mut seen = String::new();
    while !seen.contains("reasoning_content") && !seen.contains("\"content\":\"") {
        seen.push_str(&String::from_utf8_lossy(
            &stream.next().await.unwrap().unwrap(),
        ));
    }
    node.wait_for(Duration::from_secs(5), |s| s.in_engine == 1)
        .await;
    drop(stream);

    let s = node
        .wait_for(Duration::from_secs(10), |s| s.in_engine == 0)
        .await;
    assert_eq!((s.cancelled, s.finished), (1, 0), "{s:?}");
    // It stopped long before its 1000 tokens: steps stay far below that.
    assert!(s.engine_steps < 500, "{s:?}");
    let steps = s.engine_steps;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        node.stats().engine_steps,
        steps,
        "nothing runs after the cancel"
    );

    // The node keeps serving.
    let (status, _) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 200);
}

/// A request still waiting for a seat produces no events, so only dropping its guard can
/// cancel it.
#[tokio::test(flavor = "multi_thread")]
async fn disconnect_while_queued_cancels_in_the_engine() {
    let node = TestNode::start(&[(env::MAX_SEQS, "1")]).await;
    let mut long = request("Tell me a very long story.", 1000);
    long["stream"] = true.into();
    let running = node.chat(&long).send().await.unwrap();
    let mut running = running.bytes_stream();
    let mut seen = String::new();
    while !seen.contains("reasoning_content") && !seen.contains("\"content\":\"") {
        seen.push_str(&String::from_utf8_lossy(
            &running.next().await.unwrap().unwrap(),
        ));
    }
    let queued = node.chat(&long).send().await.unwrap();
    assert_eq!(queued.status().as_u16(), 200);
    let mut queued = queued.bytes_stream();
    let role = String::from_utf8_lossy(&queued.next().await.unwrap().unwrap()).to_string();
    assert!(role.contains("\"role\":\"assistant\""), "{role}");
    node.wait_for(Duration::from_secs(5), |s| s.in_engine == 2)
        .await;
    drop(queued);
    let s = node
        .wait_for(Duration::from_secs(10), |s| s.cancelled == 1)
        .await;
    assert_eq!(s.in_engine, 1, "the running request is untouched: {s:?}");
    drop(running);
    node.wait_for(Duration::from_secs(10), |s| s.in_engine == 0)
        .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn admission_is_bounded() {
    let node = TestNode::start(&[(env::MAX_REQUESTS, "1")]).await;
    let mut long = request("Tell me a very long story.", 1000);
    long["stream"] = true.into();
    let response = node.chat(&long).send().await.unwrap();
    let mut stream = response.bytes_stream();
    stream.next().await.unwrap().unwrap();

    let (status, v) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 503);
    assert_eq!(v["error"]["type"], "overloaded");
    // The overloaded request was never submitted.
    let s = node
        .wait_for(Duration::from_secs(5), |s| s.submitted >= 1)
        .await;
    assert_eq!(s.submitted, 1);

    drop(stream);
    node.wait_for(Duration::from_secs(10), |s| s.in_engine == 0)
        .await;
    // The slot is released with the request: the next one is admitted.
    let mut admitted = false;
    for _ in 0..100 {
        let (status, _) = node.chat_json(&request("hi", 2)).await;
        if status == 200 {
            admitted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(admitted);
}

/// Bodies being read or parsed are bounded before admission: with every read slot held
/// by an upload that never finishes, the next request is refused `overloaded` without
/// its body being read, and the stalled upload holds no admission slot.
#[tokio::test(flavor = "multi_thread")]
async fn reading_bodies_is_bounded_before_admission() {
    use std::io::Write;
    let node = TestNode::start(&[(env::MAX_REQUESTS, "1")]).await;
    let addr = node.base.strip_prefix("http://").unwrap();
    let mut stalled = std::net::TcpStream::connect(addr).unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: {addr}\r\nauthorization: Bearer {TOKEN}\r\n\
         x-eidola-weights-sha256: {}\r\ncontent-type: application/json\r\n\
         content-length: 1000000\r\n\r\n{{\"model\":",
        weights_hash()
    );
    stalled.write_all(head.as_bytes()).unwrap();

    // Once the stalled upload holds the only read slot, a complete request is refused.
    let mut refused = None;
    for _ in 0..500 {
        let (status, v) = node.chat_json(&request("hi", 1)).await;
        if status == 503 {
            refused = Some(v);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let v = refused.expect("a request was refused while every read slot was held");
    assert_eq!(v["error"]["type"], "overloaded");
    assert_eq!(
        node.admission.in_flight(),
        0,
        "reading holds no admission slot"
    );

    // The upload going away frees its read slot.
    drop(stalled);
    let mut admitted = false;
    for _ in 0..500 {
        let (status, _) = node.chat_json(&request("hi", 1)).await;
        if status == 200 {
            admitted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(admitted);
}

#[tokio::test(flavor = "multi_thread")]
async fn cache_key_reuses_the_prefix_only_for_the_same_key() {
    let node = TestNode::start(&[]).await;
    use base64::Engine as _;
    let b64 = |byte: u8| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32]);
    let (key_a, key_b) = (b64(1), b64(2));
    let keyed = |key: Option<&str>| {
        let mut b = request(LONG_TEXT, 4);
        if let Some(k) = key {
            b["cache_key"] = k.into();
        }
        b
    };
    let cached = |v: &Value| {
        v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
    };

    let (_, first) = node.chat_json(&keyed(Some(&key_a))).await;
    assert_eq!(cached(&first), 0);
    let prompt = first["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(prompt > 32, "the prompt spans several KV blocks ({prompt})");

    let (_, second) = node.chat_json(&keyed(Some(&key_a))).await;
    let hit = cached(&second);
    assert!(hit > 0 && hit < prompt, "{hit} of {prompt}");
    assert_eq!(hit % 16, 0, "hits are whole blocks");
    // Reuse is exact: the same greedy output.
    assert_eq!(second["choices"], first["choices"]);

    let (_, other_key) = node.chat_json(&keyed(Some(&key_b))).await;
    assert_eq!(cached(&other_key), 0, "another key shares nothing");
    let (_, no_key) = node.chat_json(&keyed(None)).await;
    assert_eq!(cached(&no_key), 0, "no key, no reuse");
    let (_, no_key_again) = node.chat_json(&keyed(None)).await;
    assert_eq!(
        cached(&no_key_again),
        0,
        "a keyless request leaves nothing behind"
    );

    // Streaming reports it too.
    let mut body = keyed(Some(&key_a));
    body["stream"] = true.into();
    body["stream_options"] = json!({"include_usage": true});
    let (chunks, _) = parse_chunks(&node.chat_stream(&body).await);
    assert_eq!(
        chunks.last().unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"],
        hit
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tools_and_tool_history_render_and_parse() {
    let node = TestNode::start(&[]).await;
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }]);
    let body = json!({
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_completion_tokens": 6,
        "temperature": 0,
    });
    let (status, v) = node.chat_json(&body).await;
    assert_eq!(status, 200, "{v}");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 20);

    // History arguments that are not a JSON object are refused before admission.
    let mut bad = body.clone();
    bad["messages"][1]["tool_calls"][0]["function"]["arguments"] = "[1, 2]".into();
    let (status, v) = node.chat_json(&bad).await;
    assert_eq!(status, 400, "{v}");
    assert_eq!(node.stats().submitted, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn prompts_beyond_the_model_length_are_refused() {
    let node = TestNode::start(&[(env::MAX_MODEL_LEN, "64")]).await;
    let (status, v) = node.chat_json(&request(&LONG_TEXT.repeat(4), 4)).await;
    assert_eq!(status, 400);
    assert_eq!(v["error"]["type"], "context_length_exceeded");
    assert_eq!(node.stats().submitted, 0);
}

// ---------------------------------------------------------------------------------------
// Boot.
// ---------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn boot_verifies_loads_and_serves() {
    // Loads a second copy of the model (the only test that does).
    let config = config_from(&env_map()).unwrap();
    let node = tokio::task::spawn_blocking(|| eidola_server_engine::boot(config))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.weights_hash, weights_hash());
    let node = TestNode::serve(node).await;
    let (status, _) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 200);
}

#[test]
fn boot_refuses_a_wrong_expected_weights_hash() {
    let mut map = env_map();
    map.insert(env::WEIGHTS_SHA256, "ab".repeat(32));
    let err = eidola_server_engine::boot(config_from(&map).unwrap()).unwrap_err();
    assert!(err.to_string().contains("weights hash"), "{err}");
    assert!(err.to_string().contains(weights_hash()), "{err}");

    // A loaded model is checked against the configuration too.
    let err = eidola_server_engine::start(config_from(&map).unwrap(), loaded()).unwrap_err();
    assert!(err.to_string().contains("weights hash"), "{err}");
}

#[test]
fn configuration_refuses_anything_missing_or_malformed() {
    let full = env_map();
    assert!(config_from(&full).is_ok());
    for key in full.keys() {
        let mut map = full.clone();
        map.remove(key);
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(key), "{key}: {err}");
        map.insert(key, String::new());
        assert!(config_from(&map).is_err(), "{key} empty");
    }
    let bad = [
        (env::GATEWAY_TOKEN, "not-the-token"),
        (env::GATEWAY_TOKEN_HASH, "not-a-hash"),
        (env::WEIGHTS_SHA256, "abc"),
        (env::EXECUTOR, "gpu"),
        (env::BIND_ADDR, "localhost"),
        (env::KV_BLOCKS, "1"),
        (env::MAX_SEQS, "0"),
        (env::DRAFT_TOKENS, "-1"),
        (env::PREFIX_CACHE, "yes"),
        (env::CACHE_IDLE_TTL_SECS, "0"),
        (env::CACHE_MAX_AGE_SECS, "60"),
        (env::MODEL_ID, "has space"),
        (env::WEIGHTS_STORAGE, "readonly"),
    ];
    for (key, value) in bad {
        let mut map = full.clone();
        map.insert(key, value.to_string());
        let err = config_from(&map).unwrap_err();
        // Errors name the variable, never echo the secret.
        assert!(err.contains(key), "{key}: {err}");
        assert!(!err.contains(TOKEN), "{err}");
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut map = full.clone();
        map.insert(env::EXECUTOR, "cuda".into());
        assert!(config_from(&map).unwrap_err().contains("cuda"));
    }
    // The CUDA executor's settings are refused, not ignored, with the CPU executor.
    for (key, value) in [
        (env::KV_DEVICE_BYTES, "1073741824"),
        (env::KERNELS_DIR, "/k"),
        (env::CUDA_GRAPHS, "on"),
    ] {
        let mut map = full.clone();
        map.insert(key, value.to_string());
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(key) && err.contains("cuda"), "{key}: {err}");
    }
    // An argon2i (not argon2id) hash is refused.
    let mut map = full.clone();
    let argon2i = {
        use argon2::PasswordHasher;
        argon2::Argon2::new(
            argon2::Algorithm::Argon2i,
            argon2::Version::V0x13,
            Default::default(),
        )
        .hash_password(TOKEN.as_bytes())
        .unwrap()
        .to_string()
    };
    map.insert(env::GATEWAY_TOKEN_HASH, argon2i);
    assert!(config_from(&map).unwrap_err().contains("Argon2id"));
}

#[test]
fn engine_sizing_is_checked_against_the_model() {
    for (key, value, needle) in [
        (env::DRAFT_TOKENS, "3", "MTP layers"),
        (env::MAX_MODEL_LEN, "5000", "max_position_embeddings"),
        // A decode row with 2 drafts costs 3 query tokens.
        (env::MAX_BATCHED_TOKENS, "2", "decode row"),
    ] {
        let mut map = env_map();
        map.insert(key, value.to_string());
        let err = eidola_server_engine::start(config_from(&map).unwrap(), loaded()).unwrap_err();
        assert!(err.to_string().contains(needle), "{key}: {err}");
    }
}

/// A disconnect while the prompt is still being prepared does not free the admission
/// slot: the preparation keeps running on the blocking pool, so it keeps the permit.
#[tokio::test(flavor = "multi_thread")]
async fn disconnect_during_preparation_keeps_the_admission_slot() {
    let node = TestNode::start(&[(env::MAX_REQUESTS, "1")]).await;
    // Seconds of rendering and tokenizing (refused afterwards as too long).
    let big = request(&"lorem ipsum dolor sit amet ".repeat(600_000), 4);
    let probe = request("hi", 1);
    let big_request = node.chat(&big);
    let big_task = tokio::spawn(async move { big_request.send().await });
    // Wait until the big request holds the only slot.
    while node.admission.in_flight() == 0 {
        assert!(!big_task.is_finished());
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert!(
        !big_task.is_finished(),
        "preparation finished too soon to test"
    );
    // The client goes away mid-preparation.
    big_task.abort();
    let _ = big_task.await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        node.admission.in_flight(),
        1,
        "the abandoned preparation still holds it"
    );
    let (status, v) = node.chat_json(&probe).await;
    assert_eq!(status, 503, "the bound was exceeded: {v}");
    assert_eq!(v["error"]["type"], "overloaded");
    // The slot comes back when the abandoned preparation ends.
    let mut admitted = false;
    for _ in 0..3000 {
        let (status, _) = node.chat_json(&probe).await;
        if status == 200 {
            admitted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(admitted);
}

/// `tool_choice: "none"` shows the model no tools: the prompt is the one rendered
/// without them.
#[tokio::test(flavor = "multi_thread")]
async fn tool_choice_none_renders_without_tools() {
    let node = TestNode::start(&[]).await;
    let tools = json!([{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
    }]);
    let prompt_tokens = |v: &Value| v["usage"]["prompt_tokens"].as_u64().unwrap();
    let (_, plain) = node.chat_json(&request("Weather in Paris?", 2)).await;
    let mut with_tools = request("Weather in Paris?", 2);
    with_tools["tools"] = tools;
    let (_, auto) = node.chat_json(&with_tools).await;
    with_tools["tool_choice"] = "none".into();
    let (status, none) = node.chat_json(&with_tools).await;
    assert_eq!(status, 200, "{none}");
    assert!(prompt_tokens(&auto) > prompt_tokens(&plain));
    assert_eq!(prompt_tokens(&none), prompt_tokens(&plain));
    assert_eq!(
        none["choices"], plain["choices"],
        "the same prompt, the same greedy output"
    );
}

/// No multimodal key can ride along on a text part: every part variant denies unknown
/// fields, so the template never sees one.
#[tokio::test(flavor = "multi_thread")]
async fn multimodal_keys_in_text_parts_are_refused() {
    let node = TestNode::start(&[]).await;
    for key in [
        "image_url",
        "image",
        "audio",
        "video",
        "input_audio",
        "file",
    ] {
        let mut part = json!({"type": "text", "text": "hi"});
        part[key] = json!({"url": "data:,"});
        let body = json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": [part]}],
            "max_completion_tokens": 2,
        });
        let (status, v) = node.chat_json(&body).await;
        assert_eq!(status, 400, "{key}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        // And on the image part itself.
        let mut part = json!({"type": "image_url", "image_url": {"url": "data:,"}});
        part[key] = json!("x");
        let body = json!({"model": MODEL_ID, "messages": [{"role": "user", "content": [part]}]});
        let (status, _) = node.chat_json(&body).await;
        assert_eq!(status, 400, "{key} on an image part");
    }
    assert_eq!(node.stats(), Stats::default());
}

/// The cached dev model is rebuilt when its recorded input fingerprint does not match.
#[test]
fn a_stale_dev_model_is_rebuilt() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("model");
    fixture::ensure_dev_model(&out).unwrap();
    let hash = fixture::weights_hash_of(&out);
    assert_eq!(hash, weights_hash());
    // Tamper with the built model and mark it as built from other inputs.
    std::fs::write(out.join("config.json"), b"{}").unwrap();
    std::fs::write(out.join(fixture::FINGERPRINT_FILE), b"other inputs").unwrap();
    fixture::ensure_dev_model(&out).unwrap();
    assert_eq!(
        fixture::weights_hash_of(&out),
        hash,
        "rebuilt from the current inputs"
    );
    // A current one is reused as is.
    std::fs::write(out.join("marker"), b"").unwrap();
    fixture::ensure_dev_model(&out).unwrap();
    assert!(out.join("marker").exists());
}

/// Concurrent builders of one stale directory all return with the current model in
/// place: none moves aside a model another has just published.
#[test]
fn concurrent_dev_model_builders_never_unpublish_each_other() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("model");
    fixture::ensure_dev_model(&out).unwrap();
    std::fs::write(out.join(fixture::FINGERPRINT_FILE), b"other inputs").unwrap();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                fixture::ensure_dev_model(&out).unwrap();
                assert_eq!(fixture::weights_hash_of(&out), weights_hash());
            });
        }
    });
}

/// Production storage is checked, not assumed: a writable weights directory is refused
/// before anything is hashed or loaded, and a model loaded under one storage mode cannot
/// start a node configured for the other.
#[test]
fn verified_readonly_refuses_a_writable_weights_directory() {
    let mut map = env_map();
    map.insert(env::WEIGHTS_STORAGE, "verified-readonly".into());
    let err = eidola_server_engine::boot(config_from(&map).unwrap()).unwrap_err();
    assert!(err.to_string().contains("read-only mount"), "{err}");
    let err = eidola_server_engine::start(config_from(&map).unwrap(), loaded()).unwrap_err();
    assert!(err.to_string().contains("storage"), "{err}");
}

/// The storage check comes before any weights file is opened or parsed: an unparseable
/// shard in a writable directory is refused for its storage, never parsed.
#[test]
fn verified_readonly_checks_storage_before_parsing() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("model.safetensors"),
        b"not a safetensors file",
    )
    .unwrap();
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    let mut map = env_map();
    map.insert(env::WEIGHTS_STORAGE, "verified-readonly".into());
    map.insert(env::WEIGHTS_DIR, dir.path().display().to_string());
    let err = eidola_server_engine::boot(config_from(&map).unwrap()).unwrap_err();
    assert!(err.to_string().contains("read-only mount"), "{err}");
    // The same directory in development mode does get parsed (and fails there).
    map.insert(env::WEIGHTS_STORAGE, "dev-writable".into());
    let err = eidola_server_engine::boot(config_from(&map).unwrap()).unwrap_err();
    assert!(err.to_string().contains("cannot open the weights"), "{err}");
}
