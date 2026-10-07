# eidola-engine-chat

The chat layer of Eidola's inference engine for the MiMo-V2.6 model family (Flash and Pro share every file this crate reads): prompt rendering, tokenization, and turning generated tokens back into OpenAI-shaped `reasoning_content` / `content` / `tool_calls` deltas. AGPL-3.0-only. Workspace context: root `AGENTS.md`.

## Pipeline

| Step | Module | Reference it must equal |
|---|---|---|
| Parse `messages` / `tools` | `json`, `template::ChatInput` | Python `json.loads` |
| Render the prompt | `template` | `transformers.apply_chat_template` |
| Encode | `tokenizer` | Python `tokenizers` (Oniguruma build) |
| Decode, incrementally | `tokenizer::Detokenizer` | `tokenizers` `decode(ids)` |
| Split reasoning | `reasoning` | (own rule, below) |
| Extract tool calls | `tool_call`, `args`, `pysem` | SGLang's MiMo detector |
| Assemble deltas, finish reason | `output` | (own rule, `output.rs` docs) |

## Model artifacts and pinning

The template, tokenizer, and generation config are read from the model directory at load time — they are part of the measured model artifact, never compiled in or downloaded. `ChatTemplate` and `MimoTokenizer` refuse any `chat_template.jinja` / `tokenizer.json` whose SHA-256 is not in `PINNED_TEMPLATE_SHA256` / `PINNED_TOKENIZER_SHA256`: every guarantee below is a tested property of those exact files. Supporting a new model revision means adding its hash *and* regenerating the fixtures against it.

`tests/fixtures/chat_template.jinja` is a copy of the upstream template (XiaomiMiMo, MIT) used only by tests. `tokenizer.json` (11 MB) is not committed; tests that need it read `EIDOLA_MIMO_MODEL_DIR` and skip, saying so, when it is unset.

## Template byte-exactness

The template is executed as-is by minijinja in an environment that mirrors the one `transformers` builds: `trim_blocks`, `lstrip_blocks`, no auto-escaping, lenient undefined, loop controls, `raise_exception`, and these overrides:

- **`tojson`** is Python's `json.dumps` (`json::dumps`), not minijinja's: no HTML escaping, `", "` / `": "` separators, insertion order, `float.__repr__` for floats, unbounded integers, and the `ensure_ascii` / `indent` / `separators` / `sort_keys` keyword arguments. Unknown kwargs are errors.
- **`is iterable`** follows Python's `iter()`: `none`, booleans and numbers are not iterable (minijinja's built-in says `none` is, which would make the template call `length` on a missing `tools`).
- Integers beyond `i128` travel as an opaque object that only `tojson` reads.

Inputs are parsed with `json::parse`, never `serde_json::Value`, because serde_json (without workspace-wide features) sorts keys and rounds big integers, and `tojson` prints both. Duplicate keys follow Python (first position, last value).

`template::validate` rejects input outside the domain where the two Jinja engines provably agree for this template: messages must be a non-empty array of objects with a string `role`; `content` a string, `null`, or an array of strings/objects (with a string `text`); `tool_calls` an array of objects whose `function` is an object with a string `name` and an object/string/null `arguments`; `tools` arrays. Python would render some of these (`role: 5` prints `5`, a `None` content part prints `None`); we refuse instead of guessing. Everything else — tool schemas, argument values, reasoning, unknown roles, unicode — renders exactly.

History tool-call `arguments` arrive from OpenAI clients as JSON strings, but the template prints a string verbatim, which is not the trained format. Serving must call `ChatInput::normalize_tool_call_arguments` first (it errors on a string that is not a JSON object, as SGLang does).

Tested by `tests/template_fixtures.rs`: every case in `tests/fixtures/template_cases.json` (300+ rendered, plus cases that must be rejected) is compared byte for byte with the string `transformers` produced.

## Tokenizer

`tokenizers` is built with `default-features = false, features = ["fancy-regex"]` (pure Rust; no Oniguruma, no C++). The Python reference uses Oniguruma for the pre-tokenizer regex, so equality is checked, not assumed: with `EIDOLA_MIMO_MODEL_DIR` set, `tokenizer::reference_tests` encodes every fixture prompt plus a few hundred unicode-heavy strings and compares ids with Python's, and checks the decode table against `tokenizers` for every id and for random sequences. `decode(encode(x))` is *not* `x` in general (the tokenizer NFC-normalizes).

Decoding uses a per-id byte table built from `tokenizers`' byte-level decoder rule. `Detokenizer` holds back at most 3 bytes of an incomplete character and replaces invalid sequences exactly as `String::from_utf8_lossy` does, so the streamed text concatenated equals `decode(ids)` for any split. Special tokens (`<|im_end|>`, …) are skipped; `<think>`, `<tool_call>` and friends are not special and arrive as text. EOS ids come from `generation_config.json`.

## Output parsing

**Reasoning** (`reasoning`): with thinking enabled (`enable_thinking` not `false`), output starts in reasoning; leading `<think>` tags are dropped; reasoning ends at the first `</think>` (dropped) or `<tool_call>` (kept as content), whichever comes first; output that ends inside reasoning is all reasoning. With thinking disabled everything is content. SGLang's non-streaming path differs when `<tool_call>` precedes `</think>`; ours is the streaming behaviour, which is split-independent.

**Tool calls** (`tool_call`): SGLang's `detect_and_parse` semantics — content is the text before the first `<tool_call>`; each closed block is a call, an undeclared function's block (plus the text before it) goes back into content, a block without a complete `<function=…>…</function>` is dropped, and other text after the first block is dropped. Calls stream whole, when their closing tag arrives, with a stable `index` and an id from a caller-supplied `CallIdSource` (pass a per-response unique prefix).

**Argument typing** (`args`): two rules, chosen per `ToolSchemas`:

- `ArgumentTyping::Sglang` (default): SGLang's schema-typed conversion, documented rule by rule in `args.rs` — `html.unescape`, `null` in any case becomes JSON null, then by declared type `int()` / `float()` / boolean word / `json.loads` / `ast.literal_eval`, with SGLang's type-name normalisation. Where SGLang would fail the request or emit `NaN`/`Infinity`, the one parameter degrades to its text instead (listed in `args.rs`).
- `ArgumentTyping::RoundTrip`: the exact inverse of the template's rendering (string-typed values verbatim, others strict JSON first), so `parse(render(args)) == args`. SGLang's rule breaks that: a string `&amp;` comes back as `&`, a URL's `&copy=` as `©=`, the string `NULL` as null, `4.0` as `4`, and `yes` for a boolean as `false`. Those changes also alter the next turn's prompt relative to what the model generated.

`pysem` reproduces the CPython pieces SGLang relies on (`str.strip`, `int`, `float`, `html.unescape`, `ast.literal_eval`), with tables generated from CPython (`src/pysem/tables.rs`). The one known gap: `\N{NAME}` escapes in a Python literal are not decoded (the parameter stays text).

**Safety rules** (all parsers): no input can panic (property tests over arbitrary text, token ids and splits); scanning is linear; recursion is bounded (JSON 256 levels, Python literals 200); arguments are always a valid JSON object; streaming output equals complete-output parsing for every split.

**Format limits**: a string value containing `</parameter>`, `</function>` or `</tool_call>`, or a name containing `>`, cannot be represented in MiMo's format by any parser.

## Tests

- `tests/template_fixtures.rs` — byte-exact template corpus.
- `tests/sglang_differential.rs` — every value × declared-type case in `sglang_param_cases.json` (28k, including 3k fuzzed literals) and every model output in `sglang_text_cases.json`, against SGLang's own code; the outputs are also re-parsed in 1-, 3- and 7-character chunks.
- `tests/round_trip.rs` — property test that `RoundTrip` inverts the template; pinned counterexamples for the SGLang rule.
- Unit and property tests in each module (`cargo test -p eidola-engine-chat`).
- `tokenizer::reference_tests` — needs `EIDOLA_MIMO_MODEL_DIR`.

## Regenerating fixtures

Dev-only Python, pinned in `dev/requirements.txt`; `cargo test` never runs it. Use an isolated virtualenv and run with `python -I`:

```sh
python -I dev/gen_template_fixtures.py --model-dir <MiMo-V2.6 model dir>
python -I dev/gen_sglang_fixtures.py --sglang-src <sglang checkout> \
    --sglang-commit <sha from requirements.txt>
python -I dev/gen_tables.py
```

Regenerate after changing a pinned artifact, the reference versions, or a case list; review the fixture diff like code. The SGLang harness applies `normalize_json_schema_types` to tools (as SGLang's request validation does) and the per-parameter containment described above, flagging each case where containment fired (`contained`).
