#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Generate the DeepSeek-V4.1 fixture corpus.

Runs the model repository's reference encoder (`encoding/encoding.py` of
`deepseek-ai/DeepSeek-V4.1-Flash-1017` on Hugging Face, MIT) over every case
and tokenizes with the reference `tokenizers` build. Dev-only: the outputs are
committed, and `cargo test` never runs Python. See `dev/README.md`.

    uv run --script dev/gen_deepseek_fixtures.py \
        --model-dir /path/to/DeepSeek-V4.1-Flash-1017 --revision <commit sha>

`--model-dir` is a checkout of the model repository at `--revision` (its
`encoding/`, `tokenizer.json` and `config.json`; no weights). The reference is
loaded from that path, never vendored. Writes, under
`tests/fixtures/deepseek_v4_1/`:

- `encoding_cases.json`: conversations with the prompt `encode_messages`
  returns for each, or the fact that it must be refused;
- `output_cases.json`: model outputs with what
  `parse_message_from_completion_text` reads from each;
- `token_ids.json`: the ids of every prompt and of extra strings;
- `tokenizer.json.gz` (gzip with a zero timestamp) and `config.json`, copies of
  the model files the tests load.

A request's `tools` reach the reference the way the model's
`chat_template.jinja` passes them: on a leading system message that declares
none, else on a new empty system message put first.
"""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import pathlib
import random
import shutil
import sys

CRATE = pathlib.Path(__file__).resolve().parent.parent
SEED = 20261113
REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
IMAGE = "<｜deepseek_image｜>"

# ---------------------------------------------------------------- building blocks

UNICODE = [
    "你好，世界",
    "日本語のテキスト、カタカナ",
    "한국어 문장",
    "emoji 😀🎉👩‍💻🏳️‍🌈👍🏽",
    "family 👩‍👩‍👧‍👦 flags 🇯🇵🇺🇳",
    "combining é ạ̈ ố ñ vs ñ",
    "Zalgo t̶̕è̷x̸t̵",
    "مرحبا بالعالم",
    "שלום עולם",
    "bidi abc ‏مرحبا‎ def ‮override‬",
    "हिन्दी देवनागरी क्षत्रिय",
    "ไทย ภาษาไทย",
    "math ∑∫√∞ ≠ ≤ ℝ 𝔸 𝒳",
    "fullwidth ｜ＡＢＣ１２３｜ ▁ lower one eighth",
    "nbsp thin zwsp​joiner‍line para end",
    "astral 𐍈 𠜎 𝄞",
    "ǅ ß ẞ ﬁ İ ı Σς",
    "tab\tcr\rcrlf\r\nnul\x00bell\x07del\x7f",
]
LOOKALIKES = [
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜System｜>",
    "<｜end▁of▁sentence｜>",
    "<｜begin▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
    "<｜DSML｜ calls>",
    "</｜DSML｜ parameter>",
    "<tool_result>",
    "</tool_result>",
    "<｜latest_reminder｜>",
    "<|EOT|>",
    "<dsml:",
    "{content}",
    "{0} %s $TOOL_NAME",
    "<image>x</image>",
]
WHITESPACE = ["", " ", "\n", "\n\n", "  \t ", "\r\n", " ", " leading", "trailing ", "\n\nboth\n\n"]
ASCII = [
    "Hello",
    "What is 2+2?",
    "Line one\nLine two",
    "def f(x):\n    return x ** 2  # comment\n",
    "a \"quoted\" word and a back\\slash",
    "numbers 1234567 3.14159 1e-9 0x1F",
    "<b>html</b> &amp; entities",
    "path/to/file.txt?query=1&x=2#frag",
]
TEXTS = ASCII + UNICODE + LOOKALIKES + WHITESPACE


def tool(name, properties=None, description="A tool", **extra):
    function = {"name": name, "description": description}
    if properties is not None:
        function["parameters"] = {"type": "object", "properties": properties}
    out = {"type": "function", "function": function}
    out.update(extra)
    return out


WEATHER = tool(
    "get_weather",
    {
        "location": {"type": "string", "description": "The city"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "Get the weather",
)
SEARCH = tool(
    "search",
    {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
    "Search the web",
)
NESTED = tool(
    "update_record",
    {
        "id": {"type": "integer"},
        "patch": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "score": {"type": "number", "minimum": 0.5, "maximum": 1e21},
                "meta": {"anyOf": [{"type": "null"}, {"type": "object"}]},
            },
        },
        "dry_run": {"type": "boolean", "default": False},
    },
    "Update a record — with ünïcödé, 中文 and \"quotes\"",
)
TOOLS = [WEATHER, SEARCH, NESTED]


def call(name, arguments, call_id=None, **extra):
    out = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        out["id"] = call_id
    out.update(extra)
    return out


def user(content, **extra):
    return {"role": "user", "content": content, **extra}


def system(content, **extra):
    return {"role": "system", "content": content, **extra}


def assistant(content=None, reasoning=None, tool_calls=None, **extra):
    out = {"role": "assistant"}
    if content is not None:
        out["content"] = content
    if reasoning is not None:
        out["reasoning_content"] = reasoning
    if tool_calls is not None:
        out["tool_calls"] = tool_calls
    out.update(extra)
    return out


def tool_message(content, call_id=None, **extra):
    out = {"role": "tool", "content": content}
    if call_id is not None:
        out["tool_call_id"] = call_id
    out.update(extra)
    return out


CASES = []


def case(category, name, messages, tools=None, thinking=None, effort=None, drop_thinking=True, reject=None):
    """Adds a case; `thinking=None` adds it in both modes. `reject` names why
    the Rust encoder must refuse it (`"image"` for the typed image refusal)."""
    for mode in [True, False] if thinking is None else [thinking]:
        CASES.append(
            {
                "name": f"{category}/{name}/{'thinking' if mode else 'chat'}",
                "category": category,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "thinking": mode,
                "reasoning_effort": effort,
                "drop_thinking": drop_thinking,
                "reject": reject,
            }
        )


# ---------------------------------------------------------------- case lists


def basic_cases():
    c = "basic"
    case(c, "user", [user("Hello")])
    case(c, "system_user", [system("You are helpful."), user("Hello")])
    case(c, "system_only", [system("You are helpful.")])
    case(c, "assistant_only", [assistant("Hi")])
    case(c, "assistant_first", [assistant("Hi"), user("Hello")])
    case(c, "one_turn", [user("Hello"), assistant("Hi there")])
    case(c, "two_turns", [system("S"), user("q1"), assistant("a1"), user("q2")])
    case(c, "two_users", [user("first"), user("second")])
    case(c, "three_users", [system("S"), user("a"), user("b"), user("c"), assistant("x"), user("d")])
    case(c, "two_assistants", [user("q"), assistant("a1"), assistant("a2")])
    case(c, "no_content_keys", [{"role": "user"}, {"role": "assistant"}, {"role": "system"}])
    case(c, "null_assistant_content", [user("q"), {"role": "assistant", "content": None}, user("q2")])
    case(c, "null_system_content", [{"role": "system", "content": None}, user("q")])
    case(c, "extra_keys", [user("q", name="bob", mask=1), assistant("a", refusal=None, mask=0), user("q2")])
    case(c, "wo_eos", [user("q"), assistant("partial answ", reasoning="r", wo_eos=True)])
    case(c, "wo_eos_false", [user("q"), assistant("a", wo_eos=False)])
    case(c, "response_format", [system("S", response_format={"type": "json_object"}), user("q")])
    case(
        c,
        "response_format_schema_and_tools",
        [system("S", tools=[SEARCH], response_format={"type": "json_schema", "json_schema": {"name": "résumé", "x": 1.5}}), user("q")],
    )
    case(c, "response_format_falsy", [system("S", response_format={}), user("q")])
    case(c, "latest_reminder", [system("S"), {"role": "latest_reminder", "content": "2026-02-21,App"}, user("q")])
    case(c, "latest_reminder_after_user", [user("q"), {"role": "latest_reminder", "content": "now"}, assistant("a")])
    case(c, "latest_reminder_last", [user("q"), {"role": "latest_reminder", "content": "now"}])
    case(c, "latest_reminder_first", [{"role": "latest_reminder", "content": "now"}, user("q")])
    for task in ["action", "query", "authority", "domain", "title", "read_url"]:
        case("task", f"user_{task}", [system("S"), user("classify me", task=task)])
        case("task", f"user_{task}_answered", [user("classify me", task=task), assistant("Search", reasoning="r")])
    case("task", "assistant_title", [user("q"), assistant("a", task="title"), assistant("A title")])
    case("task", "task_then_user", [user("a", task="query"), user("b"), assistant("x")])
    case("task", "task_null", [user("a", task=None)])
    case("task", "task_then_tool", [user("a", task="action"), tool_message("r", "c1")])


def effort_cases():
    c = "effort"
    msgs = [system("S"), user("question")]
    for effort in [None, "low", "high", "max", 1, 2, 42, 50, 75, 99, 100]:
        case(c, f"{effort}", msgs, effort=effort)
        case(c, f"{effort}_no_system", [user("question")], effort=effort)
    case(c, "max_tools", [user("question")], tools=[SEARCH], effort="max")
    case(c, "low_multi_turn", [user("q1"), assistant("a1", "r1"), user("q2")], effort="low")
    case(c, "7_assistant_first", [assistant("a"), user("q")], effort=7)


def reasoning_cases():
    c = "reasoning"
    history = [
        system("S"),
        user("q1"),
        assistant("a1", "r1 — thinking…"),
        user("q2"),
        assistant("a2", "r2"),
        user("q3"),
    ]
    case(c, "dropped", history)
    case(c, "kept", history, drop_thinking=False)
    case(c, "kept_by_tools", history, tools=[SEARCH])
    case(c, "final_assistant", history[:-1])
    case(c, "final_assistant_kept", history[:-1], drop_thinking=False)
    case(c, "empty_reasoning", [user("q"), assistant("a", "")])
    case(c, "null_reasoning", [user("q"), {"role": "assistant", "content": "a", "reasoning_content": None}])
    case(c, "reasoning_only", [user("q"), assistant(None, "only thoughts")])
    case(c, "whitespace_reasoning", [user("q"), assistant("a", "  \n ")])
    tool_turn = [
        user("q1"),
        assistant("a1", "r1"),
        user("q2"),
        assistant(None, "r2", [call("search", {"query": "x"}, "c1")]),
        tool_message("value", "c1"),
    ]
    case(c, "current_turn_across_tools", tool_turn)
    case(c, "current_turn_across_tools_answered", tool_turn + [assistant("a2", "r3")])
    case(c, "next_turn_drops", tool_turn + [assistant("a2", "r3"), user("q3")])
    case(c, "next_turn_kept", tool_turn + [assistant("a2", "r3"), user("q3")], drop_thinking=False)
    case(c, "tools_on_user_message_keep", [user("q1", tools=[SEARCH]), assistant("a1", "r1"), user("q2")])
    case(c, "tools_on_merged_user_dropped", [user("q0"), user("q1", tools=[SEARCH]), assistant("a1", "r1"), user("q2")])
    case(c, "empty_tools_drop", [system("S", tools=[]), user("q1"), assistant("a1", "r1"), user("q2")])
    case(c, "user_of_only_results_is_no_turn", [user("q"), assistant("a", "r1"), {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}])
    case(c, "no_user_at_all", [assistant("a", "r1"), assistant("b", "r2")])


def mid_system_cases():
    c = "mid_system"
    case(c, "last", [system("sys"), user("q1"), assistant("a1", "r1"), system("mid sys")], effort=88)
    case(c, "then_assistant", [user("q1"), system("mid"), assistant("a", "r")])
    case(c, "then_user", [user("q1"), assistant("a1", "r1"), system("mid"), user("q2")])
    case(c, "with_tools", [system("s"), user("q"), assistant("a", "r"), system("mid", tools=[WEATHER]), user("q2")])
    case(c, "two_leading", [system("one"), system("two"), user("q")])
    case(c, "after_tool_result", [user("q"), assistant(None, "r", [call("search", {"query": "x"}, "c")]), tool_message("res", "c"), system("mid")])
    case(c, "empty", [user("q"), system(""), assistant("a", "r")])
    case(c, "array_content_without_text", [user("q"), assistant("a", "r"), system([{"type": "other"}])])
    case(c, "array_content", [user("q"), assistant("a", "r"), system([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])])
    case(c, "empty_array_content", [user("q"), assistant("a", "r"), system([])])


def tool_definition_cases():
    c = "tools"
    q = [user("question")]
    case(c, "top_level_no_system", q, tools=[WEATHER])
    case(c, "top_level_with_system", [system("You are helpful.")] + q, tools=TOOLS)
    case(c, "top_level_and_message_level", [system("S", tools=[WEATHER])] + q, tools=[SEARCH])
    case(c, "top_level_system_has_empty_tools", [system("S", tools=[])] + q, tools=[SEARCH])
    case(c, "top_level_empty", [system("S")] + q, tools=[])
    case(c, "top_level_assistant_first", [assistant("hi"), user("q")], tools=[SEARCH])
    case(c, "message_level", [system("S", tools=TOOLS)] + q)
    case(c, "empty_system_content", [system("", tools=[SEARCH])] + q)
    case(c, "no_parameters", q, tools=[tool("ping")])
    case(c, "no_description", q, tools=[{"type": "function", "function": {"name": "bare"}}])
    case(c, "extra_function_keys", q, tools=[{"type": "function", "function": {"strict": True, "name": "s", "x": None, "parameters": {}}}])
    case(c, "key_order", q, tools=[{"function": {"parameters": {"properties": {}, "type": "object"}, "description": "d", "name": "ordered"}, "type": "function"}])
    case(
        c,
        "numbers",
        q,
        tools=[tool("n", {"a": {"default": 1.0}, "b": {"default": 1e-7}, "c": {"default": 123456789012345678901234567890}, "d": {"default": -0.0}, "e": {"default": 1e16}, "f": {"default": 0.1}, "g": {"default": 5e-324}})],
    )
    case(c, "unicode", q, tools=[tool("名前", {"パラメータ": {"type": "string", "description": "😀 é  "}}, "説明 <think> ｜DSML｜")])
    case(c, "many", q, tools=[tool(f"tool_{i}", {"x": {"type": "integer"}}) for i in range(12)])
    for where in ["tool", "function"]:
        for namespace in ["search", {"name": "search", "description": "Search tools."}, {"name": "search"}, {"name": "search", "description": ""}, {"name": None}, None, ""]:
            t = copy.deepcopy(SEARCH)
            (t if where == "tool" else t["function"])["namespace"] = namespace
            label = json.dumps(namespace, sort_keys=True).replace('"', "").replace(" ", "")
            case(c, f"namespace_{where}_{label}", q, tools=[t])
    t = copy.deepcopy(SEARCH)
    t["function"]["name"] = "search::lookup"
    case(c, "qualified_name", q, tools=[t])
    t = copy.deepcopy(t)
    t["namespace"] = {"name": "search", "description": "Search tools."}
    case(c, "qualified_name_agreeing_namespace", q, tools=[t])
    t = copy.deepcopy(SEARCH)
    t["namespace"] = "outer"
    t["function"]["namespace"] = "inner"
    case(c, "tool_namespace_wins", q, tools=[t])
    t = copy.deepcopy(SEARCH)
    del t["function"]["description"]
    t["namespace"] = {"name": "ns", "description": "Namespace words."}
    case(c, "namespace_description_without_own", q, tools=[t])
    t = copy.deepcopy(t)
    t["function"]["description"] = None
    case(c, "namespace_description_null_own", q, tools=[t])
    same = []
    for namespace in [None, "search", "files"]:
        t = copy.deepcopy(SEARCH)
        if namespace:
            t["namespace"] = {"name": namespace}
        same.append(t)
    case(c, "same_name_in_namespaces", q, tools=same)
    t = copy.deepcopy(SEARCH)
    t["function"]["name"] = "::search"
    case(c, "empty_namespace_prefix", q, tools=[t])


def tool_call_cases():
    c = "tool_calls"
    base = [system("S", tools=TOOLS), user("q")]

    def one(name, arguments, **kw):
        case(c, name, base + [assistant(None, "thinking", [call("search", arguments, "c1")])], **kw)

    one("object", {"query": "weather in Paris", "limit": 5})
    one("empty_object", {})
    one("json_string", '{"query": "weather", "limit": 5}')
    one("json_string_compact", '{"query":"weather","limit":5}')
    one("json_string_whitespace", ' \n{ "query" : "x" }\t')
    one("double_encoded", json.dumps(json.dumps({"query": "x", "limit": 1})))
    one("triple_encoded", json.dumps(json.dumps(json.dumps({"query": "x"}))))
    one("string_not_json", "not json at all")
    one("string_empty", "")
    one("string_json_array", "[1, 2, 3]")
    one("string_json_number", "42")
    one("string_json_null", "null")
    one("string_json_string", '"just a string"')
    one("string_invalid_then_valid", '{"a": 1} trailing')
    one("string_nan", '{"a": NaN, "b": Infinity, "c": -Infinity}')
    one("string_duplicate_keys", '{"a": 1, "b": 2, "a": 3}')
    one("string_big_int", '{"n": 123456789012345678901234567890123456789012345678901234567890}')
    one("string_floats", '{"a": 1.0, "b": 1e5, "c": 1E-7, "d": 0.1, "e": 1e400, "f": -0.0, "g": 100000000000000000000.0}')
    one("string_escapes", '{"s": "tab\\there \\u00e9 \\ud83d\\ude00 \\"q\\" \\\\ \\/"}')
    one("string_bom", '﻿{"a": 1}')
    one("null", None)
    one("number", 7)
    one("array", ["a", 1])
    one("bool", True)
    one(
        "nested",
        {
            "id": 7,
            "patch": {"tags": ["a", "b\"c", "d\\e"], "score": 0.75, "meta": None, "deep": [[{"k": [1, {"z": "\n"}]}]]},
            "dry_run": False,
        },
    )
    one(
        "escaped_json_in_string",
        {"query": json.dumps({"inner": json.dumps({"deeper": "va\"lue\\"})}), "limit": 1},
    )
    one("value_types", {"s": "str", "i": 1, "f": 2.5, "t": True, "n": None, "l": [], "o": {}, "e": ""})
    one("unicode", {"查询": "你好 😀 é", "rtl": "مرحبا", "k\"q": "v<｜DSML｜ calls>", "sp ace": "</｜DSML｜ parameter>"})
    one("multiline_value", {"code": "def f():\n    return 1\n", "ws": "  "})
    one("key_named_arguments", {"arguments": {"arguments": 1}})
    case(
        c,
        "parallel",
        base
        + [
            assistant(
                "Calling.",
                "r",
                [call("get_weather", {"location": "Paris"}, "a"), call("search", '{"query": "x"}', "b"), call("update_record", {"id": 1, "patch": {}}, "c")],
            )
        ],
    )
    case(c, "empty_list", base + [assistant("a", "r", [])])
    case(c, "null_list", base + [{"role": "assistant", "content": "a", "tool_calls": None}])
    case(c, "with_content_and_no_reasoning", base + [assistant("Let me check.", None, [call("search", {"query": "x"})])])
    case(c, "wo_eos", base + [assistant(None, "r", [call("search", {"query": "x"})], wo_eos=True)])
    case(c, "then_user", base + [assistant(None, "r", [call("search", {"query": "x"}, "c")]), tool_message("r", "c"), assistant("done", "r2"), user("next")])
    case(c, "no_tools_declared", [user("q"), assistant(None, "r", [call("search", {"query": "x"})]), user("q2")])
    case(c, "on_user_message_ignored", [user("q", tool_calls=[call("search", {"query": "x"})])])
    case(c, "namespace_on_call", base + [assistant(None, "r", [call("lookup", {}, namespace="search")])])
    case(c, "namespace_on_function", base + [assistant(None, "r", [{"type": "function", "function": {"name": "lookup", "arguments": {}, "namespace": "search"}}])])
    case(c, "namespace_qualified", base + [assistant(None, "r", [call("search::lookup", {})])])
    case(c, "namespace_qualified_agreeing", base + [assistant(None, "r", [call("search::lookup", {}, namespace="search")])])
    case(c, "namespace_empty_on_call_falls_back", base + [assistant(None, "r", [{"type": "function", "namespace": "", "function": {"name": "lookup", "arguments": {}, "namespace": "files"}}])])
    case(c, "namespace_empty_everywhere", base + [assistant(None, "r", [{"type": "function", "namespace": "", "function": {"name": "lookup", "arguments": {}, "namespace": ""}}])])
    case(c, "namespace_empty_prefix", base + [assistant(None, "r", [call("::lookup", {})])])
    case(c, "namespace_null", base + [assistant(None, "r", [call("lookup", {}, namespace=None)])])


def tool_result_cases():
    c = "tool_results"
    calls = assistant(None, "r", [call("get_weather", {"location": "A"}, "a"), call("search", {"query": "B"}, "b"), call("search", {"query": "C"}, "c")])
    base = [system("S", tools=TOOLS), user("q"), calls]
    case(c, "in_order", base + [tool_message("ra", "a"), tool_message("rb", "b"), tool_message("rc", "c")])
    case(c, "reordered", base + [tool_message("rc", "c"), tool_message("ra", "a"), tool_message("rb", "b")])
    case(c, "unknown_ids_sort_first", base + [tool_message("rc", "c"), tool_message("r?", "zzz"), tool_message("rb", "b"), tool_message("r-", None)])
    case(c, "no_ids", base + [tool_message("r1"), tool_message("r2")])
    case(c, "duplicate_ids", base + [tool_message("r1", "b"), tool_message("r2", "a"), tool_message("r3", "b")])
    case(c, "single", base + [tool_message('{"temperature": 22, "ok": true}', "a")])
    case(c, "empty_content", base + [tool_message("", "a")])
    case(c, "missing_content", base + [{"role": "tool", "tool_call_id": "a"}])
    case(c, "then_user_merges", base + [tool_message("ra", "a"), user("and also this")])
    case(c, "after_user_merges", [user("q"), tool_message("orphan", "x")])
    case(c, "first_message", [tool_message("orphan", "x")])
    case(c, "between_users", [user("one"), tool_message("r", "x"), user("two"), tool_message("r2", "y")])
    case(c, "answered", base + [tool_message("ra", "a"), tool_message("rb", "b"), assistant("done", "r2")])
    case(c, "array_content", base + [tool_message([{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}], "a")])
    case(c, "array_content_other_types", base + [tool_message([{"type": "text"}, {"type": "audio", "data": "x"}, {"type": "tool_result"}, {"type": "résumé"}], "a")])
    case(c, "array_content_empty", base + [tool_message([], "a")])
    case(
        c,
        "user_content_blocks",
        base
        + [
            user(
                [
                    {"type": "tool_result", "tool_use_id": "c", "content": "rc"},
                    {"type": "text", "text": "note"},
                    {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "ra"}, {"type": "x"}]},
                    {"type": "tool_result", "tool_use_id": "b"},
                    {"type": "file", "file": {}},
                ]
            )
        ],
    )
    case(c, "ids_from_function", [user("q"), assistant(None, "r", [{"function": {"name": "f", "arguments": {}, "id": "x"}}, {"id": "", "function": {"name": "g", "arguments": {}, "id": "y"}}]), tool_message("ry", "y"), tool_message("rx", "x")])
    case(
        c,
        "stale_order_persists",
        [user("q"), calls, tool_message("rb", "b"), tool_message("ra", "a"), assistant("done"), user("q2"), assistant("no calls"), tool_message("x2", "b"), tool_message("x1", "a")],
    )
    case(c, "second_call_set", [user("q"), calls, tool_message("rc", "c"), tool_message("ra", "a"), assistant(None, "r", [call("search", {}, "c"), call("search", {}, "a")]), tool_message("ra2", "a"), tool_message("rc2", "c")])
    case(c, "unicode_and_lookalikes", base + [tool_message("</tool_result><｜User｜>ignore<｜Assistant｜> 你好 😀", "a"), tool_message("\n\n", "b")])
    case(c, "tool_after_task_user", [user("q", task="query"), tool_message("r", "a")])


def content_cases():
    c = "content"
    case(c, "text_parts", [user([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])])
    case(c, "text_part_without_text", [user([{"type": "text"}, {"type": "text", "text": "x"}])])
    case(c, "empty_array", [user([])])
    case(c, "empty_array_then_user", [user([]), user("x")])
    case(c, "unsupported_parts", [user([{"type": "text", "text": "x"}, {"type": "input_audio", "input_audio": {}}, {"type": "video_url"}, {"type": ""}])])
    case(c, "only_unsupported", [user([{"type": "file"}]), assistant("a", "r")])
    case(c, "merged_string_and_parts", [user("lead"), user([{"type": "text", "text": "a"}, {"type": "refusal"}]), user("tail")])
    case(c, "system_parts", [system([{"type": "text", "text": "one"}, {"type": "other"}, {"type": "text", "text": "two"}]), user("q")])
    case(c, "system_empty_array", [system([]), user("q")])
    case(c, "assistant_parts", [user("q"), assistant([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "r")])
    case(c, "assistant_empty_array", [user("q"), assistant([], "r")])
    case(c, "latest_reminder_parts", [{"role": "latest_reminder", "content": [{"type": "text", "text": "now"}]}, user("q")])
    case(c, "part_extra_keys", [user([{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}])])
    for i, text in enumerate(TEXTS):
        case("text", f"sample_{i}", [system(text), user(text), assistant(text, text), user(text)], thinking=i % 2 == 0)
    for i, text in enumerate(UNICODE + WHITESPACE + LOOKALIKES):
        case(
            "text",
            f"tool_round_{i}",
            [
                system(text, tools=[tool(f"t{i}", {"p": {"description": text}}, text)]),
                user([{"type": "text", "text": text}, {"type": text or "x"}]),
                assistant(text, text, [call(f"t{i}", {text or "k": text, "j": [text]}, "id")]),
                tool_message(text, "id"),
            ],
            thinking=i % 2 == 1,
        )


def rejected_cases():
    c = "rejected"
    q = [user("q")]

    def bad(name, messages, tools=None, reject="invalid", **kw):
        case(c, name, messages, tools=tools, thinking=True, reject=reject, **kw)

    image_url = {"type": "image_url", "image_url": {"url": "x.png"}}
    bad("image_url_part", [user([{"type": "text", "text": "look"}, image_url])], reject="image")
    bad("image_part", [user([{"type": "image", "source": "x"}])], reject="image")
    bad("image_in_system", [system([image_url]), user("q")], reject="image")
    bad("image_in_assistant", [user("q"), assistant([image_url])], reject="image")
    bad("image_in_tool_message", [user("q"), tool_message([image_url], "a")], reject="image")
    bad("image_in_tool_result_part", [user([{"type": "tool_result", "content": [image_url]}])], reject="image")
    bad("image_in_merged_user", [user("a"), user([image_url])], reject="image")
    bad("placeholder_in_content", [user(f"hi {IMAGE}")], reject="image")
    bad("placeholder_in_tool_content", [tool_message(f"hi {IMAGE}")], reject="image")
    bad("placeholder_in_system", [system(IMAGE), user("q")], reject="image")
    bad("placeholder_in_reasoning", [user("q"), assistant("a", IMAGE)], reject="image")
    bad("placeholder_in_text_part", [user([{"type": "text", "text": IMAGE}])], reject="image")
    bad("placeholder_in_tool_result_text_part", [user([{"type": "tool_result", "content": [{"type": "text", "text": IMAGE}]}])], reject="image")

    bad("empty_messages", [])
    bad("message_not_object", ["hi"])
    bad("role_missing", [{"content": "x"}])
    bad("role_number", [{"role": 5, "content": "x"}])
    bad("role_unknown", [{"role": "developer", "content": "x"}])
    bad("role_function", [{"role": "function", "content": "x"}])
    bad("content_blocks_key", [{"role": "user", "content_blocks": [{"type": "text", "text": "x"}]}])
    bad("user_content_null", [{"role": "user", "content": None}])
    bad("tool_content_null", q + [{"role": "tool", "content": None}])
    bad("latest_reminder_no_content", [{"role": "latest_reminder"}, user("q")])
    bad("latest_reminder_empty_array", [{"role": "latest_reminder", "content": []}, user("q")])
    bad("content_number", [{"role": "user", "content": 5}])
    bad("content_object", [{"role": "system", "content": {"a": 1}}, user("q")])
    bad("part_string", [user(["plain"])])
    bad("part_without_type", [user([{"text": "x"}])])
    bad("part_type_null", [user([{"type": None}])])
    bad("part_text_null", [user([{"type": "text", "text": None}])])
    bad("part_text_number", [system([{"type": "text", "text": 1}]), user("q")])
    bad("tool_result_content_number", [user([{"type": "tool_result", "content": 1}])])
    bad("tool_call_id_number", q + [{"role": "tool", "content": "x", "tool_call_id": 5}])
    bad("reasoning_number", q + [{"role": "assistant", "content": "a", "reasoning_content": 5}])
    bad("wo_eos_number", q + [{"role": "assistant", "content": "a", "wo_eos": 1}])
    bad("task_unknown", [user("q", task="translate")])
    bad("task_number", [user("q", task=1)])
    bad("tool_calls_object", q + [{"role": "assistant", "tool_calls": {"a": 1}}])
    bad("tool_call_not_object", q + [assistant(None, None, ["x"])])
    bad("tool_call_without_function", q + [assistant(None, None, [{"name": "f", "arguments": {}}])])
    bad("tool_call_function_string", q + [assistant(None, None, [{"function": "f"}])])
    bad("tool_call_without_name", q + [assistant(None, None, [{"function": {"arguments": {}}}])])
    bad("tool_call_name_number", q + [assistant(None, None, [{"function": {"name": 1, "arguments": {}}}])])
    bad("tool_call_without_arguments", q + [assistant(None, None, [{"function": {"name": "f"}}])])
    bad("tool_call_namespace_conflict", q + [assistant(None, None, [call("search::lookup", {}, namespace="files")])])
    bad("tool_call_nested_namespace", q + [assistant(None, None, [call("a::b::c", {})])])
    bad("tool_call_namespace_with_separator", q + [assistant(None, None, [call("f", {}, namespace="a::b")])])
    bad("tool_call_namespace_number", q + [assistant(None, None, [call("f", {}, namespace=5)])])
    bad("tool_call_id_number_on_call", q + [assistant(None, None, [call("f", {}, 5)])])
    bad("arguments_lone_surrogate", q + [assistant(None, None, [call("f", '{"a": "\\ud800"}')])])
    bad("arguments_too_deep", q + [assistant(None, None, [call("f", "[" * 300 + "]" * 300)])])
    bad("tools_object", q, tools={"a": 1})
    bad("tools_string_on_message", [system("S", tools="x"), user("q")])
    bad("tool_not_object", q, tools=["x"])
    bad("tool_without_function", q, tools=[{"type": "function", "name": "f"}])
    bad("tool_function_without_name", q, tools=[{"function": {"description": "d"}}])
    bad("tool_name_number", q, tools=[{"function": {"name": 1}}])
    for name, namespace in [("search::lookup", "files"), ("search::nested::lookup", None), ("lookup", "search::nested"), ("lookup", 5), ("lookup", {"description": "no name"}), ("lookup", {"name": 5}), ("lookup", {"name": "a", "description": 5}), ("lookup", ["a"])]:
        t = tool(name, {})
        t["namespace"] = namespace
        bad(f"tool_namespace_{name}_{json.dumps(namespace)}".replace('"', "").replace(" ", ""), q, tools=[t])
    t = tool("f", {}, description=5)
    t["namespace"] = {"name": "a", "description": "words"}
    bad("tool_description_number_under_namespace_description", q, tools=[t])
    bad("tools_invalid_on_user_message", [user("q", tools=[{"function": {}}])])


def random_cases(n):
    rng = random.Random(SEED)

    def text():
        r = rng.random()
        if r < 0.5:
            return rng.choice(TEXTS)
        if r < 0.8:
            return "".join(rng.choice(TEXTS) for _ in range(rng.randint(2, 4)))
        return "".join(chr(rng.choice([rng.randint(0x20, 0x7E), rng.randint(0xA0, 0x24FF), rng.randint(0x4E00, 0x4E80), rng.randint(0x1F600, 0x1F64F)])) for _ in range(rng.randint(1, 30)))

    def value(depth=0):
        r = rng.random()
        if depth > 2 or r < 0.35:
            return text()
        if r < 0.5:
            return rng.choice([0, 1, -7, 2**63, 10**25, 1.5, -0.0, 1e-7, 1e22, 0.1, 123456.789])
        if r < 0.6:
            return rng.choice([True, False, None])
        if r < 0.8:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        return {text() or "k": value(depth + 1) for _ in range(rng.randint(0, 3))}

    def arguments():
        obj = {text() or "k": value() for _ in range(rng.randint(0, 4))}
        r = rng.random()
        if r < 0.5:
            return obj
        if r < 0.75:
            return json.dumps(obj, ensure_ascii=rng.random() < 0.5)
        if r < 0.85:
            return json.dumps(json.dumps(obj))
        return rng.choice([text(), value(), None, 3, [value()]])

    def parts():
        out = []
        for _ in range(rng.randint(0, 3)):
            r = rng.random()
            if r < 0.5:
                out.append({"type": "text", "text": text()})
            elif r < 0.75:
                content = text() if rng.random() < 0.6 else [{"type": "text", "text": text()}, {"type": "blob"}]
                out.append({"type": "tool_result", "tool_use_id": rng.choice(["c0", "c1", "zz", ""]), "content": content})
            else:
                out.append({"type": rng.choice(["file", "audio", "x y"])})
        return out

    def random_tool(i):
        t = tool(f"fn{i}", {text() or "p": {"type": rng.choice(["string", "integer", "object"]), "default": value()}}, text())
        r = rng.random()
        if r < 0.15:
            t["namespace"] = rng.choice(["ns", {"name": "ns", "description": text()}])
        return t

    for i in range(n):
        messages = []
        if rng.random() < 0.6:
            messages.append(system(text() if rng.random() < 0.8 else [{"type": "text", "text": text()}]))
            if rng.random() < 0.3:
                messages[-1]["tools"] = [random_tool(j) for j in range(rng.randint(0, 2))]
            if rng.random() < 0.1:
                messages[-1]["response_format"] = value()
        for _ in range(rng.randint(1, 5)):
            r = rng.random()
            if r < 0.35:
                messages.append(user(text() if rng.random() < 0.7 else parts()))
                if rng.random() < 0.05:
                    messages[-1]["task"] = rng.choice(["action", "query", "title"])
            elif r < 0.7:
                n_calls = rng.choice([0, 0, 1, 2, 3])
                calls = [call(rng.choice(["fn0", "fn1", "ns::fn2", "名"]), arguments(), rng.choice([f"c{k}", None])) for k in range(n_calls)]
                messages.append(
                    assistant(
                        rng.choice([None, text()]),
                        rng.choice([None, text()]),
                        calls if n_calls or rng.random() < 0.2 else None,
                    )
                )
                if rng.random() < 0.05:
                    messages[-1]["wo_eos"] = True
                for k in rng.sample(range(n_calls), n_calls):
                    if rng.random() < 0.8:
                        messages.append(tool_message(text() if rng.random() < 0.8 else [{"type": "text", "text": text()}], rng.choice([f"c{k}", None, "zz"])))
            elif r < 0.8:
                messages.append(system(text()))
            elif r < 0.85:
                messages.append({"role": "latest_reminder", "content": text()})
            else:
                messages.append(tool_message(text(), rng.choice(["c0", None])))
        tools = [random_tool(j) for j in range(rng.randint(1, 3))] if rng.random() < 0.4 else None
        case(
            "random",
            f"{i:03}",
            messages,
            tools=tools,
            thinking=rng.random() < 0.6,
            effort=rng.choice([None, None, "low", "high", "max", rng.randint(1, 100)]),
            drop_thinking=rng.random() < 0.8,
        )


# ---------------------------------------------------------------- the reference


def load_reference(path):
    spec = importlib.util.spec_from_file_location("deepseek_v41_reference_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_request_tools(messages, tools):
    """`chat_template.jinja`'s placement of a request's `tools`."""
    messages = copy.deepcopy(messages)
    if not tools:
        return messages
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        if not messages[0].get("tools"):
            messages[0]["tools"] = copy.deepcopy(tools)
        return messages
    return [{"role": "system", "content": "", "tools": copy.deepcopy(tools)}] + messages


def reference_prompt(enc, c):
    return enc.encode_messages(
        with_request_tools(c["messages"], c["tools"]),
        thinking_mode="thinking" if c["thinking"] else "chat",
        drop_thinking=c["drop_thinking"],
        reasoning_effort=c["reasoning_effort"],
    )


def shipped_cases(enc, encoding_dir):
    """The reference's own golden pairs, each loaded as its test loads it."""
    out = []
    for path in sorted((encoding_dir / "tests").glob("test_input_*.json")):
        number = path.stem.rsplit("_", 1)[1]
        loaded = enc.load_cases(str(path))[0]
        expected = (encoding_dir / "tests" / f"test_output_{number}.txt").read_text(encoding="utf-8")
        prompt, images = enc.encode_case(loaded, thinking_mode="chat")
        if prompt != expected:
            sys.exit(f"the reference does not reproduce its own test_output_{number}.txt")
        if loaded.get("context"):
            sys.exit(f"test_input_{number}.json uses context, which this corpus does not cover")
        out.append(
            {
                "name": f"shipped/test_input_{number}",
                "category": "shipped",
                "messages": loaded["messages"],
                "tools": None,
                "thinking": (loaded.get("thinking_mode") or "chat") == "thinking",
                "reasoning_effort": loaded.get("reasoning_effort"),
                "drop_thinking": True,
                "reject": "image" if images else None,
                "expected": expected,
            }
        )
    return out


# ---------------------------------------------------------------- model outputs


def is_strict_json(text):
    """Whether `text` is RFC 8259 JSON with every number finite."""

    def refuse(constant):
        raise ValueError(constant)

    def finite(value):
        if isinstance(value, float):
            return value == value and abs(value) != float("inf")
        if isinstance(value, list):
            return all(finite(v) for v in value)
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        return True

    try:
        return finite(json.loads(text, parse_constant=refuse))
    except ValueError:
        return False


def output_cases(enc):
    """Model outputs and what the reference's parser reads from each. An
    output is the text the model writes after the prompt, up to and without
    its end-of-sequence token, which the reference parser is given back."""
    rng = random.Random(SEED + 1)
    outputs = []

    def add(name, text, thinking):
        declared = set()
        try:
            parsed = enc.parse_message_from_completion_text(text + enc.eos_token, "thinking" if thinking else "chat")
            calls = []
            strict = True
            for tc in parsed["tool_calls"]:
                name_ = tc["function"]["name"]
                if tc.get("namespace") is not None:
                    name_ = f"{tc['namespace']}::{name_}"
                declared.add(name_)
                strict = strict and is_strict_json(tc["function"]["arguments"])
                calls.append({"name": name_, "arguments": tc["function"]["arguments"]})
            reference = {"reasoning_content": parsed["reasoning_content"], "content": parsed["content"], "tool_calls": calls, "strict_json": strict}
        except Exception as e:  # noqa: BLE001 - the reference raises several types
            reference = None
            error = type(e).__name__
        entry = {"name": name, "text": text, "thinking": thinking, "declared": sorted(declared)}
        if reference is None:
            entry["reference_error"] = error
        else:
            entry["reference"] = reference
        outputs.append(entry)

    def rendered(message, thinking):
        text = enc.render_message(1, [{"role": "user", "content": "q"}, message], thinking_mode="thinking" if thinking else "chat")
        assert text.endswith(enc.eos_token)
        return text[: -len(enc.eos_token)]

    messages = [
        assistant("Hello there."),
        assistant("answer", "some reasoning\n\nwith paragraphs"),
        assistant("", "only reasoning"),
        assistant("summary", "  reason  ", [call("lookup", '{"query":"value","limit":2}')]),
        assistant(None, "r", [call("get_weather", {"location": "Paris", "unit": "celsius"})]),
        assistant("Two calls.", "r", [call("a", {"x": 1}), call("b", {"y": [1, {"z": None}], "s": "str"}), call("a", {})]),
        assistant(None, None, [call("search::lookup", {"q": "ns"})]),
        assistant(None, "r", [call("lookup", {"q": "x"}, namespace="files")]),
        assistant("text\n\n", "r\n\n", [call("f", {"s": "line1\nline2", "e": "", "ws": "  ", "j": "{\"a\": 1}", "n": -1.5e-7, "b": True, "u": "你好 😀 é مرحبا"})]),
        assistant(None, "r", [call("f", {"k\"q": "v", "名": ["值"], "nested": {"a": {"b": [1, 2, {"c": "d\"e\\f"}]}}})]),
        assistant(None, "r", [call("f", '{"a": NaN, "big": 123456789012345678901234567890, "f": 1e400}')]),
        assistant("content with <b>tags</b> and a lone < or > here", "r < 1 > 0"),
    ]
    for m, message in enumerate(messages):
        for thinking in [True, False]:
            add(f"rendered_{m}_{'thinking' if thinking else 'chat'}", rendered(message, thinking), thinking)

    for i in range(60):
        n_calls = rng.choice([0, 1, 1, 2, 3])
        calls = []
        for _ in range(n_calls):
            arguments = {}
            for _ in range(rng.randint(0, 3)):
                key = rng.choice(["a", "query", "名", "k k", "x_1"])
                arguments[key] = rng.choice(["text", "", "multi\nline", 1, 2.5, True, None, [1, "two"], {"n": {"m": []}}, "你好", "😀", "{\"json\": \"inside\"}", " lead and trail "])
            calls.append(call(rng.choice(["f", "g", "ns::h"]), arguments))
        message = assistant(rng.choice([None, "", "Answer.", "多行\n\n内容", "a < b"]), rng.choice([None, "", "thinking…", "step 1\n\nstep 2"]), calls or None)
        thinking = rng.random() < 0.6
        add(f"random_{i:02}", rendered(message, thinking), thinking)

    good = rendered(messages[3], True)
    block = good[good.index("\n\n<｜DSML｜ calls>") :]
    malformed = {
        "no_think_end": "reasoning that never ends",
        "think_end_only": "</think>",
        "empty": "",
        "calls_before_think_end": "plan" + block + "</think>late",
        "text_after_calls": good + "trailing words",
        "two_blocks": good + block,
        "unclosed_block": good[: -len("</｜DSML｜ calls>")],
        "unclosed_invoke": good.replace("</｜DSML｜ invoke>\n", ""),
        "unclosed_parameter": good.replace("</｜DSML｜ parameter>", "", 1),
        "no_blank_line_before_block": good.replace("summary\n\n<", "summary<"),
        "one_newline_before_block": good.replace("summary\n\n<", "summary\n<"),
        "three_newlines_before_block": good.replace("summary\n\n<", "summary\n\n\n<"),
        "v4_tags": good.replace("｜DSML｜ calls", "｜DSML｜tool_calls").replace("｜DSML｜ invoke", "｜DSML｜invoke").replace("｜DSML｜ parameter", "｜DSML｜parameter"),
        "empty_block": "r</think>c\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>",
        "empty_invoke": 'r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>',
        "duplicate_parameter": good.replace('name="limit"', 'name="query"'),
        "bad_string_flag": good.replace('string="false"', 'string="maybe"'),
        "invalid_json_value": good.replace(">2<", ">{not json<"),
        "spaces_between_elements": good.replace(">\n<", ">\n  <"),
        "no_newlines_between_elements": good.replace(">\n<", "><"),
        "name_with_quote": good.replace('name="lookup"', 'name="look"up"'),
        "think_in_content": "r</think>content with <think> and </think> in it",
        "special_in_reasoning": "r <think> again</think>c",
        "dsml_word_in_content": "r</think>the ｜DSML｜ marker",
        "block_text_in_content_without_newlines": "r</think>see <｜DSML｜ calls> for syntax",
    }
    for name, text in malformed.items():
        add(f"malformed_{name}", text, True)
    add("malformed_chat_with_think_end", "stray</think>content", False)
    add("malformed_chat_calls_only", block, False)
    add("malformed_chat_calls_no_blank_line", block[2:], False)
    return outputs


# ---------------------------------------------------------------- tokenizer strings


def tokenizer_strings(n):
    rng = random.Random(SEED + 2)
    strings = list(TEXTS) + [
        "1 12 123 1234 12345 123456 1234567 3.14159 1,000,000 ١٢٣٤٥ ⅓ ²³ １２３４",
        "汉字漢字かなカナ混在テキスト、句読点。「引用」",
        "word,word.word;word:word!word?word'word\"word",
        "!important #tag $var %d &amp 'q (p) *x +1 ,c -d .e /f :g ;h <i =j >k ?l @m [n \\o ]p ^q _r `s {t |u }v ~w",
        "   leading spaces\n\n\ntrailing newlines\n\n\n",
        "spaces   between    words\tand\ttabs  \n  mixed \r\n endings\r",
        "camelCase snake_case kebab-case SCREAMING_CASE dotted.name path/to/x",
        "https://example.com/a/b?c=d&e=f#g user@example.com 192.168.0.1",
        "{\"json\": [1, 2.5, true, null, \"s\"], \"k\": {\"n\": -1e-9}}",
        "fn main() { println!(\"{}\", 1 + 2); } // rust\n#include <stdio.h>\nint x = y ? 1 : 0;",
        "<｜User｜>data<｜Assistant｜><think>r</think>a<｜end▁of▁sentence｜>",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        "｜DSML｜DSML｜ ｜DSML ｜DSML｜｜DSML｜ <｜User｜ >",
        "<|place_holder_mm_span_0021|><|EOT|><dsml:x></dsml:x><｜▁pad▁｜>",
        "á̂̃ é ́ lone mark; 각 jamo",
        "\U0001F468‍\U0001F469‍\U0001F467 \U0001F3F3️‍\U0001F308 ❤️ 1️⃣",
        "",
        " ",
        "\n",
    ]
    ranges = [(0x20, 0x7E), (0xA0, 0x17F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x2000, 0x206F), (0x2190, 0x22FF), (0x3040, 0x30FF), (0x4E00, 0x9FA5), (0xAC00, 0xD7A3), (0xFF00, 0xFFEF), (0x1F300, 0x1F6FF), (0x300, 0x36F), (0x30, 0x39)]
    while len(strings) < n:
        r = rng.random()
        if r < 0.4:
            strings.append("".join(rng.choice(TEXTS) + rng.choice(["", " ", "\n", "1", "12", "1234"]) for _ in range(rng.randint(1, 5))))
        else:
            out = []
            for _ in range(rng.randint(1, 40)):
                lo, hi = rng.choice(ranges)
                out.append(chr(rng.randint(lo, hi)))
                if rng.random() < 0.2:
                    out.append(rng.choice([" ", "  ", "\n", "\t"]))
            strings.append("".join(out))
    return strings


# ---------------------------------------------------------------- main


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path, help="checkout of the model repository")
    parser.add_argument("--revision", required=True, help="the checkout's commit on Hugging Face")
    parser.add_argument("--out-dir", type=pathlib.Path, default=CRATE / "tests" / "fixtures" / "deepseek_v4_1")
    args = parser.parse_args()

    import tokenizers

    encoding_dir = args.model_dir / "encoding"
    enc = load_reference(encoding_dir / "encoding.py")
    tokenizer = tokenizers.Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    basic_cases()
    effort_cases()
    reasoning_cases()
    mid_system_cases()
    tool_definition_cases()
    tool_call_cases()
    tool_result_cases()
    content_cases()
    rejected_cases()
    random_cases(160)

    cases = shipped_cases(enc, encoding_dir)
    for c in CASES:
        # A refused case records whether the reference raised too; where it
        # renders instead, the refusal is this crate's narrower input domain.
        try:
            expected = reference_prompt(enc, c)
            if c["reject"] is None:
                c["expected"] = expected
            else:
                c["reference_error"] = None
        except Exception as e:  # noqa: BLE001 - the reference raises several types
            if c["reject"] is None:
                sys.exit(f"{c['name']}: the reference raised {type(e).__name__}: {e}")
            # The type only: a message can print a set, in an order that
            # changes from run to run.
            c["reference_error"] = type(e).__name__
        cases.append(c)
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names):
        sys.exit("duplicate case names")

    token_ids = {}
    for c in cases:
        if c["reject"] is None:
            token_ids[c["name"]] = tokenizer.encode(c["expected"], add_special_tokens=False).ids
        # Order-preserving JSON text, as a request carries it.
        c["messages"] = json.dumps(c["messages"], ensure_ascii=False)
        c["tools"] = None if c["tools"] is None else json.dumps(c["tools"], ensure_ascii=False)
    strings = tokenizer_strings(400)
    string_ids = [[text, tokenizer.encode(text, add_special_tokens=False).ids] for text in strings]

    meta = {
        "repository": REPOSITORY,
        "revision": args.revision,
        "encoding_py_sha256": sha256(encoding_dir / "encoding.py"),
        "tokenizer_sha256": sha256(args.model_dir / "tokenizer.json"),
        "python": sys.version.split()[0],
        "tokenizers": tokenizers.__version__,
        "generator": "dev/gen_deepseek_fixtures.py",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        with open(args.out_dir / name, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=None if name == "token_ids.json" else 1, separators=(",", ":") if name == "token_ids.json" else None)
            f.write("\n")

    write("encoding_cases.json", {"meta": meta, "cases": cases})
    write("output_cases.json", {"meta": meta, "cases": output_cases(enc)})
    write("token_ids.json", {"meta": meta, "token_ids": token_ids, "strings": string_ids})
    with open(args.out_dir / "tokenizer.json.gz", "wb") as f:
        f.write(gzip.compress((args.model_dir / "tokenizer.json").read_bytes(), compresslevel=9, mtime=0))
    shutil.copyfile(args.model_dir / "config.json", args.out_dir / "config.json")

    by_category = {}
    for c in cases:
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
    print(f"{len(cases)} encoding cases: {by_category}")
    print(f"{sum(c['reject'] is None for c in cases)} rendered, {sum(c['reject'] is not None for c in cases)} refused")
    print(f"{len(string_ids)} tokenizer strings")


if __name__ == "__main__":
    main()
