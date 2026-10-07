#!/usr/bin/env python3
"""Generate the chat-template fixture corpus.

Renders every case with `transformers.apply_chat_template` (the reference
implementation the Rust renderer must match byte for byte) and tokenizes the
result with the reference `tokenizers` build. Dev-only: the outputs are
committed, and `cargo test` never runs Python.

Usage (versions pinned in requirements.txt):

    python -I dev/gen_template_fixtures.py --model-dir /path/to/MiMo-V2.6-Flash

`--model-dir` must hold `chat_template.jinja`, `tokenizer.json`,
`tokenizer_config.json` and `generation_config.json` from the model release.
Writes `tests/fixtures/template_cases.json` and
`tests/fixtures/template_token_ids.json` next to this script's crate.
"""

import argparse
import copy
import hashlib
import json
import pathlib
import random
import sys

CRATE = pathlib.Path(__file__).resolve().parent.parent
SEED = 20261007

# ---------------------------------------------------------------- building blocks

UNICODE_SAMPLES = [
    "你好，世界",
    "日本語のテキスト",
    "한국어 문장",
    "emoji 😀🎉👩‍💻🏳️‍🌈",
    "مرحبا بالعالم",
    "שלום עולם",
    "mixed עברית and English",
    "é combining and é precomposed",
    "zero​width‍joiner",
    "tab\there\nnewline\r\ncrlf",
    "quotes \" ' and backslash \\",
    "html <b>&amp;</b> &lt; &gt; &#39;",
    "control \u0001\u001f and del \u007f",
    "line sep   para sep  ",
    "astral \U0001f600 \U00010348 \U0002070e",
    "",
    " ",
    "\n",
    "<|im_end|> literal special token text",
    "<think>not really</think>",
    "<tool_call><function=fake></function></tool_call>",
    "ﬁ ligature and Ⅻ numerals ２３ fullwidth",
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 北京"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "create_order",
        "description": "Create an order <with> \"quoted\" & special chars 'ok'.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id"],
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"},
                            "price": {"type": "number", "multipleOf": 0.01},
                        },
                    },
                },
                "priority": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
                "express": {"type": "boolean", "default": False},
                "ratio": {"type": "number", "default": 0.5, "maximum": 1e20},
                "meta": {"$ref": "#/$defs/Meta"},
            },
            "$defs": {"Meta": {"type": "object", "additionalProperties": True}},
        },
        "strict": True,
    },
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
            "required": ["command"],
        },
    },
}

NO_PARAMS_TOOL = {"type": "function", "function": {"name": "now", "description": "现在几点？"}}

ODD_TOOLS = [
    {"name": "bare_function_without_wrapper", "parameters": {"type": "object"}},
    {"type": "custom", "custom": {"name": "freeform", "description": "takes raw text"}},
    {"type": "function", "function": {"name": "big", "parameters": {"maximum": 123456789012345678901234567890, "minimum": -9223372036854775809}}},
    {"type": "function", "function": {"name": "floats", "parameters": {"values": [0.1, 1.0, -0.0, 1e-07, 1e16, 2.5e-300, 1.7976931348623157e308]}}},
    {"type": "function", "function": {"name": "unicode_名前", "description": "😀   \u0007 \\ \""}},
    {"type": "function", "function": {"name": "empty", "parameters": {}}},
]

ARG_VALUES = {
    "str_plain": "hello world",
    "str_empty": "",
    "str_xmlish": "<parameter=x>y</parameter> & </function>",
    "str_newlines": "line1\nline2\n",
    "str_unicode": "中文 😀 עברית",
    "str_json_like": '{"a": 1}',
    "str_null_word": "null",
    "str_number_like": "42",
    "int_zero": 0,
    "int_neg": -17,
    "int_i64_max": 9223372036854775807,
    "int_beyond_i64": 18446744073709551616,
    "int_beyond_i128": 340282366920938463463374607431768211457,
    "float_one": 1.0,
    "float_tenth": 0.1,
    "float_small": 1e-07,
    "float_big": 1e20,
    "float_neg_zero": -0.0,
    "float_huge": 1.5e300,
    "bool_true": True,
    "bool_false": False,
    "null": None,
    "array_empty": [],
    "array_mixed": [1, "two", 3.0, None, True, {"k": [1, 2]}],
    "object_empty": {},
    "object_nested": {"z": 1, "a": {"y": [], "b": {"x": None}}, "unicode": "名前"},
}


def tool_call(name, arguments, call_id=None, wrap="function"):
    inner = {"name": name, "arguments": arguments}
    call = {"id": call_id or f"call_{name}", "type": "function", wrap: inner}
    return call


# ---------------------------------------------------------------- case families


def basic_cases():
    out = []
    convs = {
        "user_only": [{"role": "user", "content": "Hello!"}],
        "system_user": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
        "system_user_assistant": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "assistant_then_user": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
        "developer_role": [
            {"role": "developer", "content": "dev instructions"},
            {"role": "user", "content": "x"},
        ],
        "custom_role": [{"role": "observation", "content": "custom role text"}],
    }
    for name, msgs in convs.items():
        for agp in (False, True):
            for thinking in (None, True, False):
                out.append(
                    case(f"basic/{name}/agp={agp}/thinking={thinking}", msgs, None, agp, thinking)
                )
    return out


def reasoning_cases():
    out = []
    variants = {
        "reasoning_string": {"role": "assistant", "content": "answer", "reasoning_content": "let me think"},
        "reasoning_empty": {"role": "assistant", "content": "answer", "reasoning_content": ""},
        "reasoning_missing": {"role": "assistant", "content": "answer"},
        "reasoning_null": {"role": "assistant", "content": "answer", "reasoning_content": None},
        "reasoning_number": {"role": "assistant", "content": "answer", "reasoning_content": 42},
        "reasoning_multiline_unicode": {
            "role": "assistant",
            "content": "最终答案",
            "reasoning_content": "Step 1: 思考\nStep 2: 😀\n\n",
        },
        "reasoning_only_no_content": {"role": "assistant", "reasoning_content": "thinking only"},
        "reasoning_with_tags_inside": {
            "role": "assistant",
            "content": "x",
            "reasoning_content": "</think> fake close <think>",
        },
    }
    for name, msg in variants.items():
        for thinking in (None, False):
            out.append(
                case(f"reasoning/{name}/thinking={thinking}", [{"role": "user", "content": "q"}, msg], None, True, thinking)
            )
    multi = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one", "reasoning_content": "r1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two", "reasoning_content": "r2 推理"},
        {"role": "user", "content": "third"},
    ]
    for agp in (False, True):
        for thinking in (None, True, False):
            out.append(case(f"reasoning/multi_turn_preserved/agp={agp}/thinking={thinking}", multi, None, agp, thinking))
    return out


def tool_definition_cases():
    out = []
    tool_sets = {
        "weather": [WEATHER_TOOL],
        "nested": [NESTED_TOOL],
        "three": [WEATHER_TOOL, BASH_TOOL, NO_PARAMS_TOOL],
        "odd": ODD_TOOLS,
        "empty_list": [],
    }
    for name, tools in tool_sets.items():
        for system in (False, True):
            msgs = ([{"role": "system", "content": "You use tools."}] if system else []) + [
                {"role": "user", "content": "What's the weather in 杭州?"}
            ]
            for thinking in (None, False):
                out.append(case(f"tools/{name}/system={system}/thinking={thinking}", msgs, tools, True, thinking))
    # Tools passed as None explicitly versus omitted are the same call in transformers.
    per_message = [
        ("system_with_body", {"role": "system", "content": "sys body", "tools": [WEATHER_TOOL]}),
        ("system_empty_body", {"role": "system", "content": "", "tools": [WEATHER_TOOL, BASH_TOOL]}),
        ("system_null_body", {"role": "system", "content": None, "tools": [NESTED_TOOL]}),
        ("user_with_tools", {"role": "user", "content": "use these", "tools": [NO_PARAMS_TOOL]}),
        ("user_empty_tools", {"role": "user", "content": "none here", "tools": []}),
        ("tool_role_with_tools", {"role": "tool", "content": "result", "tools": [WEATHER_TOOL]}),
        ("assistant_tools_ignored", {"role": "assistant", "content": "ignored tools", "tools": [WEATHER_TOOL]}),
        ("parts_body_with_tools", {"role": "user", "content": [{"type": "text", "text": "part"}], "tools": [BASH_TOOL]}),
    ]
    for name, msg in per_message:
        msgs = [msg] if msg["role"] != "assistant" else [{"role": "user", "content": "q"}, msg]
        out.append(case(f"tools/per_message/{name}", msgs, None, True, None))
        out.append(case(f"tools/per_message/{name}/plus_global", msgs, [BASH_TOOL], True, None))
    return out


def tool_call_cases():
    out = []
    base = [{"role": "user", "content": "do it"}]
    # One case per argument type, plus all at once.
    for key, value in ARG_VALUES.items():
        msgs = base + [
            {"role": "assistant", "content": None, "tool_calls": [tool_call("f", {key: value})]},
            {"role": "tool", "content": "ok", "tool_call_id": "call_f"},
        ]
        out.append(case(f"tool_calls/arg_type/{key}", msgs, [NESTED_TOOL], True, None))
    msgs = base + [{"role": "assistant", "content": "", "tool_calls": [tool_call("f", dict(ARG_VALUES))]}]
    out.append(case("tool_calls/arg_type/all", msgs, None, False, None))

    shapes = {
        "content_null": {"role": "assistant", "content": None, "tool_calls": [tool_call("get_weather", {"city": "Paris"})]},
        "content_missing": {"role": "assistant", "tool_calls": [tool_call("get_weather", {"city": "Paris"})]},
        "content_empty": {"role": "assistant", "content": "", "tool_calls": [tool_call("get_weather", {"city": "Paris"})]},
        "content_text": {
            "role": "assistant",
            "content": "Let me check.",
            "reasoning_content": "need weather",
            "tool_calls": [tool_call("get_weather", {"city": "Paris"})],
        },
        "two_calls": {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("get_weather", {"city": "Paris"}, "c1"), tool_call("execute_bash", {"command": "ls -la | grep x && echo ok"}, "c2")],
        },
        "empty_tool_calls": {"role": "assistant", "content": "no calls", "tool_calls": []},
        "null_tool_calls": {"role": "assistant", "content": "null calls", "tool_calls": None},
        "empty_arguments": {"role": "assistant", "content": None, "tool_calls": [tool_call("now", {})]},
        "null_arguments": {"role": "assistant", "content": None, "tool_calls": [tool_call("now", None)]},
        "string_arguments_verbatim": {"role": "assistant", "content": None, "tool_calls": [tool_call("get_weather", '{"city": "Paris"}')]},
        "empty_string_arguments": {"role": "assistant", "content": None, "tool_calls": [tool_call("now", "")]},
        "custom_input": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "type": "custom", "custom": {"name": "freeform", "input": "raw <text> input"}}],
        },
        "custom_arguments": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "type": "custom", "custom": {"name": "freeform", "arguments": {"a": 1}}}],
        },
        "function_input_non_string": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "input": 5, "arguments": {"b": 2}}}],
        },
        "unwrapped_call": {"role": "assistant", "content": None, "tool_calls": [{"name": "bare", "arguments": {"x": "y"}}]},
        "call_without_name": {"role": "assistant", "content": None, "tool_calls": [{"function": {"arguments": {"x": 1}}}]},
        "unicode_names": {"role": "assistant", "content": None, "tool_calls": [tool_call("工具", {"参数": "值", "emoji😀": [1]})]},
        "parameter_order": {"role": "assistant", "content": None, "tool_calls": [tool_call("f", {"z": 1, "a": 2, "m": 3})]},
    }
    for name, msg in shapes.items():
        msgs = base + [msg]
        for thinking in (None, False):
            out.append(case(f"tool_calls/shape/{name}/thinking={thinking}", msgs, [WEATHER_TOOL, BASH_TOOL], True, thinking))

    results = [
        {"role": "user", "content": "weather in two cities"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "call twice",
            "tool_calls": [tool_call("get_weather", {"city": "Paris"}, "a"), tool_call("get_weather", {"city": "東京"}, "b")],
        },
        {"role": "tool", "content": '{"temp": 20}', "tool_call_id": "a"},
        {"role": "tool", "content": [{"type": "text", "text": "晴れ 25°C"}], "tool_call_id": "b"},
        {"role": "assistant", "content": "Paris 20, Tokyo 25.", "reasoning_content": "summarize"},
        {"role": "user", "content": "thanks"},
    ]
    for agp in (False, True):
        for thinking in (None, True, False):
            out.append(case(f"tool_calls/results_round_trip/agp={agp}/thinking={thinking}", results, [WEATHER_TOOL], agp, thinking))
    tool_results = {
        "empty": {"role": "tool", "content": ""},
        "null": {"role": "tool", "content": None},
        "missing": {"role": "tool"},
        "multiline_json": {"role": "tool", "content": '{\n  "a": [1, 2]\n}'},
        "unicode": {"role": "tool", "content": "结果 😀 نتيجة"},
        "parts": {"role": "tool", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    }
    for name, msg in tool_results.items():
        msgs = base + [{"role": "assistant", "content": None, "tool_calls": [tool_call("now", {})]}, msg]
        out.append(case(f"tool_calls/result/{name}", msgs, [NO_PARAMS_TOOL], True, None))
    return out


def content_cases():
    out = []
    parts = {
        "text_parts": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        "string_parts": ["plain ", "strings"],
        "image_part": [{"type": "text", "text": "look:"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
        "image_key": [{"type": "image", "image": "x"}],
        "audio_parts": [{"type": "input_audio", "input_audio": {"data": "x"}}, {"type": "audio", "audio": "y"}, {"audio_url": "z"}],
        "video_part": [{"type": "video_url", "video_url": {"url": "v"}}],
        "unknown_part": [{"type": "file", "file": {"id": "f"}}, {"type": "text", "text": "after"}],
        "text_without_type": [{"text": "no type key"}],
        "empty_parts": [],
        "empty_text_part": [{"type": "text", "text": ""}],
        "text_part_unicode": [{"type": "text", "text": s} for s in UNICODE_SAMPLES[:6]],
    }
    for name, content in parts.items():
        for role in ("user", "system", "tool"):
            out.append(case(f"content/{name}/{role}", [{"role": role, "content": content}], None, True, None))
        out.append(
            case(
                f"content/{name}/assistant",
                [{"role": "user", "content": "q"}, {"role": "assistant", "content": content}],
                None,
                False,
                None,
            )
        )
    for i, text in enumerate(UNICODE_SAMPLES):
        msgs = [
            {"role": "system", "content": text},
            {"role": "user", "content": text},
            {"role": "assistant", "content": text, "reasoning_content": text},
        ]
        out.append(case(f"unicode/{i}", msgs, None, True, None))
        tool = {"type": "function", "function": {"name": "u", "description": text, "parameters": {"type": "object", "properties": {text or "k": {"type": "string"}}}}}
        msgs = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": None, "tool_calls": [tool_call("u", {"s": text, "l": [text], "o": {text: text}})]},
        ]
        out.append(case(f"unicode/{i}/tools", msgs, [tool], True, False))
    return out


def rejected_cases():
    """Inputs Python renders but the Rust renderer refuses (outside the
    validated domain)."""
    bad = {
        "role_not_string": [{"role": 5, "content": "x"}],
        "role_missing": [{"content": "x"}],
        "content_number": [{"role": "user", "content": 7}],
        "content_object": [{"role": "user", "content": {"text": "x"}}],
        "part_number": [{"role": "user", "content": [1, "a"]}],
        "part_null": [{"role": "user", "content": [None]}],
        "part_text_number": [{"role": "user", "content": [{"type": "text", "text": 1.5}]}],
        "message_not_object": ["just a string"],
        "tool_calls_string": [{"role": "assistant", "content": "x", "tool_calls": "abc"}],
        "tool_call_name_number": [{"role": "assistant", "tool_calls": [{"function": {"name": 3, "arguments": {}}}]}],
        "function_string": [{"role": "assistant", "tool_calls": [{"function": "f"}]}],
        "message_tools_object": [{"role": "user", "content": "x", "tools": {"a": 1}}],
    }
    return [case(f"rejected/{name}", msgs, None, True, None, reject=True) for name, msgs in bad.items()]


def python_error_cases():
    """Inputs on which Jinja2 raises; the Rust renderer must also fail."""
    bad = {
        "empty_conversation": [],
        "arguments_list": [{"role": "assistant", "tool_calls": [tool_call("f", [1, 2])]}],
        "arguments_number": [{"role": "assistant", "tool_calls": [tool_call("f", 5)]}],
    }
    return [case(f"error/{name}", msgs, None, True, None, reject=True) for name, msgs in bad.items()]


def random_cases(n):
    rng = random.Random(SEED)
    roles_pool = ["system", "user", "assistant", "tool"]
    tools_pool = [WEATHER_TOOL, NESTED_TOOL, BASH_TOOL, NO_PARAMS_TOOL]
    out = []

    def text():
        k = rng.randrange(4)
        return "".join(rng.choice(UNICODE_SAMPLES + ["word", "longer sentence here. ", "x" * 40]) for _ in range(k))

    def arg_value(depth=0):
        kind = rng.randrange(8 if depth < 2 else 6)
        if kind == 0:
            return text()
        if kind == 1:
            return rng.randint(-(10**20), 10**20)
        if kind == 2:
            return rng.choice([0.1, 2.5, -1e-9, 1e300, 3.0, rng.random()])
        if kind == 3:
            return rng.choice([True, False])
        if kind == 4:
            return None
        if kind == 5:
            return rng.randint(-5, 5)
        if kind == 6:
            return [arg_value(depth + 1) for _ in range(rng.randrange(3))]
        return {f"k{i}{text()[:3]}": arg_value(depth + 1) for i in range(rng.randrange(3))}

    for i in range(n):
        msgs = []
        for _ in range(rng.randrange(1, 7)):
            role = rng.choice(roles_pool)
            msg = {"role": role}
            if role == "assistant":
                if rng.random() < 0.7:
                    msg["reasoning_content"] = text()
                if rng.random() < 0.5:
                    msg["tool_calls"] = [
                        tool_call(rng.choice(["get_weather", "execute_bash", "create_order"]), {f"p{j}": arg_value() for j in range(rng.randrange(4))}, f"c{j}")
                        for j in range(rng.randrange(1, 3))
                    ]
                    msg["content"] = rng.choice([None, "", text()])
                else:
                    msg["content"] = text()
            else:
                msg["content"] = rng.choice([text(), [{"type": "text", "text": text()}], None])
                if role in ("system", "user") and rng.random() < 0.15:
                    msg["tools"] = rng.sample(tools_pool, rng.randrange(1, 3))
            msgs.append(msg)
        tools = rng.sample(tools_pool, rng.randrange(0, 4)) if rng.random() < 0.6 else None
        out.append(case(f"random/{i}", msgs, tools, rng.random() < 0.8, rng.choice([None, True, False])))
    return out


# ---------------------------------------------------------------- driver


def case(name, messages, tools, add_generation_prompt, enable_thinking, reject=False):
    return {
        "name": name,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "reject": reject,
    }


def render(tokenizer, messages, tools, agp, thinking):
    kwargs = {}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=agp, **kwargs
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, default=CRATE / "tests" / "fixtures")
    args = parser.parse_args()

    import jinja2
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    template_bytes = (args.model_dir / "chat_template.jinja").read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tokenizer.chat_template != template_bytes.decode("utf-8"):
        sys.exit("tokenizer did not load chat_template.jinja")

    cases = (
        basic_cases()
        + reasoning_cases()
        + tool_definition_cases()
        + tool_call_cases()
        + content_cases()
        + rejected_cases()
        + python_error_cases()
        + random_cases(60)
    )
    names = [c["name"] for c in cases]
    if len(names) != len(set(names)):
        sys.exit("duplicate case names")

    out_cases = []
    token_ids = {}
    for c in cases:
        # Round-trip through JSON text so Python renders exactly the values the
        # Rust side parses from the same text.
        messages_text = json.dumps(c["messages"], ensure_ascii=False)
        tools_text = None if c["tools"] is None else json.dumps(c["tools"], ensure_ascii=False)
        messages = json.loads(messages_text)
        tools = None if tools_text is None else json.loads(tools_text)
        entry = {
            "name": c["name"],
            "messages": messages_text,
            "tools": tools_text,
            "add_generation_prompt": c["add_generation_prompt"],
            "enable_thinking": c["enable_thinking"],
        }
        try:
            rendered = render(tokenizer, messages, tools, c["add_generation_prompt"], c["enable_thinking"])
        except Exception as e:  # noqa: BLE001 - recorded as the expected outcome
            if not c["reject"]:
                sys.exit(f"{c['name']}: reference raised {e!r}")
            entry["expected_error"] = f"{type(e).__name__}: {e}"
        else:
            if c["reject"]:
                entry["reference_output"] = rendered
                entry["expected_error"] = "rejected by input validation"
            else:
                entry["expected"] = rendered
                token_ids[c["name"]] = tokenizer.encode(rendered, add_special_tokens=False)
        out_cases.append(entry)

    meta = {
        "generator": "dev/gen_template_fixtures.py",
        "python": sys.version.split()[0],
        "transformers": transformers.__version__,
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
        "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((args.model_dir / "tokenizer.json").read_bytes()).hexdigest(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "template_cases.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cases": out_cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(args.out_dir / "template_token_ids.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "token_ids": token_ids}, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    rendered_count = sum(1 for c in out_cases if "expected" in c)
    print(f"{len(out_cases)} cases ({rendered_count} rendered, {len(out_cases) - rendered_count} rejected)")


if __name__ == "__main__":
    main()
