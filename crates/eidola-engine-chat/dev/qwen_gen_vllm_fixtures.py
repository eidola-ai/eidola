#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = ["transformers==5.19.0", "jinja2==3.1.6", "tokenizers==0.23.2", "regex==2026.9.29"]
# ///
"""Record what vLLM's Qwen3 parsers read from Qwen3.8 model outputs.

vLLM serves Qwen3.8 with `--reasoning-parser qwen3 --tool-call-parser
qwen3_coder`; both names are one parser, `vllm/parser/qwen3.py`, running on
vLLM's parser engine (`vllm/parser/engine/`). This runs those files from a vLLM
checkout you supply, with the few serving-layer modules they import replaced by
minimal stand-ins, over a corpus of outputs, and writes what the parser's
non-streaming `parse` returns for each. Dev-only: the output is committed, and
`cargo test` never runs Python.

    uv run --script dev/qwen_gen_vllm_fixtures.py \\
        --vllm-src /path/to/vllm --vllm-commit <commit sha of the checkout> \\
        --model-dir /path/to/Qwen3.8-27B-FP8

The well-formed outputs are what the model's template renders for an assistant
turn after its generation prompt (`transformers.apply_chat_template`), so they
are the text a model that follows its trained format writes; the rest are
truncated and malformed variants. Writes `tests/fixtures/qwen/vllm_parser_cases.json`.

The generator executes the vLLM code it is pointed at: run it only on a
checkout you trust.
"""

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import logging
import pathlib
import sys
import types

CRATE = pathlib.Path(__file__).resolve().parent.parent
ENGINE_FILES = [
    "vllm/parser/engine/events.py",
    "vllm/parser/engine/incremental_lexer.py",
    "vllm/parser/engine/parser_engine_config.py",
    "vllm/parser/engine/token_id_scanner.py",
    "vllm/parser/engine/streaming_parser_engine.py",
    "vllm/parser/engine/parser_engine.py",
    "vllm/parser/qwen3.py",
]
# The schema-typing helpers the engine imports from `vllm/tool_parsers/utils.py`,
# compiled out of that file alone (the rest of it imports the serving layer).
UTILS_DEFINITIONS = [
    "_is_json_finite",
    "extract_types_from_schema",
    "_TYPE_ALIASES",
    "coerce_to_schema_type",
    "_is_function_tool",
    "_extract_tool_info",
    "find_tool_properties",
    "find_tool_name",
]

# ---------------------------------------------------------------- stand-ins


def module(name, **members):
    m = types.ModuleType(name)
    m.__dict__.update(members)
    sys.modules[name] = m
    return m


@dataclasses.dataclass
class FunctionCall:
    name: str = ""
    arguments: str = ""
    id: str | None = None


@dataclasses.dataclass
class ToolCall:
    function: FunctionCall = None
    id: str = ""
    type: str = "function"


@dataclasses.dataclass
class DeltaFunctionCall:
    name: str | None = None
    arguments: str | None = None


@dataclasses.dataclass
class DeltaToolCall:
    index: int = 0
    id: str | None = None
    type: str | None = None
    function: DeltaFunctionCall | None = None


@dataclasses.dataclass
class DeltaMessage:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ExtractedToolCallInformation:
    tools_called: bool = False
    tool_calls: list = dataclasses.field(default_factory=list)
    content: str | None = None


@dataclasses.dataclass
class StreamState:
    history_tool_call_cnt: int = 0
    tool_call_id_type: str = "random"


class Parser:
    def _initialize_history_tool_call_cnt(self, request):
        pass


class ChatCompletionToolsParam:
    """The request's `tools` entry, as the engine reads it."""

    def __init__(self, tool):
        function = tool.get("function", {})
        self.function = types.SimpleNamespace(name=function.get("name"), parameters=function.get("parameters"))


class Unused:
    pass


def install_stand_ins():
    module("vllm")
    module("vllm.parser")
    module("vllm.parser.engine")
    module("vllm.entrypoints")
    module("vllm.entrypoints.generate")
    module("vllm.entrypoints.generate.base")
    module(
        "vllm.entrypoints.generate.base.protocol",
        FunctionCall=FunctionCall,
        ToolCall=ToolCall,
        DeltaFunctionCall=DeltaFunctionCall,
        DeltaToolCall=DeltaToolCall,
        DeltaMessage=DeltaMessage,
        ExtractedToolCallInformation=ExtractedToolCallInformation,
    )
    module("vllm.logger", init_logger=logging.getLogger)
    module("vllm.parser.abstract_parser", Parser=Parser, StreamState=StreamState)
    module("vllm.renderers")
    module(
        "vllm.renderers.chat_utils",
        get_tool_call_id_type=lambda model_config: "random",
        make_tool_call_id=lambda id_type, func_name, idx: f"call_{idx}",
    )


def load_utils(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in UTILS_DEFINITIONS:
            keep.append(node)
    found = {getattr(n, "name", None) or getattr(getattr(n, "target", None), "id", None) or n.targets[0].id for n in keep}
    if found != set(UTILS_DEFINITIONS):
        sys.exit(f"vllm/tool_parsers/utils.py: missing {set(UTILS_DEFINITIONS) - found}")
    namespace = {
        "__name__": "vllm.tool_parsers.utils",
        "json": json,
        "math": __import__("math"),
        "Any": object,
        "Tool": object,
        "FunctionTool": Unused,
        "NamespaceTool": Unused,
        "CustomTool": Unused,
        "ChatCompletionToolsParam": ChatCompletionToolsParam,
        "iter_response_function_tool_info": lambda tool: (),
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - pinned reference code
    module("vllm.tool_parsers")
    module("vllm.tool_parsers.utils", **{k: namespace[k] for k in UTILS_DEFINITIONS})


def load_engine(src):
    for rel in ENGINE_FILES:
        name = rel[: -len(".py")].replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, src / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["vllm.parser.qwen3"].Qwen3Parser


class Tokenizer:
    """The engine's view of the tokenizer: the vocabulary, the special tokens,
    and decoding single ids."""

    def __init__(self, hf):
        self.hf = hf
        self.all_special_tokens = list(hf.all_special_tokens)
        self.all_special_ids = list(hf.all_special_ids)

    def get_vocab(self):
        return self.hf.get_vocab()

    def decode(self, ids):
        return self.hf.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------- corpus

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}},
    },
}
TYPED = {
    "type": "function",
    "function": {
        "name": "typed",
        "parameters": {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "o": {"type": "object"},
                "a": {"type": "array"},
            },
        },
    },
}
TOOLS = [WEATHER, BASH, TYPED]

REASONING = [
    "The user wants the weather.",
    "Step 1: parse.\nStep 2: answer.\n\n- bullet\n- bullet",
    "思考：用户问天气。",
    "Consider `x < y` and a <b>tag</b> & entities.",
    "",
]
CONTENT = [
    "Hello!",
    "Paris is the capital of France.",
    "Here is code:\n```python\ndef f():\n    return 1\n```",
    "多行\n回答 😀",
    "",
]
ARGUMENTS = [
    ("get_weather", {"city": "Paris"}),
    ("get_weather", {"city": "São Paulo", "days": 3, "unit": "celsius"}),
    ("get_weather", {"city": "line one\nline two"}),
    ("get_weather", {"city": "  padded  "}),
    ("get_weather", {"city": "<b>&amp;</b>"}),
    ("bash", {"command": "ls -la && echo 'done' | grep -v x"}),
    ("bash", {"command": "cat <<EOF\nhello\nEOF\n", "timeout": 2.5}),
    ("bash", {"command": "echo 中文", "timeout": 30}),
    ("typed", {"s": "x", "i": -7, "n": 0.125, "b": True, "o": {"k": [1, 2], "z": None}, "a": [1, "two", {"x": False}]}),
    ("typed", {"b": False, "o": {}, "a": []}),
    ("typed", {"i": 12345678901234567890, "n": 1e-07}),
    ("typed", {"s": '{"looks": "like json"}'}),
    ("typed", {"s": "42"}),
    ("typed", {}),
]


def tool_calls(args):
    return [{"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": a}} for i, (name, a) in enumerate(args)]


def well_formed(tokenizer):
    """Outputs the template renders: an assistant turn, cut after the
    generation prompt and before its end-of-turn token."""
    out = []
    question = [{"role": "user", "content": "Question?"}]

    def one(name, thinking, assistant, tools):
        kwargs = {} if thinking else {"enable_thinking": False}
        prompt = tokenizer.apply_chat_template(question, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
        full = tokenizer.apply_chat_template(question + [assistant], tools=tools, tokenize=False, **kwargs)
        if not full.startswith(prompt) or not full.endswith("<|im_end|>\n"):
            sys.exit(f"{name}: the turn does not extend the generation prompt")
        out.append({"name": name, "thinking": thinking, "tools": tools, "output": full[len(prompt) : -len("<|im_end|>\n")], "well_formed": True})

    for thinking in (True, False):
        t = "on" if thinking else "off"
        for i, content in enumerate(CONTENT):
            for j, reasoning in enumerate(REASONING if thinking else [""]):
                for tools in (None, TOOLS):
                    label = "tools" if tools else "no_tools"
                    one(f"answer/{t}/{label}/{i}_{j}", thinking, {"role": "assistant", "content": content, "reasoning_content": reasoning}, tools)
        for k, args in enumerate(ARGUMENTS):
            for content in ("", "Let me check."):
                c = "content" if content else "bare"
                one(f"call/{t}/{c}/{k}", thinking, {"role": "assistant", "content": content, "reasoning_content": "Need a tool." if thinking else "", "tool_calls": tool_calls([args])}, TOOLS)
        for k in range(len(ARGUMENTS) - 2):
            pair = [ARGUMENTS[k], ARGUMENTS[k + 2]]
            one(f"calls/{t}/{k}", thinking, {"role": "assistant", "content": "Two calls.", "reasoning_content": "Plan." if thinking else "", "tool_calls": tool_calls(pair)}, TOOLS)
        one(f"calls/{t}/three", thinking, {"role": "assistant", "content": "", "tool_calls": tool_calls(ARGUMENTS[:3])}, TOOLS)
    return out


def edge_cases():
    """Truncated and malformed outputs, where the parsers' rules can part."""
    call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    texts = {
        "truncated_reasoning": "I am still thinking about",
        "truncated_partial_end_tag": "Thinking</thi",
        "truncated_after_end": "Done.\n</think>\n\n",
        "truncated_content": "Done.\n</think>\n\nThe answer is",
        "truncated_in_call": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
        "truncated_after_function": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
        "call_in_reasoning": "I should call it.\n\n" + call,
        "repeated_think_open": "<think>\nR\n</think>\n\nC",
        "think_open_mid_reasoning": "R1 <think> R2\n</think>\n\nC",
        "repeated_think_end": "R\n</think>\n\nC </think> D",
        "no_reasoning_end": "Just an answer.",
        "function_without_tool_call": "R\n</think>\n\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>",
        "undeclared_function": "R\n</think>\n\n<tool_call>\n<function=nope>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
        "missing_parameter_end": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
        "compact_markup": "R\n</think>\n\n<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
        "spaced_names": "R\n</think>\n\n<tool_call>\n<function= get_weather >\n<parameter= city >\nParis\n</parameter>\n</function>\n</tool_call>",
        "text_after_call": "R\n</think>\n\n" + call + "\nAnd some words after.",
        "text_between_calls": "R\n</think>\n\n" + call + "\nmiddle\n" + call,
        "prose_mentions_tag": "R\n</think>\n\nWrite <tool_call> to call a tool.",
        "repeated_parameter": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>",
        "string_null_for_integer": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nnull\n</parameter>\n<parameter=days>\nnull\n</parameter>\n</function>\n</tool_call>",
        "float_for_integer": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nX\n</parameter>\n<parameter=days>\n4.0\n</parameter>\n</function>\n</tool_call>",
        "boolean_words": "R\n</think>\n\n<tool_call>\n<function=typed>\n<parameter=b>\nTrue\n</parameter>\n</function>\n</tool_call>",
        "undeclared_parameter": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nX\n</parameter>\n<parameter=extra>\n{\"a\": 1}\n</parameter>\n</function>\n</tool_call>",
        "value_two_newlines": "R\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n\nX\n\n</parameter>\n</function>\n</tool_call>",
        "empty_function_name": "R\n</think>\n\n<tool_call>\n<function=>\n</function>\n</tool_call>",
    }
    out = []
    for name, text in texts.items():
        out.append({"name": f"edge/{name}", "thinking": True, "tools": TOOLS, "output": text, "well_formed": False})
    out.append({"name": "edge/thinking_off_tags", "thinking": False, "tools": TOOLS, "output": "<think>\nx\n</think>\n\nAnswer", "well_formed": False})
    out.append({"name": "edge/no_tools_call_markup", "thinking": True, "tools": None, "output": "R\n</think>\n\n" + call, "well_formed": False})
    return out


def parse(Qwen3Parser, tokenizer, hf, case):
    tools = [ChatCompletionToolsParam(t) for t in case["tools"]] if case["tools"] else None
    kwargs = {"chat_template_kwargs": {"enable_thinking": case["thinking"]}}
    parser = Qwen3Parser(tokenizer, tools, **kwargs)
    request = types.SimpleNamespace(tools=tools, tool_choice="auto", include_reasoning=True, chat_template_kwargs=kwargs["chat_template_kwargs"])
    ids = hf.encode(case["output"], add_special_tokens=False)
    reasoning, content, calls = parser.parse(case["output"], request, enable_auto_tools=True, model_output_token_ids=ids)
    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": [[c.name, c.arguments] for c in calls or []],
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vllm-src", required=True, type=pathlib.Path)
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=CRATE / "tests" / "fixtures" / "qwen" / "vllm_parser_cases.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(args.model_dir))
    install_stand_ins()
    load_utils(args.vllm_src / "vllm" / "tool_parsers" / "utils.py")
    Qwen3Parser = load_engine(args.vllm_src)
    tokenizer = Tokenizer(hf)

    cases = well_formed(hf) + edge_cases()
    names = [c["name"] for c in cases]
    if len(names) != len(set(names)):
        sys.exit("duplicate case names")
    for c in cases:
        c["vllm"] = parse(Qwen3Parser, tokenizer, hf, c)
        c["tools"] = None if c["tools"] is None else json.dumps(c["tools"], ensure_ascii=False)

    meta = {
        "generator": "dev/qwen_gen_vllm_fixtures.py",
        "vllm_commit": args.vllm_commit,
        "files_sha256": {rel: sha256(args.vllm_src / rel) for rel in ENGINE_FILES + ["vllm/tool_parsers/utils.py"]},
        "template_sha256": sha256(args.model_dir / "chat_template.jinja"),
        "python": sys.version.split()[0],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(cases)} cases ({sum(c['well_formed'] for c in cases)} well-formed)")


if __name__ == "__main__":
    main()
