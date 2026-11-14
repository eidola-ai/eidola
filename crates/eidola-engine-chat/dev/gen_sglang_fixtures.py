#!/usr/bin/env python3
"""Generate differential fixtures for the tool-call parser from SGLang's
MiMo detector.

Executes the real SGLang sources (`python/sglang/srt/function_call/*.py` at
the commit pinned in requirements.txt) with the rest of the package stubbed
out, so the expected outputs are the reference implementation's, not a
re-implementation's. Dev-only: outputs are committed; `cargo test` never runs
Python.

    python -I dev/gen_sglang_fixtures.py --sglang-src /path/to/sglang --sglang-commit <sha>

`--sglang-src` is a checkout (or an export of `python/sglang/srt/function_call/`
laid out at the same relative path) of the pinned commit.

Tool schemas are normalised with SGLang's own `normalize_json_schema_types`,
as SGLang's request validation does before the detector sees them.

Where SGLang would fail the request or emit invalid JSON, the Rust parser
contains the failure to the one parameter (documented in src/args.rs). The
harness applies the same containment around SGLang's `_convert_param_value`
so those cases still compare value for value, and marks each case where it
fired (`contained: true`).

Writes tests/fixtures/sglang_param_cases.json and
tests/fixtures/sglang_text_cases.json.
"""

import argparse
import html
import importlib.util
import itertools
import json
import logging
import pathlib
import random
import sys
import types

CRATE = pathlib.Path(__file__).resolve().parent.parent
SEED = 20261007


def load_sglang(src):
    """Import SGLang's function_call modules with minimal stubs."""
    fc_dir = src / "python" / "sglang" / "srt" / "function_call"

    def package(name):
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        return mod

    for name in ["sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai", "sglang.srt.function_call"]:
        package(name)

    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")

    class Function:
        def __init__(self, name, parameters):
            self.name = name
            self.parameters = parameters

    class Tool:
        def __init__(self, name, parameters):
            self.type = "function"
            self.function = Function(name, parameters)

    class ToolChoice:
        pass

    protocol.Tool = Tool
    protocol.ToolChoice = ToolChoice
    sys.modules[protocol.__name__] = protocol

    environ = types.ModuleType("sglang.srt.environ")

    class _Flag:
        def get(self):
            return False

    environ.envs = types.SimpleNamespace(SGLANG_FORWARD_UNKNOWN_TOOLS=_Flag())
    sys.modules[environ.__name__] = environ

    loaded = {}
    for name in ["core_types", "utils", "base_format_detector", "mimo_detector"]:
        full = f"sglang.srt.function_call.{name}"
        spec = importlib.util.spec_from_file_location(full, fc_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return Tool, loaded


# ---------------------------------------------------------------- inputs

TYPES = [
    "string", "integer", "number", "boolean", "object", "array", "null",
    "str", "text", "varchar(255)", "VARCHAR", "char", "enum", "uuid", "date", "binary", "bytes",
    "int", "int32", "int64", "uint8", "long", "short", "unsigned", "bigint", "Integer", "internal",
    "num", "numeric", "float", "float64", "double", "decimal", "real",
    "bool", "Boolean",
    "arr", "tuple", "set", "list", "list[str]", "List[int]", "dict", "dict[str, int]", "map",
    "custom_type", "Object", "ARRAY",
    ["integer", "null"], ["null", "number"], ["string", "integer"], ["boolean"], ["null"], [],
    None,  # property without "type"
]

VALUES = [
    # strings / words
    "", " ", "hello", "hello world", "  padded  ", "\n", "line1\nline2", "中文", "😀", "עברית",
    "x" * 50, "None", "none", "null", "NULL", "Null", " null", "nil", "undefined",
    # html
    "&amp;", "a &amp;&amp; b", "&lt;div&gt;", "&copy=2", "?a=1&copy=2&reg=3", "&notit;", "&#65;",
    "&#x42;", "&#0;", "&#x80;", "&#xd800;", "&#1;", "&#99999999999;", "&unknown;", "& amp",
    "&#;", "&#x;", "&AMP;", "&ampx", "&amp", "&#" + "9" * 5000 + ";",
    # numbers
    "0", "1", "-1", "+1", "42", " 42 ", "007", "-0", "1_000", "1__0", "_1", "1_", "٣٤", "５",
    "1.0", "1.5", "-2.25", ".5", "5.", "1e3", "1E-3", "1e400", "-1e400", "1e-400", "1.5e300",
    "inf", "-inf", "Infinity", "nan", "NaN", "-nan", "0x10", "0o7", "0b11", "1j", "1,000",
    "12345678901234567890", "9" * 400, "9" * 4301, "3.141592653589793", "0.1", "1e16", "1e15",
    "123456789.125", "2.5e-7", "1_0.5", "4.0", "-0.0", "\u3000 7 \u3000",
    # booleans
    "true", "false", "True", "False", "TRUE", "yes", "no", "1 ", "t",
    # json
    "{}", "[]", "{\"a\": 1}", "{\"a\": [1, 2.5, null, true]}", "[1, \"two\", {\"three\": 3}]",
    "{\"b\": 1, \"a\": 2, \"b\": 3}", "[NaN]", "[Infinity, -Infinity]", "{\"x\": 1e400}",
    "[123456789012345678901234567890]", "{\"u\": \"\\u00e9\\ud83d\\ude00\"}", "{\"s\": \"\\ud800\"}",
    "[1, 2,]", "{'a': 1}", "{\"a\": 1} trailing", "  [1]  ", "\"just a string\"", "5", "true",
    "null ", "[[[[[[[[[[1]]]]]]]]]]",
    # python literals
    "{'a': True, 'b': None, 'c': (1, 2)}", "['x', 'y',]", "(1,)", "()", "1, 2", "{1, 2}", "set()",
    "{1: 'a', '1': 'b'}", "{1: 'a', True: 'b', 1.0: 'c'}", "{0: 'z', False: 'y', -0.0: 'x', 0.5: 'h'}",
    "{None: 1, True: 2, 2.5: 3}", "{(1, 2): 3}", "{[1]: 2}", "b'bytes'", "u'unicode'", "r'raw\\n'",
    "'a' 'b'", "'a' b'b'", "'''triple\nquoted'''", "\"\"\"x\"\"\"", "'esc \\x41 \\u00e9 \\U0001F600 \\101 \\q'",
    "'\\N{BULLET}'", "'\\ud800'", "'\\x4'", "'unterminated", "'a\nb'", "-5", "-(5)", "--5", "+3.5",
    "-True", "1+2", "1+2j", "...", "[1, # comment\n 2]", "\n\n[1]", "\n [1]", "[1]\n", "[1]\n2",
    "f'x'", "lambda: 1", "x", "[x]", "{**a}", "[*a]", "0x_ff", "0b1_0", "1_000.5", "00", "01",
    "1.real", "[1][0]", "print(1)", "'a' # comment", "(\n1,\n2\n)", "[1\\\n, 2]", "\\\n[1]",
    "{'k': [1, {'n': (None, False)}]}", "[" * 150 + "]" * 150, "[" * 250 + "]" * 250,
    "{\"deep\": " + "[" * 300 + "]" * 300 + "}", "\ufeff[1]", "[1]\x00", "\t[1]", "\x0c[1]",
    "{'a': 1e999}", "[1e999]", "0" * 4400, "0x" + "f" * 4000, "'a\\\nb'", "r'a\\\nb'", "'\\\\'",
    "'\r'", "[1,\r\n2]",
]

FUZZ_TYPES = ["object", "array", "integer", "number", "boolean", "custom_type"]
FUZZ_ALPHABET = "()[]{},:'\"\\#\n -+.0eEjx_"


def fuzz_values(rng, n):
    """Random Python-literal / JSON / numeric texts, some of them mutated."""

    def number():
        kind = rng.randrange(9)
        if kind == 0:
            return str(rng.randint(-(10**30), 10**30))
        if kind == 1:
            return repr(rng.uniform(-1e6, 1e6))
        if kind == 2:
            return f"{rng.randint(0, 999)}_{rng.randint(0, 999):03d}"
        if kind == 3:
            return rng.choice(["0x", "0X", "0o", "0b"]) + rng.choice(["1f", "_7", "10", "777", "1_0"])
        if kind == 4:
            return f"{rng.randint(0, 99)}.{rng.randint(0, 99)}e{rng.choice(['', '+', '-'])}{rng.randint(0, 400)}"
        if kind == 5:
            return rng.choice([".5", "5.", "1j", "2.5J", "00", "0_0", "1e5", "1E-5", "007.5"])
        if kind == 6:
            return rng.choice(["-", "+", "- ", "-(", ""]) + str(rng.randint(0, 9)) + ("" if rng.random() < 0.8 else ")")
        if kind == 7:
            return rng.choice(["inf", "nan", "Infinity", "NaN", "-Infinity", "1e999"])
        return "".join(rng.choice("0123456789_٣５") for _ in range(rng.randrange(1, 6)))

    pieces = ["a", " ", "é", "😀", "\\n", "\\x41", "\\u00e9", "\\101", "\\'", '\\"', "\\q", "'", '"', "#", "\\\n"]

    def string():
        body = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 6)))
        prefix = rng.choice(["", "", "", "r", "u", "R", "b", "f", "rb", "U"])
        quote = rng.choice(["'", '"', "'" * 3, '"' * 3])
        return prefix + quote + body + quote

    def value(depth):
        kind = rng.randrange(8 if depth < 4 else 5)
        if kind == 0 or kind == 4:
            return number()
        if kind == 1:
            return string()
        if kind == 2:
            return rng.choice(["True", "False", "None", "true", "null", "...", "set()", "x"])
        if kind == 3:
            return string() + rng.choice([" ", "", "\n"]) + string()
        items = [value(depth + 1) for _ in range(rng.randrange(0, 4))]
        sep = rng.choice([", ", ",", " , ", ",\n", ", # c\n"])
        trail = rng.choice(["", ",", ""])
        if kind == 5:
            return "[" + sep.join(items) + trail + "]"
        if kind == 6:
            return "(" + sep.join(items) + trail + ")"
        keys = [rng.choice([string(), number(), "None", "True", "(1, 2)", "[1]"]) for _ in items]
        return "{" + sep.join(f"{k}{rng.choice([': ', ':'])}{v}" for k, v in zip(keys, items)) + trail + "}"

    out = []
    for _ in range(n):
        text = value(0)
        if rng.random() < 0.25:
            chars = list(text)
            for _ in range(rng.randrange(1, 3)):
                pos = rng.randrange(len(chars) + 1)
                op = rng.randrange(3)
                if op == 0 and pos < len(chars):
                    del chars[pos]
                elif op == 1:
                    chars.insert(pos, rng.choice(FUZZ_ALPHABET))
                elif pos < len(chars):
                    chars[pos] = rng.choice(FUZZ_ALPHABET)
            text = "".join(chars)
        if rng.random() < 0.1:
            text = rng.choice([" ", "\t", "\n", "  \n"]) + text
        out.append(text)
    return out


def tools_for(param_type):
    schema = {} if param_type is None else {"type": param_type}
    return [
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"p": schema}}}},
    ]


def text_tools():
    return [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"type": "string", "enum": ["c", "f"]},
            "lat": {"type": "number"}, "precise": {"type": "boolean"}}}}},
        {"type": "function", "function": {"name": "run", "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string"}, "env": {"type": "object"}, "args": {"type": "array"}, "timeout": {"type": ["number", "null"]}}}}},
        {"type": "function", "function": {"name": "union", "parameters": {"anyOf": [
            {"type": "object", "properties": {"kind": {"type": "string"}, "n": {"type": "integer"}}},
            {"type": "object", "properties": {"n": {"type": "string"}, "flag": {"type": "bool"}}}]}}},
        {"type": "function", "function": {"name": "noparams"}},
        {"type": "function", "function": {"name": "odd", "parameters": {"type": "object", "properties": {"t": True, "u": {"description": "no type"}}}}},
        {"type": "function", "function": {"name": "dup", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "dup", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}},
    ]


def text_cases(rng):
    blocks = [
        "<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></function></tool_call>",
        "<tool_call><function=get_weather><parameter=city>&amp;Ville</parameter><parameter=lat>48.85</parameter><parameter=precise>True</parameter><parameter=unit>c</parameter></function></tool_call>",
        "<tool_call><function=run><parameter=cmd>ls -la && echo \"done\" > /tmp/x</parameter><parameter=env>{\"A\": \"1\"}</parameter><parameter=args>['-l', 2]</parameter><parameter=timeout>null</parameter></function></tool_call>",
        "<tool_call>\n<function=run>\n<parameter=cmd>\npwd\n</parameter>\n</function>\n</tool_call>",
        "<tool_call><function=union><parameter=kind>k</parameter><parameter=n>5</parameter><parameter=flag>false</parameter></function></tool_call>",
        "<tool_call><function=noparams></function></tool_call>",
        "<tool_call><function=noparams><parameter=extra>1</parameter></function></tool_call>",
        "<tool_call><function=odd><parameter=t>1</parameter><parameter=u>[1]</parameter></function></tool_call>",
        "<tool_call><function=dup><parameter=a>7</parameter><parameter=a>8</parameter></function></tool_call>",
        "<tool_call><function= get_weather ><parameter= city >  Rome </parameter></function></tool_call>",
        "<tool_call><function=unknown_fn><parameter=x>1</parameter></function></tool_call>",
        "<tool_call><function=get_weather><parameter=city>no close</function></tool_call>",
        "<tool_call><function=get_weather><parameter=city>a<parameter=days>2</parameter></function></tool_call>",
        "<tool_call>no function here</tool_call>",
        "<tool_call><function=>empty name</function></tool_call>",
        "<tool_call><function=get_weather>missing function close</tool_call>",
        "<tool_call><function=run><parameter=>x</parameter><parameter=cmd>y</parameter></function></tool_call>",
        "<tool_call><function=run><parameter=cmd><tool_call>nested</parameter></function></tool_call>",
        "<tool_call><function=<function=run>><parameter=cmd>z</parameter></function></tool_call>",
        "<tool_call><function=run><parameter=cmd>中文 😀</parameter></function></tool_call>",
        "<tool_call><function=get_weather><parameter=days>1e400</parameter><parameter=lat>inf</parameter></function></tool_call>",
        "<tool_call><function=run><parameter=args>[NaN]</parameter><parameter=env>{1, 2}</parameter></function></tool_call>",
        "<tool_call><function=run><parameter=timeout>3.0</parameter></function></tool_call>",
        "<tool_call>",
        "</tool_call>",
        "<tool_call><function=run><parameter=cmd>unclosed",
    ]
    prefixes = ["", "Sure, calling.", "I'll check.\n\n", "<think>leftover</think>", "中文内容 ", "text with </tool_call> stray "]
    gaps = ["", "\n", " between ", "\n\n"]
    cases = []
    for prefix, block in itertools.product(prefixes[:3], blocks):
        cases.append(prefix + block)
    for prefix in prefixes:
        cases.append(prefix)
        cases.append(prefix + blocks[0] + "\ntrailing text")
    for _ in range(300):
        n = rng.randrange(1, 5)
        parts = [rng.choice(prefixes)]
        for i in range(n):
            if i:
                parts.append(rng.choice(gaps))
            parts.append(rng.choice(blocks))
        if rng.random() < 0.3:
            parts.append(rng.choice(gaps + ["tail", "<tool_", "<tool_call><function=run>"]))
        cases.append("".join(parts))
    # Random parameter blocks drawing from the value corpus.
    names = ["city", "days", "lat", "precise", "unit", "cmd", "env", "args", "timeout", "kind", "n", "flag", "zzz"]
    funcs = ["get_weather", "run", "union", "odd"]
    for _ in range(300):
        params = "".join(
            f"<parameter={rng.choice(names)}>{rng.choice(SAFE_VALUES)}</parameter>" for _ in range(rng.randrange(0, 5))
        )
        cases.append(f"<tool_call><function={rng.choice(funcs)}>{params}</function></tool_call>")
    return cases


# Values that cannot contain markup closing a parameter early, for text cases.
SAFE_VALUES = [v for v in VALUES if "</" not in v and len(v) < 500]

# Known divergences, excluded from the corpus and pinned by Rust-side tests:
# `\N{...}` escapes (no Unicode name table), JSON nesting deeper than 256
# (CPython's limit is environment-dependent).
EXCLUDED_VALUES = {"'\\N{BULLET}'", "{\"deep\": " + "[" * 300 + "]" * 300 + "}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sglang-src", required=True, type=pathlib.Path)
    parser.add_argument("--sglang-commit", required=True, help="commit the sources were taken from")
    parser.add_argument("--out-dir", type=pathlib.Path, default=CRATE / "tests" / "fixtures")
    args = parser.parse_args()

    commit = args.sglang_commit
    logging.disable(logging.CRITICAL)
    Tool, mods = load_sglang(args.sglang_src)
    mimo = mods["mimo_detector"]
    utils = mods["utils"]

    original = mimo._convert_param_value
    state = {"contained": False}

    def contained(param_value, param_name, func_name, tools):
        try:
            value = original(param_value, param_name, func_name, tools)
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            return value
        except Exception:  # noqa: BLE001 - the containment rule under test
            state["contained"] = True
            try:
                return html.unescape(param_value)
            except Exception:  # noqa: BLE001
                return param_value

    mimo._convert_param_value = contained

    def resolve_type_rule(schema_tools):
        """Apply the Rust rule for `type` values SGLang cannot handle
        (non-string types, non-object property schemas)."""
        for tool in schema_tools:
            params = tool["function"].get("parameters")
            for props in iter_properties(params):
                for name, prop in list(props.items()):
                    if not isinstance(prop, dict):
                        props[name] = {}
                        continue
                    t = prop.get("type")
                    if isinstance(t, list):
                        strs = [x for x in t if isinstance(x, str)]
                        if "string" in strs:
                            prop["type"] = "string"
                        else:
                            rest = [x for x in strs if x != "null"]
                            prop["type"] = rest[0] if rest else "string"
                    elif "type" in prop and not isinstance(t, str):
                        prop["type"] = "string"
        return schema_tools

    def iter_properties(schema):
        if not isinstance(schema, dict):
            return
        props = schema.get("properties")
        if isinstance(props, dict):
            yield props
            return
        for key in ("anyOf", "oneOf", "allOf"):
            for branch in schema.get(key) or []:
                yield from iter_properties(branch)

    def sglang_tools(raw_tools):
        normalized = json.loads(json.dumps(raw_tools))
        for tool in normalized:
            if tool["function"].get("parameters") is not None:
                utils.normalize_json_schema_types(tool["function"]["parameters"])
        resolve_type_rule(normalized)
        return [Tool(t["function"]["name"], t["function"].get("parameters")) for t in normalized]

    detector = mimo.MiMoDetector()

    # Parameter-level cases: every value against every declared type. An
    # expected value of null means "the raw text as a JSON string"
    # (decided here by comparison, to keep the file small).
    values = [v for v in VALUES if v not in EXCLUDED_VALUES]
    param_types = []
    for param_type in TYPES:
        raw_tools = tools_for(param_type)
        tools = sglang_tools(raw_tools)
        expected = []
        contained_flags = []
        for value in values:
            state["contained"] = False
            text = json.dumps(contained(value, "p", "f", tools), ensure_ascii=False)
            expected.append(None if text == json.dumps(value, ensure_ascii=False) else text)
            contained_flags.append(state["contained"])
        param_types.append({
            "tools": json.dumps(raw_tools, ensure_ascii=False),
            "expected": expected,
            "contained": contained_flags,
        })
    fuzz = fuzz_values(random.Random(SEED + 2), 3000)
    fuzz_types = []
    for param_type in FUZZ_TYPES:
        raw_tools = tools_for(param_type)
        tools = sglang_tools(raw_tools)
        expected = []
        contained_flags = []
        for value in fuzz:
            state["contained"] = False
            text = json.dumps(contained(value, "p", "f", tools), ensure_ascii=False)
            expected.append(None if text == json.dumps(value, ensure_ascii=False) else text)
            contained_flags.append(state["contained"])
        fuzz_types.append({
            "tools": json.dumps(raw_tools, ensure_ascii=False),
            "expected": expected,
            "contained": contained_flags,
        })
    param_cases = [c for t in param_types + fuzz_types for c in t["contained"]]

    # Whole-output cases through detect_and_parse.
    rng = random.Random(SEED)
    raw_tools = text_tools()
    tools = sglang_tools(raw_tools)
    text_out = []
    for text in text_cases(rng):
        state["contained"] = False
        result = detector.detect_and_parse(text, tools)
        text_out.append({
            "text": text,
            "expected_content": result.normal_text,
            "expected_calls": [[c.name, c.parameters] for c in result.calls],
            "contained": state["contained"],
        })

    meta = {
        "generator": "dev/gen_sglang_fixtures.py",
        "python": sys.version.split()[0],
        "sglang_commit": commit,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "sglang_param_cases.json", "w", encoding="utf-8") as f:
        sections = [{"values": values, "types": param_types}, {"values": fuzz, "types": fuzz_types}]
        json.dump({"meta": meta, "sections": sections}, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    with open(args.out_dir / "sglang_text_cases.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tools": json.dumps(raw_tools, ensure_ascii=False), "cases": text_out}, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(
        f"{len(param_cases)} parameter cases ({sum(param_cases)} contained), "
        f"{len(text_out)} text cases ({sum(c['contained'] for c in text_out)} contained)"
    )


if __name__ == "__main__":
    main()
