//! End to end: arguments rendered by the chat template and parsed back.
//!
//! Under `ArgumentTyping::RoundTrip` the parser is the exact inverse of the
//! template for schema-conforming arguments. Under the SGLang rule it is not;
//! the counterexamples are pinned here so the difference stays visible.

use eidola_engine_chat::json::{self, BigInt, Json};
use eidola_engine_chat::tool_call::parse_complete;
use eidola_engine_chat::{ArgumentTyping, ChatInput, ChatTemplate, RenderOptions, ToolSchemas};
use proptest::prelude::*;

fn template() -> ChatTemplate {
    ChatTemplate::from_model_dir(
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
    )
    .unwrap()
}

const PARAM_TYPES: &[&str] = &[
    "string",
    "integer",
    "number",
    "boolean",
    "object",
    "array",
    "nullable_int",
];

fn tools_json() -> Json {
    let props: Vec<(String, Json)> = PARAM_TYPES
        .iter()
        .map(|t| {
            let ty = if *t == "nullable_int" {
                Json::Array(vec![Json::Str("integer".into()), Json::Str("null".into())])
            } else {
                Json::Str((*t).into())
            };
            (format!("p_{t}"), Json::Object(vec![("type".into(), ty)]))
        })
        .collect();
    json::parse(&format!(
        r#"[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object","properties":{}}}}}}}]"#,
        json::dumps_default(&Json::Object(props))
    ))
    .unwrap()
}

/// Renders a single call and returns the `<tool_call>` block of the prompt.
fn render_call(template: &ChatTemplate, arguments: &Json) -> String {
    let messages = Json::Array(vec![
        Json::Object(vec![
            ("role".into(), Json::Str("user".into())),
            ("content".into(), Json::Str("x".into())),
        ]),
        Json::Object(vec![
            ("role".into(), Json::Str("assistant".into())),
            ("content".into(), Json::Null),
            (
                "tool_calls".into(),
                Json::Array(vec![Json::Object(vec![(
                    "function".into(),
                    Json::Object(vec![
                        ("name".into(), Json::Str("f".into())),
                        ("arguments".into(), arguments.clone()),
                    ]),
                )])]),
            ),
        ]),
    ]);
    let input = ChatInput::from_json(&json::dumps_default(&messages), None).unwrap();
    let prompt = template.render(&input, RenderOptions::default()).unwrap();
    let start = prompt.find("<tool_call>").unwrap();
    let end = prompt.rfind("</tool_call>").unwrap() + "</tool_call>".len();
    prompt[start..end].to_string()
}

fn parse(block: &str, typing: ArgumentTyping) -> String {
    let schemas = ToolSchemas::from_tools(&tools_json()).with_typing(typing);
    let parsed = parse_complete(block, &schemas);
    assert_eq!(parsed.calls.len(), 1, "{block}");
    parsed.calls[0].1.clone()
}

/// Strings the format can carry: anything without a closing tag.
fn safe_string() -> impl Strategy<Value = String> {
    prop_oneof![
        "\\PC{0,24}",
        Just("null".to_string()),
        Just("None".to_string()),
        Just("&amp; &lt;b&gt; &copy=2".to_string()),
        Just("{\"a\": 1}".to_string()),
        Just("  padded\n".to_string()),
        Just(String::new()),
    ]
    .prop_filter("closing tags cannot be represented", |s| {
        !s.contains("</parameter>") && !s.contains("</function>") && !s.contains("</tool_call>")
    })
}

fn json_value() -> impl Strategy<Value = Json> {
    let leaf = prop_oneof![
        Just(Json::Null),
        any::<bool>().prop_map(Json::Bool),
        any::<i64>()
            .prop_map(|i| Json::Int(BigInt::from_digits(i < 0, &i.unsigned_abs().to_string()))),
        any::<f64>()
            .prop_filter("finite", |f| f.is_finite())
            .prop_map(Json::Float),
        safe_string().prop_map(Json::Str),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..4).prop_map(Json::Array),
            proptest::collection::vec(("[a-z]{1,3}", inner), 0..4).prop_map(|members| {
                let mut seen = Vec::new();
                Json::Object(
                    members
                        .into_iter()
                        .filter(|(k, _)| {
                            let fresh = !seen.contains(k);
                            seen.push(k.clone());
                            fresh
                        })
                        .collect(),
                )
            }),
        ]
    })
}

fn arguments() -> impl Strategy<Value = Json> {
    (
        safe_string(),
        any::<i64>(),
        any::<f64>().prop_filter("finite", |f| f.is_finite()),
        any::<bool>(),
        proptest::collection::vec(("[a-z]{1,3}", json_value()), 0..3),
        proptest::collection::vec(json_value(), 0..3),
        proptest::option::of(any::<i32>()),
    )
        .prop_map(|(s, i, f, b, o, a, n)| {
            let mut seen = Vec::new();
            let object = o
                .into_iter()
                .filter(|(k, _)| {
                    let fresh = !seen.contains(k);
                    seen.push(k.clone());
                    fresh
                })
                .collect();
            Json::Object(vec![
                ("p_string".into(), Json::Str(s)),
                (
                    "p_integer".into(),
                    Json::Int(BigInt::from_digits(i < 0, &i.unsigned_abs().to_string())),
                ),
                ("p_number".into(), Json::Float(f)),
                ("p_boolean".into(), Json::Bool(b)),
                ("p_object".into(), Json::Object(object)),
                ("p_array".into(), Json::Array(a)),
                (
                    "p_nullable_int".into(),
                    n.map_or(Json::Null, |v| {
                        Json::Int(BigInt::from_digits(v < 0, &v.unsigned_abs().to_string()))
                    }),
                ),
            ])
        })
}

proptest! {
    #[test]
    fn round_trip_typing_inverts_the_template(args in arguments()) {
        let block = render_call(&template(), &args);
        prop_assert_eq!(parse(&block, ArgumentTyping::RoundTrip), json::dumps_default(&args));
    }
}

#[test]
fn sglang_typing_counterexamples() {
    let template = template();
    let cases = [
        (
            "p_string",
            Json::Str("&amp; &copy=2".into()),
            r#"{"p_string": "& ©=2"}"#,
        ),
        (
            "p_string",
            Json::Str("NULL".into()),
            r#"{"p_string": null}"#,
        ),
        ("p_number", Json::Float(4.0), r#"{"p_number": 4}"#),
    ];
    for (name, value, sglang) in cases {
        let args = Json::Object(vec![(name.into(), value)]);
        let block = render_call(&template, &args);
        assert_eq!(parse(&block, ArgumentTyping::Sglang), sglang);
        assert_eq!(
            parse(&block, ArgumentTyping::RoundTrip),
            json::dumps_default(&args)
        );
    }
    // A non-boolean word for a boolean parameter: SGLang says false.
    let block =
        "<tool_call><function=f><parameter=p_boolean>yes</parameter></function></tool_call>";
    assert_eq!(
        parse(block, ArgumentTyping::Sglang),
        r#"{"p_boolean": false}"#
    );
    assert_eq!(
        parse(block, ArgumentTyping::RoundTrip),
        r#"{"p_boolean": "yes"}"#
    );
}
