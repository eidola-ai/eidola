//! Refuse the YAML constructs an engine config has no use for, read from a
//! real parser's event stream.
//!
//! Tinfoil decodes a config with Go's `yaml.v3` and this crate with
//! `serde_yaml` (libyaml). The two agree on plain block mappings, sequences and
//! scalars, which is all an engine config needs; the constructs refused here
//! are where they can disagree, or where one node can stand for another:
//!
//! - anchors and aliases (Tinfoil's shim decoder refuses aliases, shim.go
//!   `validateYAMLTree`; an alias lets one value stand in for another field's);
//! - explicit tags (`!!str`, `!custom`) and `%YAML` / `%TAG` directives, which
//!   change how a scalar resolves;
//! - merge keys (`<<`), which `yaml.v3` applies and `serde_yaml` does not;
//! - any mapping key that is not a plain scalar: a quoted key, or a complex
//!   (`?`-introduced mapping or sequence) key;
//! - a key set twice in one mapping (`yaml.v3` refuses it; `serde_yaml` keeps
//!   the last when it fills a map);
//! - an env value that is not a quoted scalar. A plain scalar is resolved to
//!   a type by each parser's own rules before it becomes the text the node
//!   reads (`0100` is a string to `serde_yaml` and octal 64 to a YAML 1.1
//!   reader, `1_000` a string or 1000), and a quoted one is its text to both.
//!
//! The events come from `unsafe-libyaml`, the parser `serde_yaml` itself is
//! built on (already in every graph that enables this module), driven through
//! its C-shaped API exactly as `serde_yaml` drives it.

use std::collections::BTreeSet;
use std::mem::MaybeUninit;

use unsafe_libyaml as libyaml_unsafe;

/// Owns an initialized libyaml parser and deletes it on drop.
struct Parser(Box<MaybeUninit<libyaml_unsafe::yaml_parser_t>>);

impl Drop for Parser {
    fn drop(&mut self) {
        // SAFETY: constructed only after `yaml_parser_initialize` succeeded,
        // and deleted exactly once, here.
        unsafe { libyaml_unsafe::yaml_parser_delete(self.0.as_mut_ptr()) }
    }
}

/// Where a node sits: the next node of a mapping alternates key and value.
enum Frame {
    Mapping {
        expecting_key: bool,
        /// Every key seen.
        keys: BTreeSet<Vec<u8>>,
        /// The pending value's key is [`ENV_KEY`].
        at_env: bool,
        /// An element of an `env` sequence.
        env_entry: bool,
    },
    Sequence {
        /// The value of an `env` key.
        env: bool,
    },
}

/// The key every `NAME: "value"` entry list sits under.
const ENV_KEY: &[u8] = b"env";

/// Refuse anchors, aliases, explicit tags, directives, merge keys, non-plain
/// and repeated keys anywhere in `bytes`, and env values that are not quoted.
/// A document libyaml cannot parse is refused too; the typed parse that
/// follows would refuse it with more detail.
pub(super) fn refuse_divergent_constructs(bytes: &[u8], path: &str) -> Result<(), String> {
    let mut boxed = Box::new(MaybeUninit::<libyaml_unsafe::yaml_parser_t>::uninit());
    // SAFETY: the parser is initialized in place on the heap and never moved;
    // `bytes` outlives it (the `Parser` guard is dropped before this function
    // returns), as `yaml_parser_set_input_string` requires.
    let parser = unsafe {
        if libyaml_unsafe::yaml_parser_initialize(boxed.as_mut_ptr()).fail {
            return Err(format!("{path}: cannot start the YAML parser"));
        }
        libyaml_unsafe::yaml_parser_set_encoding(
            boxed.as_mut_ptr(),
            libyaml_unsafe::YAML_UTF8_ENCODING,
        );
        libyaml_unsafe::yaml_parser_set_input_string(
            boxed.as_mut_ptr(),
            bytes.as_ptr(),
            bytes.len() as u64,
        );
        Parser(boxed)
    };
    let mut parser = parser;
    let mut stack: Vec<Frame> = Vec::new();
    loop {
        let mut event = MaybeUninit::<libyaml_unsafe::yaml_event_t>::uninit();
        // SAFETY: the parser is initialized; `event` is written by a
        // successful `yaml_parser_parse` before it is read, and deleted after.
        let verdict = unsafe {
            if libyaml_unsafe::yaml_parser_parse(parser.0.as_mut_ptr(), event.as_mut_ptr()).fail {
                return Err(format!("{path}: not valid YAML"));
            }
            let e = &*event.as_ptr();
            let line = e.start_mark.line + 1;
            let refuse = |what: &str| Err(format!("{path}: line {line}: {what}"));
            let verdict = match e.type_ {
                libyaml_unsafe::YAML_STREAM_END_EVENT => Ok(true),
                libyaml_unsafe::YAML_DOCUMENT_START_EVENT => {
                    let d = e.data.document_start;
                    if !d.version_directive.is_null()
                        || d.tag_directives.start != d.tag_directives.end
                    {
                        refuse("YAML directives are refused")
                    } else {
                        Ok(false)
                    }
                }
                libyaml_unsafe::YAML_ALIAS_EVENT => refuse("YAML anchors and aliases are refused"),
                libyaml_unsafe::YAML_SCALAR_EVENT => {
                    let s = e.data.scalar;
                    let (is_key, env_value) = match stack.last() {
                        Some(Frame::Mapping {
                            expecting_key,
                            env_entry,
                            ..
                        }) => (*expecting_key, *env_entry && !*expecting_key),
                        _ => (false, false),
                    };
                    let value = std::slice::from_raw_parts(s.value, s.length as usize);
                    let quoted = matches!(
                        s.style,
                        libyaml_unsafe::YAML_SINGLE_QUOTED_SCALAR_STYLE
                            | libyaml_unsafe::YAML_DOUBLE_QUOTED_SCALAR_STYLE
                    );
                    if !s.anchor.is_null() {
                        refuse("YAML anchors and aliases are refused")
                    } else if !s.tag.is_null() {
                        refuse("explicit YAML tags are refused")
                    } else if is_key && s.style != libyaml_unsafe::YAML_PLAIN_SCALAR_STYLE {
                        refuse("mapping keys must be plain scalars")
                    } else if is_key && value == b"<<" {
                        refuse("YAML merge keys are refused")
                    } else if is_key && !new_key(&mut stack, value) {
                        refuse(&format!(
                            "duplicate mapping key {:?}",
                            String::from_utf8_lossy(value)
                        ))
                    } else if env_value && !quoted {
                        refuse(
                            "each env entry must be one NAME: \"value\" pair with a quoted \
                             string value",
                        )
                    } else {
                        node_done(&mut stack);
                        Ok(false)
                    }
                }
                libyaml_unsafe::YAML_SEQUENCE_START_EVENT
                | libyaml_unsafe::YAML_MAPPING_START_EVENT => {
                    let (anchor, tag) = if e.type_ == libyaml_unsafe::YAML_SEQUENCE_START_EVENT {
                        (e.data.sequence_start.anchor, e.data.sequence_start.tag)
                    } else {
                        (e.data.mapping_start.anchor, e.data.mapping_start.tag)
                    };
                    if !anchor.is_null() {
                        refuse("YAML anchors and aliases are refused")
                    } else if !tag.is_null() {
                        refuse("explicit YAML tags are refused")
                    } else if matches!(
                        stack.last(),
                        Some(Frame::Mapping {
                            expecting_key: true,
                            ..
                        })
                    ) {
                        refuse("mapping keys must be plain scalars")
                    } else {
                        let frame = if e.type_ == libyaml_unsafe::YAML_SEQUENCE_START_EVENT {
                            Frame::Sequence {
                                env: matches!(
                                    stack.last(),
                                    Some(Frame::Mapping { at_env: true, .. })
                                ),
                            }
                        } else {
                            Frame::Mapping {
                                expecting_key: true,
                                keys: BTreeSet::new(),
                                at_env: false,
                                env_entry: matches!(
                                    stack.last(),
                                    Some(Frame::Sequence { env: true })
                                ),
                            }
                        };
                        stack.push(frame);
                        Ok(false)
                    }
                }
                libyaml_unsafe::YAML_SEQUENCE_END_EVENT
                | libyaml_unsafe::YAML_MAPPING_END_EVENT => {
                    stack.pop();
                    node_done(&mut stack);
                    Ok(false)
                }
                _ => Ok(false),
            };
            libyaml_unsafe::yaml_event_delete(event.as_mut_ptr());
            verdict
        };
        match verdict {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(e) => return Err(e),
        }
    }
}

/// A node finished: in a mapping, the next node is the other half of a pair.
fn node_done(stack: &mut [Frame]) {
    if let Some(Frame::Mapping { expecting_key, .. }) = stack.last_mut() {
        *expecting_key = !*expecting_key;
    }
}

/// Record `key` in the innermost mapping; false when that mapping has it.
fn new_key(stack: &mut [Frame], key: &[u8]) -> bool {
    let Some(Frame::Mapping { keys, at_env, .. }) = stack.last_mut() else {
        return true;
    };
    *at_env = key == ENV_KEY;
    keys.insert(key.to_vec())
}

#[cfg(test)]
mod tests {
    use super::refuse_divergent_constructs;

    fn check(yaml: &str) -> Result<(), String> {
        refuse_divergent_constructs(yaml.as_bytes(), "config")
    }

    /// An env value is quoted text: a plain scalar is typed by whoever reads
    /// it, and two readers can type it differently.
    #[test]
    fn env_values_are_quoted_scalars() {
        let config = |value: &str| {
            format!(
                "cpus: 16\ncontainers:\n  - name: engine\n    runtime: nvidia\n    env:\n      \
                 - A: \"1\"\n      - B: {value}\n    gpus: all\nshim:\n  paths:\n    - /*\n"
            )
        };
        for quoted in ["\"0100\"", "'0100'", "\"\"", "\"on\"", "'2001-01-01'"] {
            check(&config(quoted)).unwrap_or_else(|e| panic!("{quoted}: {e}"));
        }
        for plain in [
            "0100",
            "1_000",
            "0o17",
            "+16",
            "16",
            "on",
            "~",
            "2001-01-01",
            "cuda",
            "|\n          16",
            ">\n          16",
        ] {
            let err = check(&config(plain)).unwrap_err();
            assert!(
                err.contains("line 7") && err.contains("with a quoted string value"),
                "{plain}: {err}"
            );
        }
        // Flow style is the same nodes.
        check("env: [{A: \"1\"}, {B: '2'}]\n").unwrap();
        assert!(check("env: [{A: \"1\"}, {B: 2}]\n").is_err());
        // Only an env entry's own value: plain scalars elsewhere stand.
        check("env:\n  - A: \"1\"\nother:\n  - B: 2\n").unwrap();
        check("env: plain\nnested:\n  env:\n    C: 3\n").unwrap();
    }

    /// A key appears once in its mapping, at every level.
    #[test]
    fn a_mapping_sets_each_key_once() {
        check("a: 1\nb:\n  a: 2\n  c: [{a: 3}, {a: 4}]\n").unwrap();
        for (twice, line) in [
            ("cpus: 16\ncpus: 32\n", 2),
            ("env:\n  - A: \"1\"\n    A: \"2\"\n", 3),
            ("env: [{A: \"1\", A: \"2\"}]\n", 1),
            ("shim:\n  paths: []\n  port: 1\n  paths: []\n", 4),
        ] {
            let err = check(twice).unwrap_err();
            assert!(
                err.contains(&format!("line {line}: duplicate mapping key")),
                "{twice:?}: {err}"
            );
        }
    }
}
