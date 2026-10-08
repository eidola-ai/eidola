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
//!   (`?`-introduced mapping or sequence) key.
//!
//! The events come from `unsafe-libyaml`, the parser `serde_yaml` itself is
//! built on (already in every graph that enables this module), driven through
//! its C-shaped API exactly as `serde_yaml` drives it.

use std::mem::MaybeUninit;

use unsafe_libyaml as libyaml_unsafe;

/// Owns an initialized libyaml parser and deletes it on drop.
struct Parser(Box<MaybeUninit<sys::yaml_parser_t>>);

impl Drop for Parser {
    fn drop(&mut self) {
        // SAFETY: constructed only after `yaml_parser_initialize` succeeded,
        // and deleted exactly once, here.
        unsafe { sys::yaml_parser_delete(self.0.as_mut_ptr()) }
    }
}

/// Where a node sits: the next node of a mapping alternates key and value.
enum Frame {
    Mapping { expecting_key: bool },
    Sequence,
}

/// Refuse anchors, aliases, explicit tags, directives, merge keys and
/// non-plain keys anywhere in `bytes`. A document libyaml cannot parse is
/// refused too; the typed parse that follows would refuse it with more detail.
pub(super) fn refuse_divergent_constructs(bytes: &[u8], path: &str) -> Result<(), String> {
    let mut boxed = Box::new(MaybeUninit::<sys::yaml_parser_t>::uninit());
    // SAFETY: the parser is initialized in place on the heap and never moved;
    // `bytes` outlives it (the `Parser` guard is dropped before this function
    // returns), as `yaml_parser_set_input_string` requires.
    let parser = unsafe {
        if sys::yaml_parser_initialize(boxed.as_mut_ptr()).fail {
            return Err(format!("{path}: cannot start the YAML parser"));
        }
        sys::yaml_parser_set_encoding(boxed.as_mut_ptr(), sys::YAML_UTF8_ENCODING);
        sys::yaml_parser_set_input_string(boxed.as_mut_ptr(), bytes.as_ptr(), bytes.len() as u64);
        Parser(boxed)
    };
    let mut parser = parser;
    let mut stack: Vec<Frame> = Vec::new();
    loop {
        let mut event = MaybeUninit::<sys::yaml_event_t>::uninit();
        // SAFETY: the parser is initialized; `event` is written by a
        // successful `yaml_parser_parse` before it is read, and deleted after.
        let verdict = unsafe {
            if sys::yaml_parser_parse(parser.0.as_mut_ptr(), event.as_mut_ptr()).fail {
                return Err(format!("{path}: not valid YAML"));
            }
            let e = &*event.as_ptr();
            let line = e.start_mark.line + 1;
            let refuse = |what: &str| Err(format!("{path}: line {line}: {what}"));
            let verdict = match e.type_ {
                sys::YAML_STREAM_END_EVENT => Ok(true),
                sys::YAML_DOCUMENT_START_EVENT => {
                    let d = e.data.document_start;
                    if !d.version_directive.is_null()
                        || d.tag_directives.start != d.tag_directives.end
                    {
                        refuse("YAML directives are refused")
                    } else {
                        Ok(false)
                    }
                }
                sys::YAML_ALIAS_EVENT => refuse("YAML anchors and aliases are refused"),
                sys::YAML_SCALAR_EVENT => {
                    let s = e.data.scalar;
                    let is_key = matches!(
                        stack.last(),
                        Some(Frame::Mapping {
                            expecting_key: true
                        })
                    );
                    if !s.anchor.is_null() {
                        refuse("YAML anchors and aliases are refused")
                    } else if !s.tag.is_null() {
                        refuse("explicit YAML tags are refused")
                    } else if is_key && s.style != sys::YAML_PLAIN_SCALAR_STYLE {
                        refuse("mapping keys must be plain scalars")
                    } else if is_key
                        && std::slice::from_raw_parts(s.value, s.length as usize) == b"<<"
                    {
                        refuse("YAML merge keys are refused")
                    } else {
                        node_done(&mut stack);
                        Ok(false)
                    }
                }
                sys::YAML_SEQUENCE_START_EVENT | sys::YAML_MAPPING_START_EVENT => {
                    let (anchor, tag) = if e.type_ == sys::YAML_SEQUENCE_START_EVENT {
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
                            expecting_key: true
                        })
                    ) {
                        refuse("mapping keys must be plain scalars")
                    } else {
                        stack.push(if e.type_ == sys::YAML_SEQUENCE_START_EVENT {
                            Frame::Sequence
                        } else {
                            Frame::Mapping {
                                expecting_key: true,
                            }
                        });
                        Ok(false)
                    }
                }
                sys::YAML_SEQUENCE_END_EVENT | sys::YAML_MAPPING_END_EVENT => {
                    stack.pop();
                    node_done(&mut stack);
                    Ok(false)
                }
                _ => Ok(false),
            };
            sys::yaml_event_delete(event.as_mut_ptr());
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
    if let Some(Frame::Mapping { expecting_key }) = stack.last_mut() {
        *expecting_key = !*expecting_key;
    }
}
