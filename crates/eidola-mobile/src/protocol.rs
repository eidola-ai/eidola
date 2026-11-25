//! The shell contract's wire format (`mobile/shell-contract.md` → The bridge,
//! Errors, Methods): how a request is written, how a reply is read, and every
//! way a reply can be wrong, as a typed error.
//!
//! Nothing here trusts the shell's text: a reply that is not JSON, lacks its
//! result, carries bytes that are not base64 or numbers out of range is a
//! [`ShellError::Malformed`], never a panic.

use std::fmt;

use base64::Engine as _;
use base64::display::Base64Display;
use base64::engine::general_purpose::STANDARD;
use serde::de::{self, DeserializeOwned, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The contract version this app is written against: 2 added
/// `shell.openExternal`, which the app uses.
pub const CONTRACT: u64 = 2;

/// Every method the app calls. A shell missing one cannot run it.
pub const REQUIRED_METHODS: &[&str] = &[
    "bundle.activate",
    "bundle.begin",
    "bundle.confirm",
    "bundle.remove",
    "bundle.seal",
    "bundle.state",
    "bundle.write",
    "files.list",
    "files.read",
    "files.remove",
    "files.rename",
    "files.sync",
    "files.truncate",
    "files.write",
    "shell.hello",
    "shell.openExternal",
    "tcp.close",
    "tcp.connect",
    "tcp.receive",
    "tcp.send",
    "tcp.shutdown",
];

/// The most bytes one request may carry (`data`, decoded).
pub const MAX_REQUEST_BYTES: usize = 8 << 20;

/// An error code the shell answered with (`shell-contract.md` → Errors).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Code {
    BadRequest,
    UnknownMethod,
    NotFound,
    InvalidState,
    ConnectFailed,
    Closed,
    Io,
    /// A code this app does not know: a newer shell's addition.
    Other(String),
}

impl Code {
    fn parse(code: &str) -> Self {
        match code {
            "bad_request" => Self::BadRequest,
            "unknown_method" => Self::UnknownMethod,
            "not_found" => Self::NotFound,
            "invalid_state" => Self::InvalidState,
            "connect_failed" => Self::ConnectFailed,
            "closed" => Self::Closed,
            "io" => Self::Io,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::BadRequest => "bad_request",
            Self::UnknownMethod => "unknown_method",
            Self::NotFound => "not_found",
            Self::InvalidState => "invalid_state",
            Self::ConnectFailed => "connect_failed",
            Self::Closed => "closed",
            Self::Io => "io",
            Self::Other(code) => code,
        }
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a call to the shell failed.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ShellError {
    /// The shell answered with an error.
    #[error("{method}: {code}: {message}")]
    Refused {
        method: &'static str,
        code: Code,
        message: String,
    },
    /// The message never reached the shell, or its reply never came back:
    /// there is no shell, or the platform refused the message.
    #[error("{method}: the bridge to the shell failed: {message}")]
    Bridge {
        method: &'static str,
        message: String,
    },
    /// The shell's reply is not what the contract says.
    #[error("{method}: malformed reply: {detail}")]
    Malformed {
        method: &'static str,
        detail: String,
    },
    /// The shell speaks a contract this app cannot run on.
    #[error(
        "the shell speaks contract version {contract}, without {missing:?}; this app needs version {CONTRACT} or later"
    )]
    Contract { contract: u64, missing: Vec<String> },
}

impl ShellError {
    /// The shell's code, when it answered with one.
    pub fn code(&self) -> Option<&Code> {
        match self {
            Self::Refused { code, .. } => Some(code),
            _ => None,
        }
    }
}

impl From<ShellError> for std::io::Error {
    fn from(error: ShellError) -> Self {
        use std::io::ErrorKind;
        let kind = match &error {
            ShellError::Refused { code, .. } => match code {
                Code::BadRequest => ErrorKind::InvalidInput,
                Code::UnknownMethod => ErrorKind::Unsupported,
                Code::NotFound => ErrorKind::NotFound,
                Code::ConnectFailed => ErrorKind::ConnectionRefused,
                Code::Closed => ErrorKind::ConnectionAborted,
                Code::InvalidState | Code::Io | Code::Other(_) => ErrorKind::Other,
            },
            ShellError::Bridge { .. } => ErrorKind::BrokenPipe,
            ShellError::Malformed { .. } => ErrorKind::InvalidData,
            ShellError::Contract { .. } => ErrorKind::Unsupported,
        };
        std::io::Error::new(kind, error)
    }
}

/// Bytes written as the contract's base64 (standard alphabet, padded),
/// straight into the request text.
pub struct B64Ref<'a>(pub &'a [u8]);

impl Serialize for B64Ref<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&Base64Display::new(self.0, &STANDARD))
    }
}

/// Bytes read from the contract's base64, decoded as the reply is parsed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct B64(pub Vec<u8>);

impl<'de> Deserialize<'de> for B64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct B64Visitor;
        impl Visitor<'_> for B64Visitor {
            type Value = B64;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a base64 string")
            }
            fn visit_str<E: de::Error>(self, text: &str) -> Result<B64, E> {
                STANDARD
                    .decode(text)
                    .map(B64)
                    .map_err(|e| E::custom(format!("not base64: {e}")))
            }
        }
        deserializer.deserialize_str(B64Visitor)
    }
}

/// The JSON text of request `id`.
pub fn encode<P: Serialize>(id: u64, method: &str, params: &P) -> Result<String, String> {
    #[derive(Serialize)]
    struct Request<'a, P> {
        id: u64,
        method: &'a str,
        params: &'a P,
    }
    serde_json::to_string(&Request { id, method, params }).map_err(|e| e.to_string())
}

/// The `id` a reply echoes, for a transport that matches replies to
/// requests itself. `None` when the text is not a reply with a numeric id.
pub fn reply_id(text: &str) -> Option<u64> {
    // A reply that opens with its id (Android's does) is routed without
    // reading the rest, which may be megabytes of base64.
    if let Some(rest) = text.strip_prefix("{\"id\":") {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if matches!(rest.as_bytes().get(digits), Some(b',' | b'}')) && digits <= 16 {
            return rest[..digits].parse().ok();
        }
    }
    #[derive(Deserialize)]
    struct Id {
        id: Option<u64>,
    }
    serde_json::from_str::<Id>(text)
        .ok()
        .and_then(|reply| reply.id)
}

#[derive(Deserialize)]
struct ErrorBody {
    code: String,
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
#[serde(bound = "T: DeserializeOwned")]
struct Reply<T> {
    ok: bool,
    result: Option<T>,
    error: Option<ErrorBody>,
}

/// Read `method`'s reply: its result, or the error the shell answered with.
pub fn decode<T: DeserializeOwned>(method: &'static str, text: &str) -> Result<T, ShellError> {
    let reply: Reply<T> = serde_json::from_str(text).map_err(|e| ShellError::Malformed {
        method,
        detail: e.to_string(),
    })?;
    match (reply.ok, reply.result, reply.error) {
        (true, Some(result), _) => Ok(result),
        (true, None, _) => Err(ShellError::Malformed {
            method,
            detail: "ok without a result".into(),
        }),
        (false, _, Some(error)) => Err(ShellError::Refused {
            method,
            code: Code::parse(&error.code),
            message: error.message,
        }),
        (false, _, None) => Err(ShellError::Malformed {
            method,
            detail: "an error without its code".into(),
        }),
    }
}

/// A result this app reads nothing from (`{}`).
#[derive(Debug, Deserialize)]
pub struct Empty {}

/// `shell.hello`.
#[derive(Debug, Clone, Deserialize)]
pub struct Hello {
    pub contract: u64,
    pub platform: String,
    pub shell: String,
    pub methods: Vec<String>,
}

impl Hello {
    /// Whether this app can run on the shell: a contract at least
    /// [`CONTRACT`] (versions only ever add, so a later one keeps everything
    /// this app relies on) that lists every method the app calls.
    pub fn check(&self) -> Result<(), ShellError> {
        let missing: Vec<String> = REQUIRED_METHODS
            .iter()
            .filter(|method| !self.methods.iter().any(|m| m == *method))
            .map(|method| (*method).to_owned())
            .collect();
        if self.contract < CONTRACT || !missing.is_empty() {
            return Err(ShellError::Contract {
                contract: self.contract,
                missing,
            });
        }
        Ok(())
    }
}

/// `tcp.receive`: bytes, or the end of the stream.
#[derive(Debug, Deserialize)]
pub struct Received {
    pub data: Option<B64>,
    #[serde(default)]
    pub eof: bool,
}

/// `files.read`: the file, or `None` when it is missing.
#[derive(Debug, Deserialize)]
pub struct FileData {
    pub data: Option<B64>,
}

/// `files.write`: the file's length after the write.
#[derive(Debug, Deserialize)]
pub struct Written {
    pub size: u64,
}

/// `files.list`.
#[derive(Debug, Deserialize)]
pub struct Names {
    pub names: Vec<String>,
}

/// The bundle store's results (`bundle.seal`, `bundle.state`, and the
/// state `bundle.activate` and `bundle.confirm` return) are app-core's own
/// types, which its updater stages through (`eidola_app_core::updater::mobile`):
/// one definition of the contract's shapes, read straight from the reply.
pub use eidola_app_core::updater::mobile::{
    BundleState, SealedBundle as Sealed, ServingBundle as Serving, StoredBundle as StoredVersion,
};

/// `debug.syncLog`, a debug shell's test hook outside the contract: every
/// flush since the last call, as paths relative to the shell's storage.
#[derive(Debug, Deserialize)]
pub struct SyncLog {
    pub syncs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_carries_its_bytes_as_padded_base64() {
        #[derive(Serialize)]
        struct P<'a> {
            data: B64Ref<'a>,
        }
        let text = encode(
            7,
            "tcp.send",
            &P {
                data: B64Ref(b"hi?"),
            },
        )
        .unwrap();
        assert_eq!(
            text,
            r#"{"id":7,"method":"tcp.send","params":{"data":"aGk/"}}"#
        );
        let text = encode(8, "tcp.send", &P { data: B64Ref(b"h") }).unwrap();
        assert!(text.contains(r#""aA==""#), "{text}");
    }

    #[test]
    fn bytes_decode_through_escaped_slashes() {
        // Both shells' JSON writers may escape `/` as `\/`.
        let got: Received = decode(
            "tcp.receive",
            r#"{"id":1,"ok":true,"result":{"data":"aGk\/"}}"#,
        )
        .unwrap();
        assert_eq!(got.data.unwrap().0, b"hi?");
    }

    #[test]
    fn a_reply_names_its_id() {
        assert_eq!(reply_id(r#"{"ok":true,"result":{},"id":42}"#), Some(42));
        assert_eq!(reply_id(r#"{"ok":true,"id":null}"#), None);
        assert_eq!(reply_id("not json"), None);
        assert_eq!(reply_id(r#"{"id":-1}"#), None);
        assert_eq!(
            reply_id(r#"{"id":7,"ok":true,"result":{"data":"AA=="}}"#),
            Some(7)
        );
        assert_eq!(reply_id(r#"{"id":7}"#), Some(7));
        // Not the opening member's whole value: the full parse decides.
        assert_eq!(reply_id(r#"{"id":7.5,"ok":true}"#), None);
        assert_eq!(reply_id(r#"{"id":7 ,"ok":true}"#), Some(7));
        assert_eq!(reply_id(r#"{"id":"7"}"#), None);
    }

    #[test]
    fn every_wrong_reply_is_a_typed_error() {
        let malformed = |text: &str| match decode::<Received>("tcp.receive", text) {
            Err(ShellError::Malformed { .. }) => {}
            other => panic!("{text}: {other:?}"),
        };
        malformed("");
        malformed("not json");
        malformed("[]");
        malformed(r#"{"id":1,"ok":true}"#);
        malformed(r#"{"id":1,"ok":true,"result":{"data":"!!"}}"#);
        malformed(r#"{"id":1,"ok":true,"result":{"data":7}}"#);
        malformed(r#"{"id":1,"ok":false}"#);
        malformed(r#"{"id":1,"ok":"yes","result":{}}"#);
        match decode::<Written>("files.write", r#"{"id":1,"ok":true,"result":{"size":-3}}"#) {
            Err(ShellError::Malformed { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_error_reply_carries_its_code() {
        let err = decode::<Empty>(
            "tcp.connect",
            r#"{"id":3,"ok":false,"error":{"code":"connect_failed","message":"refused"}}"#,
        )
        .unwrap_err();
        assert_eq!(err.code(), Some(&Code::ConnectFailed));
        assert_eq!(
            std::io::Error::from(err).kind(),
            std::io::ErrorKind::ConnectionRefused
        );
        let err = decode::<Empty>("x.y", r#"{"id":3,"ok":false,"error":{"code":"new_code"}}"#)
            .unwrap_err();
        assert_eq!(err.code(), Some(&Code::Other("new_code".into())));
    }

    #[test]
    fn unknown_result_members_are_ignored() {
        let hello: Hello = decode(
            "shell.hello",
            r#"{"id":1,"ok":true,"result":{"contract":3,"platform":"ios","shell":"1","methods":[],"later":{"x":1}}}"#,
        )
        .unwrap();
        assert_eq!(hello.contract, 3);
    }

    #[test]
    fn a_serving_digest_still_being_computed_reads_as_none() {
        // The Android shell answers `null` until it has hashed the bundle it
        // serves; a state read then must not fail the updater.
        let state: BundleState = decode(
            "bundle.state",
            r#"{"id":1,"ok":true,"result":{"serving":{"version":"embedded","digest":null,"confirmed":true},"active":"embedded","confirmed":true,"previous":"embedded","failed":null,"versions":[{"version":"v1","sealed":false,"digest":null}]}}"#,
        )
        .unwrap();
        assert_eq!(state.serving.digest, None);
        assert_eq!(state.versions[0].digest, None);
    }

    /// The Rust side's share of a receive: routing and decoding a 1 MiB
    /// reply as a shell writes it (slashes escaped), and encoding a 1 MiB
    /// send. `cargo test --release -p eidola-mobile --lib -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn measure_the_bridge_codec() {
        let bytes: Vec<u8> = (0..1u32 << 20).map(|i| (i * 31 % 256) as u8).collect();
        let b64 = STANDARD.encode(&bytes).replace('/', "\\/");
        let reply = format!(r#"{{"id":9,"ok":true,"result":{{"data":"{b64}"}}}}"#);
        let rounds = 50;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            assert_eq!(reply_id(&reply), Some(9));
            let got: Received = decode("tcp.receive", &reply).unwrap();
            assert_eq!(got.data.unwrap().0.len(), 1 << 20);
        }
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
        #[derive(Serialize)]
        struct P<'a> {
            socket: u64,
            data: B64Ref<'a>,
        }
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            let text = encode(
                9,
                "tcp.send",
                &P {
                    socket: 1,
                    data: B64Ref(&bytes),
                },
            )
            .unwrap();
            assert!(text.len() > 1 << 20);
        }
        let encode_ms = started.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
        println!("1 MiB: route + decode {decode_ms:.2} ms, encode {encode_ms:.2} ms");
    }

    fn hello(contract: u64, methods: &[&str]) -> Hello {
        Hello {
            contract,
            platform: "ios".into(),
            shell: "1".into(),
            methods: methods.iter().map(|m| (*m).to_owned()).collect(),
        }
    }

    #[test]
    fn a_shell_is_accepted_only_with_the_contract_the_app_needs() {
        hello(CONTRACT, REQUIRED_METHODS).check().unwrap();
        // Versions only add: a later shell keeps every method.
        let mut later = REQUIRED_METHODS.to_vec();
        later.push("shell.later");
        hello(CONTRACT + 1, &later).check().unwrap();
        // Version 1 had no shell.openExternal.
        let v1: Vec<&str> = REQUIRED_METHODS
            .iter()
            .copied()
            .filter(|m| *m != "shell.openExternal")
            .collect();
        match hello(1, &v1).check() {
            Err(ShellError::Contract {
                contract: 1,
                missing,
            }) => {
                assert_eq!(missing, ["shell.openExternal"]);
            }
            other => panic!("{other:?}"),
        }
        // A version number that claims more than its methods deliver.
        assert!(hello(CONTRACT + 1, &v1).check().is_err());
        let message = hello(1, &v1).check().unwrap_err().to_string();
        assert!(message.contains("contract version 1"), "{message}");
    }
}
