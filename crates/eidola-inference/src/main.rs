//! eidola-inference — measured boot supervisor for the self-hosted inference
//! container.
//!
//! This binary is PID 1 of the inference container. Its whole job is to make
//! sure the model weights the engine will serve are *exactly* the weights the
//! enclave measurement committed to, then get out of the way:
//!
//! 1. Read `MODEL_URL` + `MODEL_SHA256` from the environment. Both are set in
//!    `tinfoil-config.yml`, whose SHA-256 is bound into the enclave
//!    measurement — so the weight hash is part of what the client attests.
//! 2. If `MODEL_PATH` already holds a file with the expected hash (a warm
//!    volume in dev; enclaves boot cold), skip the fetch.
//! 3. Otherwise stream the weights to `MODEL_PATH.partial`, then re-read the
//!    bytes *from disk* and compare their SHA-256 against `MODEL_SHA256`.
//!    Verifying the on-disk bytes (not the in-flight stream) means the bytes
//!    the engine will mmap are the bytes that were verified.
//! 4. Atomically rename into place and `exec` the engine command given after
//!    `--` on our command line, substituting `{model}` with the verified
//!    path. The engine command line lives in the Containerfile ENTRYPOINT, so
//!    it is bound to the image digest and therefore also measured.
//!
//! Any failure — missing config, network error, hash mismatch — exits
//! non-zero without exec'ing the engine. The container fails closed: no
//! verified weights, no inference.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tracing::{error, info};

/// How often to log download progress, in bytes (512 MiB).
const PROGRESS_INTERVAL: u64 = 512 * 1024 * 1024;

/// Buffer size for the read-back verification pass.
const VERIFY_BUF_SIZE: usize = 4 * 1024 * 1024;

struct Config {
    /// Where to fetch the weights from (https; typically a Hugging Face
    /// `resolve/<revision>` URL or a mirror we control).
    model_url: String,
    /// Expected SHA-256 of the weight file, lowercase hex. Measured via
    /// `tinfoil-config.yml`.
    model_sha256: [u8; 32],
    /// Destination path for the verified weights.
    model_path: PathBuf,
    /// Engine argv (everything after `--`), with `{model}` placeholders.
    /// `INFERENCE_EXTRA_ARGS` (whitespace-split) is appended — Tinfoil
    /// configs can only pass environment, and env declared in
    /// `tinfoil-config.yml` is measured, so tuning flags added there stay
    /// bound to the enclave measurement.
    engine_argv: Vec<String>,
}

impl Config {
    fn load() -> Result<Self, String> {
        let model_url =
            std::env::var("MODEL_URL").map_err(|_| "MODEL_URL environment variable is required")?;

        let sha_hex = std::env::var("MODEL_SHA256")
            .map_err(|_| "MODEL_SHA256 environment variable is required")?;
        let model_sha256 = parse_sha256(&sha_hex)?;

        let model_path = PathBuf::from(
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "/models/model.gguf".to_string()),
        );

        // Engine command: everything after a literal `--` argument.
        let mut args = std::env::args().skip(1);
        match args.next() {
            Some(arg) if arg == "--" => {}
            Some(arg) => {
                return Err(format!(
                    "unexpected argument {arg:?} (usage: eidola-inference -- <engine> [args...])"
                ));
            }
            None => {
                return Err("no engine command given after `--`".to_string());
            }
        }
        let mut engine_argv: Vec<String> = args.collect();
        if engine_argv.is_empty() {
            return Err("no engine command given after `--`".to_string());
        }
        if let Ok(extra) = std::env::var("INFERENCE_EXTRA_ARGS") {
            engine_argv.extend(extra.split_whitespace().map(String::from));
        }

        Ok(Config {
            model_url,
            model_sha256,
            model_path,
            engine_argv,
        })
    }
}

/// Parse a 64-char lowercase/uppercase hex string into a 32-byte digest.
fn parse_sha256(hex_str: &str) -> Result<[u8; 32], String> {
    let bytes =
        hex::decode(hex_str.trim()).map_err(|e| format!("MODEL_SHA256 is not valid hex: {e}"))?;
    bytes
        .try_into()
        .map_err(|_| "MODEL_SHA256 must be exactly 32 bytes (64 hex chars)".to_string())
}

/// Compute the SHA-256 of a file by streaming it from disk.
async fn sha256_file(path: &Path) -> std::io::Result<[u8; 32]> {
    use tokio::io::AsyncReadExt;

    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; VERIFY_BUF_SIZE];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

/// Stream `url` to `dest`, fsync, and return the number of bytes written.
///
/// The transport stream is not the verification pass — after this returns,
/// the caller re-reads `dest` from disk and checks the digest there.
async fn download(client: &reqwest::Client, url: &str, dest: &Path) -> Result<u64, String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("upstream returned error status: {e}"))?;

    let total = response.content_length();
    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| format!("failed to create {}: {e}", dest.display()))?;

    let mut stream = response.bytes_stream();
    let mut written: u64 = 0;
    let mut next_progress = PROGRESS_INTERVAL;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read error mid-transfer: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write error: {e}"))?;
        written += chunk.len() as u64;
        if written >= next_progress {
            match total {
                Some(total) => info!(
                    "downloaded {} / {} MiB",
                    written / (1024 * 1024),
                    total / (1024 * 1024)
                ),
                None => info!("downloaded {} MiB", written / (1024 * 1024)),
            }
            next_progress += PROGRESS_INTERVAL;
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("flush error: {e}"))?;
    file.sync_all()
        .await
        .map_err(|e| format!("fsync error: {e}"))?;
    Ok(written)
}

/// Ensure verified weights exist at `config.model_path`.
async fn ensure_verified_weights(config: &Config) -> Result<(), String> {
    let expected = config.model_sha256;
    let path = &config.model_path;

    // Reuse an existing file only if its on-disk bytes match the expected
    // digest (a warm volume in dev; enclaves boot cold and skip this).
    if tokio::fs::try_exists(path).await.unwrap_or(false) {
        info!("found existing weights at {}, verifying...", path.display());
        match sha256_file(path).await {
            Ok(actual) if actual == expected => {
                info!("existing weights match expected SHA-256; skipping fetch");
                return Ok(());
            }
            Ok(actual) => {
                info!(
                    "existing weights do not match (expected {}, got {}); re-fetching",
                    hex::encode(expected),
                    hex::encode(actual)
                );
            }
            Err(e) => {
                info!("could not read existing weights ({e}); re-fetching");
            }
        }
    }

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }

    // reqwest with `rustls-no-provider` needs the process-global provider
    // installed (done in main); roots come bundled because the container is
    // FROM scratch with no system trust store.
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let client = reqwest::Client::builder()
        .use_preconfigured_tls(tls)
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    let partial = path.with_extension("partial");
    info!("fetching model weights from {}", config.model_url);
    let written = download(&client, &config.model_url, &partial).await?;
    info!(
        "download complete ({} MiB); verifying on-disk SHA-256...",
        written / (1024 * 1024)
    );

    // Verify the bytes as they exist on disk — these are the bytes the
    // engine will mmap.
    let actual = sha256_file(&partial)
        .await
        .map_err(|e| format!("failed to read back {}: {e}", partial.display()))?;
    if actual != expected {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(format!(
            "model weight hash mismatch:\n  expected: {}\n  actual:   {}\n\
             The fetched weights are not the weights this enclave was measured \
             to serve. Refusing to start.",
            hex::encode(expected),
            hex::encode(actual)
        ));
    }

    tokio::fs::rename(&partial, path)
        .await
        .map_err(|e| format!("failed to move weights into place: {e}"))?;
    info!(
        "model weights verified (sha256:{}) at {}",
        hex::encode(expected),
        path.display()
    );
    Ok(())
}

/// Replace `{model}` placeholders and exec the engine, never returning on
/// success.
fn exec_engine(config: &Config) -> String {
    let model = config.model_path.display().to_string();
    let argv: Vec<String> = config
        .engine_argv
        .iter()
        .map(|a| a.replace("{model}", &model))
        .collect();

    info!("exec: {}", argv.join(" "));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(&argv[0]).args(&argv[1..]).exec();
        format!("exec {} failed: {err}", argv[0])
    }
    #[cfg(not(unix))]
    {
        "eidola-inference only supports unix targets".to_string()
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = match Config::load() {
        Ok(config) => config,
        Err(e) => {
            error!("configuration error: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = ensure_verified_weights(&config).await {
        error!("weight verification failed: {e}");
        return ExitCode::FAILURE;
    }

    // exec only returns on failure.
    let err = exec_engine(&config);
    error!("{err}");
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sha256_accepts_valid_hex() {
        let hex_str = "a".repeat(64);
        let parsed = parse_sha256(&hex_str).unwrap();
        assert_eq!(parsed, [0xaa; 32]);
    }

    #[test]
    fn parse_sha256_rejects_bad_input() {
        assert!(parse_sha256("").is_err());
        assert!(parse_sha256("zz").is_err());
        assert!(parse_sha256(&"a".repeat(63)).is_err());
        assert!(parse_sha256(&"a".repeat(66)).is_err());
    }

    #[tokio::test]
    async fn mismatched_existing_weights_are_not_trusted() {
        // An existing file whose hash doesn't match must trigger a re-fetch;
        // with an unreachable URL that re-fetch fails, so the whole
        // verification errors out (fail closed) instead of serving the
        // wrong bytes.
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let dir =
            std::env::temp_dir().join(format!("eidola-inference-mismatch-{}", std::process::id()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let path = dir.join("model.gguf");
        tokio::fs::write(&path, b"not the weights you measured")
            .await
            .unwrap();

        let config = Config {
            model_url: "http://127.0.0.1:1/unreachable.gguf".to_string(),
            model_sha256: [0xab; 32],
            model_path: path.clone(),
            engine_argv: vec!["/bin/true".to_string()],
        };

        let err = ensure_verified_weights(&config).await.unwrap_err();
        assert!(err.contains("request failed"), "unexpected error: {err}");
        // The mismatched file must not have been blessed in place.
        let existing = tokio::fs::read(&path).await.unwrap();
        assert_eq!(existing, b"not the weights you measured");
        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[tokio::test]
    async fn sha256_file_matches_known_digest() {
        let dir =
            std::env::temp_dir().join(format!("eidola-inference-test-{}", std::process::id()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let path = dir.join("weights.bin");
        tokio::fs::write(&path, b"hello world").await.unwrap();

        let digest = sha256_file(&path).await.unwrap();
        assert_eq!(
            hex::encode(digest),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }
}
