# Model Weight Management

## Context

Model weights define application behavior as much as code does, but they're too
large (100MB–100GB) to compile into a binary. This creates a tension between two
project values:

1. **Reproducibility and transparency.** The project uses Nix for deterministic
   builds, pins the Rust toolchain, and produces distroless OCI images. Weights
   should have the same guarantees — identical bytes, verified integrity, auditable
   changes.

2. **Minimal runtime dependencies.** Libraries are statically linked into the
   binary. Weights can't be, but we want to minimize reliance on network services
   at runtime.

There's also a spectrum between "application dependency" and "user data." Base
model weights are application dependencies — they define the model's behavior and
must be exact. User-specific adaptation layers (LoRA, fine-tunes) are user data,
but they depend on the *exact* base weights they were trained against. Getting
this relationship wrong means silent behavioral drift.

### Landscape

We evaluated several approaches used in the ecosystem:

- **Ollama:** OCI-style content-addressed blobs (SHA256 per layer). Tags are
  mutable; reproducibility requires pinning by digest.
- **llamafile:** Entire model + inference engine bundled into a single
  executable. Maximum reproducibility but no deduplication or incremental updates.
- **HuggingFace Hub:** Git repos with LFS or Xet chunk-level dedup. Files are
  SHA256-addressed. Reproducible if you pin the commit revision, but the common
  pattern (`model_name` without revision) is mutable.
- **Nix + GGUF (qmx, 2026):** Model files as `fetchurl` derivations with
  pinned SHA256 hashes. Nix enforces integrity. Best-in-class reproducibility.
- **OCI artifacts:** Docker Model Runner, CNCF ModelPack, KitOps all package
  models as OCI artifacts with content-addressed layers.
- **Sigstore/OMS model signing:** Emerging standard for cryptographic model
  provenance. NVIDIA NGC adopted it in 2025.

## Decision

Treat model weights as **pinned dependencies**, analogous to how `Cargo.lock`
pins library versions by hash. Verify integrity at every boundary.

### 1. Pin models by hash in version control

A manifest in the repository pins every supported model by content hash:

```toml
# models.lock.toml
[models.qwen3-0_6b]
source = "Qwen/Qwen3-0.6B"
format = "safetensors"
files = [
  { name = "model.safetensors", sha256 = "abc123..." },
  { name = "config.json", sha256 = "def456..." },
  { name = "tokenizer.json", sha256 = "789abc..." },
]
```

Changes to model pins are visible in version control diffs, just like dependency
updates.

### 2. Nix as the build-time source of truth

Model weights are Nix fixed-output derivations with pinned hashes:

```nix
qwen3-0_6b = pkgs.fetchurl {
  url = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors";
  hash = "sha256-...";
};
```

Nix will not accept bytes that don't match the hash. Models are cached in the Nix
binary cache alongside other build artifacts, shared across developers and CI.

### 3. Verify on load at runtime

Regardless of how weights arrive on disk — Nix store, app bundle, HuggingFace
download, user-provided path — the inference crate verifies the SHA256 hash
before loading:

```rust
pub struct VerifiedModel {
    pub path: PathBuf,
    pub expected_sha256: [u8; 32],
}

impl VerifiedModel {
    pub fn load(self) -> Result<Model, ModelIntegrityError> {
        let actual = sha256_file(&self.path)?;
        if actual != self.expected_sha256 {
            return Err(ModelIntegrityError::HashMismatch { expected, actual });
        }
        // Hash matches — safe to load
    }
}
```

This makes the verification independent of the distribution channel. Any path
that delivers the correct bytes is valid.

### 4. Adaptation layers reference the base model hash

User-specific adaptation layers (LoRA weights, fine-tunes) record the SHA256
of the base model they were trained against:

```toml
# adapter metadata
base_model_sha256 = "abc123..."
adapter_sha256 = "xyz789..."
```

The runtime refuses to compose an adapter with a base model whose hash doesn't
match. This prevents silent behavioral drift when base models are updated.

### 5. Distribution is context-dependent

The hash is the model's identity. Any channel that delivers the right bytes is
valid:

| Context | Strategy |
|---|---|
| Development / CI | Nix store + binary cache |
| macOS app | First-launch download with hash verification; small models may be bundled |
| OCI server | Model as a separate OCI layer, pinned by digest |
| User-provided | Accepted, verified against manifest; unknown hashes warn but don't block |

### 6. SafeTensors as the weight format

SafeTensors prevents arbitrary code execution during deserialization (unlike
pickle-based PyTorch formats). It was security-audited by Trail of Bits with no
critical findings. Combined with hash verification, this gives both integrity and
safety at the load boundary.

## Consequences

**Benefits:**

- Model identity is defined by content hash, not by mutable names or tags.
- The same verification logic works regardless of distribution channel.
- Model changes are auditable in version control (hash changes in the manifest).
- Nix binary cache eliminates redundant downloads across the team and CI.
- Adaptation layers are guaranteed to match their base model.

**Trade-offs we accept:**

- First-launch download is required for the macOS app (weights are too large to
  bundle for most models).
- SHA256 verification on large files adds load-time latency (mitigated by
  caching the verification result).
- Updating a pinned model requires a manifest change and PR, adding friction.
  This is intentional — weight changes should be deliberate and reviewed.

## Future Considerations

- **Content-addressed distribution (IPFS or similar).** For models we publish or
  endorse, distribute through a content-addressed network rather than relying
  solely on commercial third parties like HuggingFace. HuggingFace remains
  valuable for development and discovery, but our distribution channel for
  end users should not depend on a single commercial platform. The hash-based
  identity model makes this a transport-layer change, not an architectural one.

- **Sigstore/OMS model signing.** Once we distribute our own models or curated
  model sets, adopt the OpenSSF Model Signing specification for cryptographic
  provenance. This adds "who published these weights" on top of "are these the
  right bytes."

- **Quantization provenance.** If we distribute quantized variants, the
  manifest should record the quantization tool version and source model hash,
  so users can verify the full chain from source weights to quantized artifact.

- **Behavioral verification.** Hash integrity proves the bytes are correct but
  not that the model behaves as expected. A suite of deterministic probe inputs
  with expected outputs could detect adversarial weight modifications that
  preserve the format but alter behavior. This is an open research area.
