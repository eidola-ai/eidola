# manifest-determinism fixtures

Documents `scripts/check-manifest-determinism.sh` must **reject**. Every default run of the check re-runs them and fails if one is accepted, or if either class is missing — a check that has stopped catching anything is a worse outcome than a check that fails.

None of these is a real manifest or a real workflow. The `.yml` files deliberately live here rather than in `.github/workflows/`.

| Fixture | The hole it keeps closed |
| --- | --- |
| `bad-signed-block.json` | Envelope material in the manifest: a top-level `signing` block with a Team ID and a ticket hash, plus a per-row Apple signature hash. Caught by the key deny-list and the per-type field allow-list. |
| `bad-build-nonce.json` | A well-formed manifest carrying extra top-level fields (`build_nonce`, `run_id`) that name no signing material at all. Only the exact top-level key set catches this — determinism dies to any unvalidated field, not just to key-shaped ones. |
| `bad-empty-artifacts.json` | `artifacts: {}`. A manifest that records nothing would otherwise satisfy every per-entry rule vacuously. |
| `bad-signing-feeds-manifest.yml` | A signing job is a direct `needs:` of the manifest job, which also holds the signing environment. |
| `bad-transitive-signing.yml` | The same, one hop further away: the manifest job needs `apple`, and `apple` needs the signing job. Ancestors are walked, not just direct edges. |
| `bad-mapping-environment.yml` | `environment:` in mapping form (`name: apple-signing`) on the job that computes a manifest partial — valid GitHub Actions that a scalar-only parser reads as "no environment". |
