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
| `bad-out-of-range-schema.json` | `schema_version: 1e100` — integral, positive, and unreadable as the u64 the client parses that field into. A double-based reader sees nothing wrong with it. |
| `bad-duplicate-members.json` | Two `artifacts` members in one file. Every JSON parser keeps one and drops the other, so the first — carrying a forbidden row — disappears before any check runs, and the document validated stops being the document signed. Caught by a duplicate-aware parse before anything else reads the file. |
| `bad-environment-case.yml` | `environment: Apple-Signing`. Environment names are case-insensitive to GitHub: same environment, same secrets, different string. |
| `bad-block-scalar-needs.yml` | `needs: >-` with the job name on the next line. |
| `bad-flow-sequence-needs.yml` | A flow sequence broken across lines, so the key's own line carries no name. |
| `bad-spaced-colon-needs.yml` | `needs : release-assets` — YAML allows whitespace before a colon, so this is the `needs` key and the dependency resolves, while a pattern anchored to `needs:` collects nothing. Job-level keys are now read the way job keys are, so the spelling stops mattering. |
| `bad-commented-job-key.yml` | A comment after the job key, which ends the key early for YAML but not for a pattern anchored to end-of-line. |
| `bad-quoted-needs.yml` | The dependency on the key-holding job written as a quoted scalar (`- "apple-envelope"`), with a job name that says nothing about signing. Names are unquoted before the graph is built. |
| `bad-quoted-job-name.yml` | The same graph with the quotes on the job key (`"apple-envelope":`), which an unquoted-only header pattern never registers as a job at all. |
| `bad-fractional-schema.json` | `schema_version: 2.5`. The client reads that field with `as_u64()`, so a non-integer is malformed *there*; the gate has to reject what the verifier would. |
| `bad-anchored-environment.yml` | The protected environment reached through a YAML alias (`environment: *signing_env`). GitHub has resolved aliases since September 2025; this scanner does not resolve YAML, so an alias in a parsed field is refused rather than guessed at. |
| `bad-aliased-needs.yml` | The same refusal in the other parsed position — the whole dependency list is an alias. |
| `bad-escape-in-environment-mapping.yml` | A composition: an escaped scalar inside the *mapping* form of `environment:`. Each construct is refused on its own; a checker with a rule per parse site misses their pairing. |
| `bad-anchored-flow-job.yml` | The other composition: an anchor in front of a flow-mapping job body, which matches neither a flow-body pattern nor a block-body one, so the job is never registered at all. |
| `bad-quote-escape-in-needs.yml` | Found by sweeping constructs against extraction paths: a plain job key holding an apostrophe, named from `needs:` in the quoted spelling where `''` is an escape. A tokenizer ending the scalar at the first closing quote splits the name in two and drops the edge. |
| `bad-escaped-environment.yml` | The environment name written with a YAML escape (`"apple\u002dsigning"`). YAML decodes it; this scanner decodes nothing, so escapes in a parsed field are refused rather than guessed at. |
| `bad-escaped-job-name.yml` | The single-quoted scalar's one escape (`''`), on a job key — the job is `apple'-envelope` to GitHub and something else to a reader that only strips outer quotes, so the dependency edge lands on a job the graph does not have. The key-holder produces no manifest partial, so no other rule covers it. |
| `bad-flow-mapping-job.yml` | The job body as a one-line flow mapping, which a block-form header pattern never registers — its `environment:` is never read, and its name says nothing about signing. |
| `bad-json-constant.json` | `"narHash": Infinity`. Python accepts it, JSON has no such literal, and the client's parser refuses the whole document — the checked bytes would be unreadable to every reader that matters. |
| `bad-computed-environment.yml` | The environment name assembled by an expression (`format('{0}-{1}', 'apple', 'signing')`). `environment:` accepts expressions, this check evaluates none, and the file contains no matchable string — so expressions are refused on jobs that can reach the manifest, with the workflow's own cachix-write expression pinned in the check as read-and-cleared. |
| `bad-reusable-workflow-ancestor.yml` | A job-level `uses:` in the manifest job's ancestry. The callee's environments and outputs are in a file this scanner does not read, so the call is refused rather than assumed harmless. |
| `bad-merge-key-job.yml` | A merge key splicing an `environment:` into the job, in the form that needs no anchor at all. GitHub rejects `<<:` today, but actions that expand it before GitHub parses exist. |
| `bad-mapping-environment.yml` | `environment:` in mapping form (`name: apple-signing`) on the job that computes a manifest partial — valid GitHub Actions that a scalar-only parser reads as "no environment". |
