# Platform endorsement attestation fixtures

Genuine Sigstore bundles of GitHub artifact attestations, used by `src/platform_endorsements.rs`'s tests as real negatives against the vendored release in `releases/trust/platform-endorsements/`. Each is the `bundle` member of GitHub's `GET /repos/<owner>/<repo>/attestations/sha256:<digest>` response for the named subject, verifiable offline against `releases/trust/sigstore-trusted-root.json`.

| File | Signer | Subject |
| --- | --- | --- |
| `platform-v0.1.10.sigstore.json` | `tinfoilsh/cvmimage/.github/workflows/platform-release.yml@refs/tags/platform-v0.1.10` | `platform-endorsements.json`, sha256 `ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9548`: byte-identical to the file `platform-v0.1.11` carries, so it is an authentic attestation of the vendored file under another tag |
| `confidential-model-router-v0.0.115.sigstore.json` | a release workflow of `tinfoilsh/confidential-model-router` at `refs/tags/v0.0.115` | that release's `tinfoil-deployment.json`: an authentic attestation by the same provider from another repository |
