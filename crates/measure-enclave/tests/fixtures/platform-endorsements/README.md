# Platform endorsement attestation fixtures

`platform-v0.1.11/` is a verbatim copy of that release as vendored under `releases/trust/platform-endorsements/` (`platform-endorsements.json` and `attestation.sigstore.json`): the release this crate's tests hold known answers about, kept here so the vendored tree can rotate. The tests read it wherever a deployment names a vendored release.

The other files are genuine Sigstore bundles of GitHub artifact attestations, used by `src/platform_endorsements.rs`'s tests as real negatives against that release. Each is the `bundle` member of GitHub's `GET /repos/<owner>/<repo>/attestations/sha256:<digest>` response for the named subject, verifiable offline against `releases/trust/sigstore-trusted-root.json`.

| File | Signer | Subject |
| --- | --- | --- |
| `platform-v0.1.10.sigstore.json` | `tinfoilsh/cvmimage/.github/workflows/platform-release.yml@refs/tags/platform-v0.1.10` | `platform-endorsements.json`, sha256 `ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9548`: byte-identical to the file `platform-v0.1.11` carries, so it is an authentic attestation of that file under another tag |
| `confidential-model-router-v0.0.115.sigstore.json` | a release workflow of `tinfoilsh/confidential-model-router` at `refs/tags/v0.0.115` | that release's `tinfoil-deployment.json`: an authentic attestation by the same provider from another repository |
