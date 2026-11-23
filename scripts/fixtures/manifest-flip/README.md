# manifest-flip fixtures

The inputs `scripts/test-manifest-schema-flip.sh` feeds the next manifest schema's emit side, in a copy of `scripts/artifact-manifest.sh` with `MANIFEST_SCHEMA_VERSION` flipped. The composed result is `../manifest-determinism/good-schema-6-emitted.json`.

| File | What |
| --- | --- |
| `bundle/` | A small mobile app bundle (an `index.html` and two scripts, one in a subdirectory), recorded by `record-mobile` as the `eidola-mobile-bundle` row and container. |
| `unsigned.apk` | Stand-in bytes for the unsigned APK: the producer records its `sha256` and nothing else about it. |
| `rows.json` | Every other row schema 6 requires, with placeholder values, as one partial. |
| `server-enclave.json` | The enclave block, supplied instead of measured (measuring downloads the platform's image), so a release that moves the real enclave does not move the composed manifest. |
