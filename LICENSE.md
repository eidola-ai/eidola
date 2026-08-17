# Licensing

Copyright © Eidola, Inc. and contributors.

This repository contains several components under different licenses. This file is the human-readable map; [`REUSE.toml`](REUSE.toml) declares the same information machine-readably per the [REUSE](https://reuse.software) specification, each Rust crate declares its license in its `Cargo.toml`, and the full license texts live in [`LICENSES/`](LICENSES/).

**Default rule: unless a file, directory, or crate manifest states otherwise, the contents of this repository are licensed under either the [MIT License](LICENSES/MIT.txt) or the [Apache License, Version 2.0](LICENSES/Apache-2.0.txt), at your option.**

## Rust crates

| Crate | License |
|---|---|
| `eidola-server` | [AGPL-3.0-only](LICENSES/AGPL-3.0-only.txt) |
| `eidola-app-core` | [GPL-3.0-only](LICENSES/GPL-3.0-only.txt) |
| `eidola-cli` | [GPL-3.0-only](LICENSES/GPL-3.0-only.txt) |
| `eidola-gui` | [GPL-3.0-only](LICENSES/GPL-3.0-only.txt) |
| `eidola-apple` | [MIT](LICENSES/MIT.txt) OR [Apache-2.0](LICENSES/Apache-2.0.txt) |
| all other crates | [MIT](LICENSES/MIT.txt) OR [Apache-2.0](LICENSES/Apache-2.0.txt) |

The split is deliberate. Eidola's privacy guarantees are properties of verifiable code, so the full applications are copyleft: a fork of the server or the apps must offer its users the same source access that Eidola's own trust chain depends on. The reusable pieces — the attestation verifier, enclave measurement, shared contract logic, the markdown editor widget, and the operational utilities — are permissively licensed so that auditors, researchers, and other projects can freely reuse and independently verify them.

## Documentation and website

The documentation (`docs/`) and website content (`www/`) are licensed under
[CC-BY-4.0](LICENSES/CC-BY-4.0.txt).

**Exception:** the legal instruments of Eidola, Inc. — the terms of service (`www/pages/terms.md`), the privacy policy (`www/pages/privacy.md`), and the contributor license agreements (`CLA-INDIVIDUAL.md`, `CLA-CORPORATE.md`) — are published for transparency, not reuse. See
[`LICENSES/LicenseRef-Eidola-Legal.txt`](LICENSES/LicenseRef-Eidola-Legal.txt).

## Trademarks

The Eidola name and the Eidola logo are trademarks of Eidola, Inc. The licenses above do not grant any right to use them. Forks must not present themselves as Eidola or as endorsed by Eidola, Inc.

The logo is the hexagon-grid mark generated from [`brand/`](brand/); its derived assets are the macOS app icon (`crates/eidola-gui/Support/AppIcon.icns` and `Assets.car`), the website favicon and home-screen tile (`www/static/`), and the Linux themed icons (`releases/linux/icons/`). A fork should replace those files with its own identity.

## Other licensing

If these licenses do not fit your use case, contact <hello@eidola.ai>.

## Contributions

Contributions are accepted under a contributor license agreement — see [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are licensed outbound under the license of the component they touch, per the map above.
