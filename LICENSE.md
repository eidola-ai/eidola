# Licensing

Copyright © Eidola, Inc. and contributors.

This repository contains several components under different licenses. This file is the human-readable map; [`REUSE.toml`](REUSE.toml) declares the same information machine-readably per the [REUSE](https://reuse.software) specification, each Rust crate declares its license in its `Cargo.toml`, and the full license texts live in [`LICENSES/`](LICENSES/).

**Default rule: unless a file, directory, or crate manifest states otherwise, the contents of this repository are licensed under either the [MIT License](LICENSES/MIT.txt) or the [Apache License, Version 2.0](LICENSES/Apache-2.0.txt), at your option.**

## Rust crates

| Crate | License |
|---|---|
| `eidola-server-gateway`, `eidola-server-engine` | [AGPL-3.0-only](LICENSES/AGPL-3.0-only.txt) |
| `eidola-billing`, `eidola-billing-cli` | [AGPL-3.0-only](LICENSES/AGPL-3.0-only.txt) |
| `eidola-engine`, `eidola-engine-model`, `eidola-engine-chat`, `eidola-engine-kernels`, `eidola-engine-cpu`, `eidola-engine-cuda` | [AGPL-3.0-only](LICENSES/AGPL-3.0-only.txt) |
| `eidola-app-core` | [GPL-3.0-only with the app store permission](LICENSES/LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission.txt) |
| `eidola-cli` | [GPL-3.0-only](LICENSES/GPL-3.0-only.txt) |
| `eidola-gui` | [GPL-3.0-only](LICENSES/GPL-3.0-only.txt) |
| `eidola-notepad` | [GPL-3.0-only](LICENSES/GPL-3.0-only.txt) |
| `eidola-mobile` | [GPL-3.0-only with the app store permission](LICENSES/LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission.txt) |
| `eidola-theme`, `eidola-i18n` (shared by the GUI and the mobile app) | [GPL-3.0-only with the app store permission](LICENSES/LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission.txt) |
| `eidola-apple` | [MIT](LICENSES/MIT.txt) OR [Apache-2.0](LICENSES/Apache-2.0.txt) |
| all other crates | [MIT](LICENSES/MIT.txt) OR [Apache-2.0](LICENSES/Apache-2.0.txt) |

The split is deliberate. Eidola's privacy guarantees are properties of verifiable code, so the full applications are copyleft: a fork of the server or the apps must offer its users the same source access that Eidola's own trust chain depends on. The reusable pieces — the attestation verifier, enclave measurement, shared contract logic, the markdown editor widget, and the operational utilities — are permissively licensed so that auditors, researchers, and other projects can freely reuse and independently verify them.

## App stores

The mobile app is distributed through app stores, and GPLv3 alone does not allow that: a store's terms of use and usage rules (limits on devices and copies, no redistribution) and the measures it applies to what it delivers (Apple re-signs and encrypts it) are further restrictions on recipients, which section 10 forbids anyone conveying the program to impose, and they conflict with the conditions under which section 6 lets object code be conveyed.

So the GPL crates the mobile app bundle is built from — `eidola-app-core`, `eidola-mobile`, `eidola-theme` and `eidola-i18n` — are licensed under GPL-3.0-only **with an additional permission under section 7**, granted by Eidola, Inc. Its full text follows the GPL's in [`LICENSES/LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission.txt`](LICENSES/LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission.txt). It permits anyone — Eidola or a fork — to convey these crates, or a work based on them, in object code through an app store on that store's terms, provided that:

- the store's own restrictions are the only ones: the conveyor adds none of its own;
- the complete Corresponding Source stays available under the GPL with this permission, to anyone, at no charge, from a network server, for as long as the app is offered through the store;
- the app or its store listing tells its users, prominently, that it is free software under GPLv3 with this permission, and where its source is;
- anything also conveyed outside the store is conveyed under the GPL.

It takes no right away from anyone: a copy obtained any other way, the source included, is under the GPL as usual, and anyone may remove the permission from their copy, after which only the GPL applies to it. The source obligation is the reason for it: a fork shipped through a store owes its users the same source access as one shipped any other way.

The permission is declared in [`REUSE.toml`](REUSE.toml) as `LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission`, a single licence reference holding the GPL and the permission, because the REUSE specification admits no exception that is not on the SPDX list. Cargo's `license` field takes only SPDX list identifiers, so those crates' manifests keep `license = "GPL-3.0-only"` and name the reference under `[package.metadata.eidola] licence`. The mobile app shows its licence, with the permission's text, on its Licences screen, and says so on its About screen. `eidola-cli`, `eidola-gui` and `eidola-notepad` are not distributed through app stores and are GPL-3.0-only alone.

## Mobile shells

The native iOS and Android shells (`mobile/`), with their contract, test page and tools, are licensed under [MIT](LICENSES/MIT.txt) OR [Apache-2.0](LICENSES/Apache-2.0.txt). They host the app bundle, which carries the licenses of the crates it is built from: GPL-3.0-only with the app store permission for Eidola's own (above), and the third-party crates', fonts' and icons' own, which the app lists with their texts.

## Fonts

The Newsreader faces (`crates/eidola-theme/assets/fonts/`, and the website's `www/static/fonts/`) are © The Newsreader Project Authors, under the [SIL Open Font License 1.1](LICENSES/OFL-1.1.txt).

## Documentation and website

The documentation (`docs/`) and website content (`www/`) are licensed under
[CC-BY-4.0](LICENSES/CC-BY-4.0.txt).

**Exception:** the legal instruments of Eidola, Inc. — the terms of service (`www/pages/terms.md`), the privacy policy (`www/pages/privacy.md`), and the contributor license agreements (`CLA-INDIVIDUAL.md`, `CLA-CORPORATE.md`) — are published for transparency, not reuse. See
[`LICENSES/LicenseRef-Eidola-Legal.txt`](LICENSES/LicenseRef-Eidola-Legal.txt).

## Trademarks

The Eidola name and the Eidola logo are trademarks of Eidola, Inc. The licenses above do not grant any right to use them. Forks must not present themselves as Eidola or as endorsed by Eidola, Inc.

The logo is the hexagon-grid mark generated from [`brand/`](brand/); its derived assets are the macOS app icon (`crates/eidola-gui/Support/AppIcon.icns` and `Assets.car`), the website favicon and home-screen tile (`www/static/`), the Linux themed icons (`releases/linux/icons/`), and the mobile app icons (`mobile/ios/AppIcon.icon/`, `mobile/android/res/`). A fork should replace those files with its own identity.

## Other licensing

If these licenses do not fit your use case, contact <hello@eidola.ai>.

## Contributions

Contributions are accepted under a contributor license agreement — see [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are licensed outbound under the license of the component they touch, per the map above.
