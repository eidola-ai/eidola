# dcap-qvl TDX sample

`tdx_quote` and `tdx_quote_collateral.json` are verbatim copies of `sample/tdx_quote` and `sample/tdx_quote_collateral.json` from the `dcap-qvl` 0.6.6 crate (<https://github.com/Phala-Network/dcap-qvl>, MIT, © 2026 Phala Network).

- `tdx_quote` is a version 4 TDX quote from Intel TDX hardware (FMSPC `b0c06f000000`), with its PCK chain embedded (certification data type 5).
- `tdx_quote_collateral.json` is the Intel PCS collateral it verifies against (TCB evaluation data number 17), in `dcap-qvl`'s `QuoteCollateralV3` serialization. Intel's signed bytes are unmodified. It is valid from 2025-06-19 to 2025-07-19, so the tests verify at a fixed time inside that window.

The quote's launch extended RTMR0, so under the verifier's IGVM-model appraisal it authenticates and is then refused. That makes it the real-hardware negative case.
