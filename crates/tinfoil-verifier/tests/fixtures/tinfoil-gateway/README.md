# Captured gateway attestation

A real attestation document from a test deployment of the gateway on Tinfoil Containers (AMD SEV-SNP), platform release `0.15.3-rc.1`, captured on 2026-11-20 (Unix time 1795192347).

- `document.json` is the response to `GET /.well-known/tinfoil-attestation?nonce=2ff46a4e19df1d202e305e91a1059de1605d3dae9e17ee8f88f89a40348cc270`, byte for byte. Its collateral carries the AMD VCEK and CRL and the release's reference values: the runtime manifest, the platform endorsements, and a freshness approval of each.
- `tls-leaf.pem` is the TLS certificate the deployment presented on the same host, whose SubjectPublicKeyInfo hash the document's `tls` key names.
- `tinfoil-config.yml` is the deployment's config as `deploy/gateway/tinfoil-config.yml` held it when the document was captured. Its SHA-256 is the report's HOST_DATA.
- `expected.json` is the record made at capture: the release and its manifest's SHA-256, the nonce, the capture time, and the launch identity the report carries (launch digest, HOST_DATA, guest policy), which was then the gateway's SEV-SNP pin in `releases/trust/server-enclave.json`.

The tests pin the document to `expected.json`, never to the tree's current release pin, and verify it at its capture time, inside the AMD CRL's validity window. So they keep passing after the CRL expires, after the gateway moves to another release or config, and without any deployment.
