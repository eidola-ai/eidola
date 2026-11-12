# Builder for reproducing tinfoilsh/cvmimage releases.
# Mirrors nix/install.sh (same Nix release, same checksum, same nix.conf),
# except single-user (no systemd in a container).
FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 builder && mkdir -p /etc/nix && printf '%s\n' \
      'sandbox = true' \
      'sandbox-fallback = false' \
      'restrict-eval = true' \
      'allowed-uris = https://github.com/NixOS/nixpkgs/archive/' \
      '# Deviation: Rosetta cannot load x86_64 seccomp BPF; Nix syscall filter off.' \
      'filter-syscalls = false' \
      > /etc/nix/nix.conf
COPY install-nix.sh /usr/local/bin/install-nix.sh
