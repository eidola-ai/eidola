#!/bin/sh
# Classify changed paths for the independently gated Rust, Apple, and markdown checks.
set -eu

rust=false
apple=false
markdown=false
while IFS= read -r path; do
  [ -n "$path" ] || continue
  case "$path" in
    *.md|.rumdl.toml) markdown=true ;;
  esac
  case "$path" in
    .github/workflows/rust-checks.yml|.github/scripts/rust-check-scope.sh|.github/scripts/test-rust-check-scope.sh)
      rust=true
      apple=true
      markdown=true ;;
    # Glob rather than an enumeration so a script added under `scripts/` with
    # `apple` in its path is gated by existence, not by remembering to list
    # it. `macho_facts.py` is the one Apple instrument whose name does not
    # carry the word.
    scripts/*apple*|scripts/macho_facts.py)
      apple=true ;;
    *.md|.rumdl.toml) : ;;
    .github/*|docs/*|www/*|scripts/*|*.sh|justfile) : ;;
    compose*.yaml|compose*.yml|Containerfile*|*.hcl|.dockerignore) : ;;
    flake.nix|flake.lock|*.swift-format) : ;;
    tinfoil-config.yml|artifact-manifest.json) : ;;
    .env.example|.env.release) : ;;
    .envrc|.gitignore|.gitattributes|.editorconfig) : ;;
    LICENSE*|SECURITY.md|GEMINI.md|CLA-SIGNERS.txt) : ;;
    *) rust=true ;;
  esac
done

printf 'rust=%s\napple=%s\nmarkdown=%s\n' "$rust" "$apple" "$markdown"
