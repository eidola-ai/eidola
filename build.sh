#!/bin/bash
export SOURCE_DATE_EPOCH=0  # or a fixed timestamp
export RUSTFLAGS="--remap-path-prefix=$(pwd)=/build"
cargo build --release --locked
