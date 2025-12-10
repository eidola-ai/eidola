#!/bin/bash
set -euo pipefail

# Deterministic build using Docker
# This ensures builds are reproducible across different machines

IMAGE_NAME="eidolons-builder"
CONTAINER_NAME="eidolons-build-$$"
OUTPUT_DIR="./docker-output"
TARGET="x86_64-unknown-linux-musl"

echo "Building Docker image for linux/amd64 (x86_64)..."
echo "Note: On ARM64 hosts, this uses emulation for reproducible builds"
docker buildx build --platform linux/amd64 --load -t "${IMAGE_NAME}" .

echo "Creating container to extract artifacts..."
docker create --name "${CONTAINER_NAME}" "${IMAGE_NAME}"

echo "Extracting build artifacts for ${TARGET}..."
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
docker cp "${CONTAINER_NAME}:/build/target/${TARGET}/release/." "${OUTPUT_DIR}/"

echo "Cleaning up container..."
docker rm "${CONTAINER_NAME}"

echo ""
echo "Build complete! Artifacts are in ${OUTPUT_DIR}/"
echo ""
echo "Computing checksums..."
cd "${OUTPUT_DIR}"
find . -type f -exec sha256sum {} \; | sort -k 2
cd - > /dev/null

echo ""
echo "To verify determinism, run this script again and compare checksums."
echo "To get the digest of the rust base image for absolute pinning:"
echo "  docker pull rust:1.91.1 && docker images --digests rust:1.91.1"
