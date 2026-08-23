#!/bin/sh
# Pack a directory tree into the macOS shipping container, byte for byte.
#
# This is the recipe, and the only copy of it: `mkShippingZip` in flake.nix
# runs this same script, so what CI publishes and what a verifier produces
# cannot drift into two recipes that merely look alike.
#
# Why it exists as a script at all: the documented forward check —
# `apply(archive, envelope) = installable` — reconstructs a *tree*, and the
# file a browser downloaded is a zip. Comparing them means re-zipping, and
# the answer is only meaningful if the packing is a function of the tree.
# The flake attribute cannot serve that: it packs the Nix-built payload, on
# a Mac. A verifier has some other tree, often on Linux.
#
# The result is a function of the payload and nothing else:
#
#   * modes are normalized to exactly what the tar archive uses
#     (`u=rwX,go=rX`, keyed off the executable bit — the only mode a NAR
#     carries), because zip records Unix modes and `-X` does not drop them;
#     without this the hash would follow the caller's umask;
#   * every mtime is pinned to 1980-01-01 UTC (`-h`, so a symlink gets its
#     own rather than its target's), the earliest instant the zip format
#     can represent, with TZ fixed because zip writes DOS *local* time;
#   * entries are ordered by a `LC_ALL=C` sort rather than readdir order,
#     which is filesystem state;
#   * `-y` stores symlinks as symlinks. Without it Info-ZIP follows them
#     and silently writes copies — a bundle with a framework comes back
#     invalid, and nothing errors;
#   * `-X` drops the extra fields carrying uid/gid and second-resolution
#     timestamps.
#
# One thing it cannot normalize: a symlink's own mode, which no portable
# chmod sets and which differs by platform (0755 on macOS, 0777 on Linux).
# The payload contains none; if one ever appears, the tar archive remains
# the cross-platform identity.
#
# Portability contract, the same one `verify-apple.sh` states: POSIX shell,
# POSIX `find`/`chmod`/`touch`, and Info-ZIP `zip`. Runs on macOS or Linux.
#
# Usage:
#   scripts/pack-shipping-zip.sh <tree> <output.zip>
#
# <tree> is packed as its *contents*: a tree holding `Eidola.app` produces
# an archive whose entries begin `Eidola.app/`, which is the shape a macOS
# download has.

set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree> <output.zip>" >&2
  exit 2
fi

tree=$1
out=$2

if [ ! -d "$tree" ]; then
  echo "error: no such directory: $tree" >&2
  exit 2
fi

if ! command -v zip > /dev/null 2>&1; then
  echo "error: Info-ZIP zip is required" >&2
  exit 2
fi

LC_ALL=C
TZ=UTC
export LC_ALL TZ

# Written beside the tree and moved into place: zip writes its temporary
# archive in the *output's* directory, which may be read-only (the Nix
# store, when the derivation calls this).
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT INT TERM

chmod -R u=rwX,go=rX "$tree"
# `-t` rather than a GNU `-d @epoch`: BSD touch has no @epoch form, and
# both accept this stamp. 198001010000.00 is 1980-01-01T00:00:00Z under the
# TZ pinned above.
find "$tree" -exec touch -h -t 198001010000.00 {} +

(
  cd "$tree" || exit 1
  find . -mindepth 1 | LC_ALL=C sort | zip -q -X -y -@ "$work/shipping.zip"
)

mv "$work/shipping.zip" "$out"
