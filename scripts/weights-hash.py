#!/usr/bin/env python3
"""Recompute a model's weights hash, and check its files, without Eidola's code.

The weights hash is specified in docs/model-identity.md. This script uses the
Python standard library only (3.11 or later, for tomllib).

    weights-hash.py hash DIR
        Print the weights hash of a weights directory: every *.safetensors
        file, config.json, tokenizer.json, chat_template.jinja,
        generation_config.json, and model.safetensors.index.json when
        present. This is what an Eidola engine computes when it starts.

    weights-hash.py verify CATALOG.toml DIR
        Check DIR against a catalog file (models/<family>/<label>.toml, or the
        same data from the gateway's /v1/models as JSON): every listed file is
        present with its size and SHA-256, no other covered file is present
        (the engine would hash it too), the manifest's aggregate is the
        catalog's sha256, and so is DIR's own weights hash.

    weights-hash.py download CATALOG.toml DIR
        Download every listed file from Hugging Face at the catalog's
        repository and revision into DIR (an empty or missing directory),
        then verify as above. Nothing else is fetched.

A catalog file may also be the JSON of one entry of the gateway's
/v1/models (the object with "sha256" and "weights"), saved to a file ending
in .json.

Exit status 0 when every check passes, 1 otherwise.
"""

import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request

REQUIRED = ["config.json", "tokenizer.json", "chat_template.jinja", "generation_config.json"]
OPTIONAL = ["model.safetensors.index.json"]
SHARD_SUFFIX = ".safetensors"
SAFE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CHUNK = 1 << 24


def covered(name):
    return (
        (name.endswith(SHARD_SUFFIX) and len(name) > len(SHARD_SUFFIX))
        or name in REQUIRED
        or name in OPTIONAL
    )


def manifest_text(files):
    """The canonical manifest: `<sha256>  <name>\\n` per file, sorted by name
    in byte order (exactly sha256sum's output)."""
    lines = sorted(files, key=lambda f: f[0].encode())
    return "".join(f"{digest}  {name}\n" for name, digest in lines)


def aggregate(files):
    return hashlib.sha256(manifest_text(files).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(directory):
    """The weights hash of a directory, as the engine computes it."""
    names = sorted(n for n in os.listdir(directory) if covered(n))
    for name in names:
        if not SAFE_NAME.match(name):
            raise SystemExit(f"{name!r}: the engine refuses a file name outside [A-Za-z0-9_][A-Za-z0-9._-]*")
    for name in REQUIRED:
        if name not in names:
            raise SystemExit(f"{directory}: {name} is missing")
    if not any(n.endswith(SHARD_SUFFIX) for n in names):
        raise SystemExit(f"{directory}: no *{SHARD_SUFFIX} shard")
    return aggregate([(n, file_sha256(os.path.join(directory, n))) for n in names])


def load_catalog(path):
    """(sha256, repo, revision, [(path, size, sha256)]) from a catalog file or
    a /v1/models entry."""
    with open(path, "rb") as f:
        raw = f.read()
    if path.endswith(".json"):
        entry = json.loads(raw)
        weights = entry["weights"]
        repo, revision, files = weights["repo"], weights["revision"], weights["files"]
    else:
        import tomllib

        entry = tomllib.loads(raw.decode())
        repo, revision = entry["source"]["repo"], entry["source"]["revision"]
        files = entry["files"]
    rows = [(f["path"], int(f["size"]), f["sha256"]) for f in files]
    return entry["sha256"], repo, revision, rows


def verify(catalog, directory):
    sha256, _, _, rows = load_catalog(catalog)
    ok = True

    def fail(message):
        nonlocal ok
        ok = False
        print(f"FAIL {message}")

    if not HEX64.match(sha256):
        fail(f"catalog sha256 {sha256!r} is not 64 lowercase hex digits")
    names = [name for name, _, _ in rows]
    if len(set(names)) != len(names):
        fail("a file is listed twice")
    for name in names:
        if not SAFE_NAME.match(name) or not covered(name):
            fail(f"{name!r} is not a file the weights hash covers")
    for name in REQUIRED:
        if name not in names:
            fail(f"{name} is not listed")

    listed = aggregate([(name, digest) for name, _, digest in rows])
    if listed == sha256:
        print(f"ok   the manifest of the listed files hashes to {sha256}")
    else:
        fail(f"the manifest of the listed files hashes to {listed}, not {sha256}")

    for name, size, digest in rows:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            fail(f"{name}: missing")
            continue
        actual_size = os.path.getsize(path)
        if actual_size != size:
            fail(f"{name}: {actual_size} bytes, the catalog says {size}")
            continue
        actual = file_sha256(path)
        if actual != digest:
            fail(f"{name}: sha256 {actual}, the catalog says {digest}")
        else:
            print(f"ok   {name}")

    extra = sorted(n for n in os.listdir(directory) if covered(n) and n not in names)
    for name in extra:
        fail(f"{name}: present but not listed; the engine would hash it")

    if ok:
        actual = directory_hash(directory)
        if actual == sha256:
            print(f"ok   {directory} has weights hash {sha256}")
        else:
            fail(f"{directory} has weights hash {actual}, not {sha256}")
    return ok


def download(catalog, directory):
    _, repo, revision, rows = load_catalog(catalog)
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo):
        raise SystemExit(f"repository {repo!r} is not <owner>/<name>")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"revision {revision!r} is not a 40-hex-digit commit")
    os.makedirs(directory, exist_ok=True)
    if os.listdir(directory):
        raise SystemExit(f"{directory} is not empty")
    for name, size, digest in rows:
        if not SAFE_NAME.match(name):
            raise SystemExit(f"{name!r} is not a safe file name")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"
        target = os.path.join(directory, name)
        print(f"get  {url}")
        h = hashlib.sha256()
        received = 0
        with urllib.request.urlopen(url) as response, open(target + ".part", "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                h.update(chunk)
                received += len(chunk)
                if received > size:
                    raise SystemExit(f"{name}: more than the catalog's {size} bytes")
        if received != size or h.hexdigest() != digest:
            raise SystemExit(f"{name}: {received} bytes with sha256 {h.hexdigest()}, the catalog says {size} and {digest}")
        os.replace(target + ".part", target)
    return verify(catalog, directory)


def main(argv):
    if len(argv) == 3 and argv[1] == "hash":
        print(directory_hash(argv[2]))
        return 0
    if len(argv) == 4 and argv[1] == "verify":
        return 0 if verify(argv[2], argv[3]) else 1
    if len(argv) == 4 and argv[1] == "download":
        return 0 if download(argv[2], argv[3]) else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
