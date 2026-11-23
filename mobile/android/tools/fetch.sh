# Sourced by ../tools.sh (and its test, ../test-tools-cache.sh). Fetches a pinned archive into a
# cache and hands out its contents, under one rule: nothing runs unless its bytes were checked
# against the pinned SHA-256 in this run.
#
# The cache ($ANDROID_TOOLS/archives) holds downloaded archives only, and CI caches only that.
# Every run hashes the archive it is about to use, fetches it again if the cached copy does not
# match, and unpacks it afresh into $ANDROID_TOOLS/unpacked; an unpacked tree from an earlier run
# is never used, whatever it contains. Within one run (a script and the scripts it calls, which
# share EIDOLA_ANDROID_TOOLS_RUN, a random id) a tree is unpacked once.

# sha256 FILE... -> `<hex>  <file>` lines, with whichever tool the host has.
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

if [[ -z "${EIDOLA_ANDROID_TOOLS_RUN:-}" ]]; then
  EIDOLA_ANDROID_TOOLS_RUN="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
  export EIDOLA_ANDROID_TOOLS_RUN
fi

# fetch URL SHA256 [NAME] -> prints a directory holding the archive's contents, unpacked in this
# run from an archive that hashed to SHA256 in this run. With NAME the download is a single file,
# placed there under that name (a jar is not unpacked).
fetch() {
  local url="$1" sha="$2" name="${3:-}"
  local file="$ANDROID_TOOLS/archives/$sha/${name:-archive.zip}" dir="$ANDROID_TOOLS/unpacked/$sha"
  if [[ -f "$dir/.run" && "$(cat "$dir/.run")" == "$EIDOLA_ANDROID_TOOLS_RUN" ]]; then
    echo "$dir"
    return
  fi
  if [[ -f "$file" && "$(sha256 "$file" | cut -d' ' -f1)" != "$sha" ]]; then
    echo "$file: the cached archive does not hash to $sha; fetching it again" >&2
    rm -f "$file"
  fi
  if [[ ! -f "$file" ]]; then
    echo "fetching $url" >&2
    mkdir -p "$(dirname "$file")"
    curl -sSfL --retry 3 -o "$file.part" "$url"
    local got
    got="$(sha256 "$file.part" | cut -d' ' -f1)"
    if [[ "$got" != "$sha" ]]; then
      echo "$url: SHA-256 $got, expected $sha" >&2
      rm -f "$file.part"
      exit 1
    fi
    mv "$file.part" "$file"
  fi
  rm -rf "$dir" && mkdir -p "$dir"
  if [[ -z "$name" ]]; then
    unzip -q "$file" -d "$dir"
  else
    cp "$file" "$dir/$name"
  fi
  printf '%s' "$EIDOLA_ANDROID_TOOLS_RUN" >"$dir/.run"
  echo "$dir"
}
