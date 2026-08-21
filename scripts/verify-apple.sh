#!/bin/sh
# Reconstruct and inspect a signed app from the two published verification inputs.
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <unsigned-app.tar.gz> <signature-bundle.zip>" >&2
  exit 2
fi

for input in "$1" "$2"; do
  [ -f "$input" ] || {
    echo "verify-apple: not a readable file: $input" >&2
    exit 2
  }
done

for required_command in awk od; do
  command -v "$required_command" >/dev/null 2>&1 || {
    echo "verify-apple requires $required_command" >&2
    exit 2
  }
done

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "verify-apple requires $1 to read this archive" >&2
    exit 2
  }
}

verify_parent=${EIDOLA_VERIFY_TMPDIR:-${TMPDIR:-/tmp}}
verify_root=$(mktemp -d "$verify_parent/eidola-apple-verify.XXXXXX")
cleanup() {
  # Archive modes may make either extracted tree read-only. find does not
  # follow symlinks, so only directories inside this private root are changed.
  find "$verify_root" -type d -exec chmod u+w {} + 2>/dev/null || true
  rm -rf "$verify_root"
}
trap cleanup EXIT HUP INT TERM

# Container format from the leading magic bytes, never the file name. The
# canonical unsigned macOS archive is the flake's gzip'd POSIX tar
# (`.#eidola-gui-macos-universal-archive` — the bytes `archiveSha256`
# covers); the detached signature material has no manifest-bound container,
# so either form is read on either side.
detect_format() {
  case $(od -An -N2 -tx1 <"$1" | tr -d ' \n') in
  504b)
    archive_format=zip
    require_command unzip
    ;;
  1f8b)
    archive_format=tgz
    require_command tar
    ;;
  *)
    echo "$2 archive is neither a zip nor a gzip'd tar" >&2
    exit 1
    ;;
  esac
}

list_members() {
  archive=$1
  format=$2
  label=$3
  listing=$4
  case $format in
  zip)
    if ! unzip -Z1 "$archive" >"$listing"; then
      echo "cannot list $label archive" >&2
      exit 1
    fi
    ;;
  tgz)
    if ! tar -tzf "$archive" >"$listing.raw"; then
      echo "cannot list $label archive" >&2
      exit 1
    fi
    # The flake packs the payload directory as `.`, so the archive root
    # arrives as a bare `./` member with no extraction path of its own.
    # Only that exact name is dropped: any other name that normalizes to
    # nothing stays a rejection below.
    LC_ALL=C awk '$0 != "." && $0 != "./"' "$listing.raw" >"$listing"
    ;;
  esac
}

extract_archive() {
  case $2 in
  # -n never overwrites, which also keeps unzip out of its interactive
  # overwrite prompt. tar overwrites silently instead; both are equivalent
  # here because colliding members are refused before extraction.
  zip) unzip -nq "$1" -d "$3" ;;
  tgz) tar -xzf "$1" -C "$3" ;;
  esac
}

validate_archive_members() {
  archive=$1
  format=$2
  label=$3
  listing=$4
  probe_listing=$5
  probe_root=$6
  list_members "$archive" "$format" "$label" "$listing"
  : >"$probe_listing"
  LC_ALL=C awk -v label="$label" -v probe_listing="$probe_listing" '
    function extraction_path(name, count, components, position, result) {
      if (name ~ /^\//) {
        unsafe = 1
        return ""
      }
      count = split(name, components, "/")
      result = ""
      for (position = 1; position <= count; position++) {
        if (components[position] == "" || components[position] == ".") {
          continue
        }
        if (components[position] == "..") {
          unsafe = 1
          return ""
        }
        result = result (result == "" ? "" : "/") components[position]
      }
      return result
    }

    {
      unsafe = 0
      if ($0 ~ /[[:cntrl:]]/) {
        unsafe = 1
      }
      target = extraction_path($0)
      if (unsafe || target == "") {
        printf "%s archive contains an unsafe archive member: %s\n", label, $0 > "/dev/stderr"
        exit 1
      }
      if (target in seen) {
        printf "%s archive contains a duplicate or colliding archive member: %s\n", label, $0 > "/dev/stderr"
        exit 1
      }
      seen[target] = $0
      printf "%s\t%s\n", ($0 ~ /\/$/ ? "d" : "f"), target > probe_listing
    }
  ' "$listing" || exit 1

  mkdir "$probe_root"
  probe_marker=.eidola-name-${verify_root##*.}
  tab=$(printf '\t')
  while IFS="$tab" read -r member_type member_target; do
    probe_cursor=$probe_root
    probe_prefix=
    remainder=$member_target
    while [ "$remainder" != "${remainder#*/}" ]; do
      component=${remainder%%/*}
      remainder=${remainder#*/}
      probe_prefix=${probe_prefix:+$probe_prefix/}$component
      probe_component d "$component" "$probe_prefix"
    done
    probe_prefix=${probe_prefix:+$probe_prefix/}$remainder
    probe_component "$member_type" "$remainder" "$probe_prefix"
  done <"$probe_listing"
}

probe_component() {
  required_type=$1
  component=$2
  expected_name=$3
  path=$probe_cursor/$component
  marker=$path/$probe_marker
  if [ ! -e "$path" ]; then
    mkdir "$path" || {
      echo "$label archive contains a duplicate or colliding archive member: $expected_name" >&2
      exit 1
    }
    printf '%s\t%s\n' "$required_type" "$expected_name" >"$marker"
  else
    if [ ! -d "$path" ] || [ ! -f "$marker" ]; then
      echo "$label archive contains a duplicate or colliding archive member: $expected_name" >&2
      exit 1
    fi
    stored_type=
    stored_name=
    IFS="$tab" read -r stored_type stored_name <"$marker"
    if [ "$stored_name" != "$expected_name" ] ||
      [ "$stored_type" = f ] || [ "$required_type" = f ]; then
      echo "$label archive contains a duplicate or colliding archive member: $expected_name" >&2
      exit 1
    fi
  fi
  probe_cursor=$path
}

detect_format "$1" unsigned
unsigned_format=$archive_format
detect_format "$2" signature-bundle
detached_format=$archive_format

validate_archive_members \
  "$1" "$unsigned_format" unsigned "$verify_root/unsigned.members" \
  "$verify_root/unsigned.probe-members" "$verify_root/unsigned.probe"
validate_archive_members \
  "$2" "$detached_format" signature-bundle "$verify_root/detached.members" \
  "$verify_root/detached.probe-members" "$verify_root/detached.probe"
mkdir -p "$verify_root/unsigned" "$verify_root/detached"
extract_archive "$1" "$unsigned_format" "$verify_root/unsigned"
extract_archive "$2" "$detached_format" "$verify_root/detached"
# Nix archives can preserve read-only directory modes. Make only directories
# in the private extracted input owner-writable; find does not follow symlinks.
find "$verify_root/unsigned" -type d -exec chmod u+w {} +

top_level_count() {
  find "$1" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' '
}

direct_app() {
  root=$1
  label=$2
  expected_entries=$3
  if [ "$expected_entries" -ne 0 ] &&
    [ "$(top_level_count "$root")" -ne "$expected_entries" ]; then
    echo "$label archive root has an unexpected entry or wrapper" >&2
    exit 1
  fi

  app=$(find "$root" -mindepth 1 -maxdepth 1 -type d -name '*.app' -prune)
  if [ "$(printf '%s\n' "$app" | sed '/^$/d' | wc -l | tr -d ' ')" -ne 1 ]; then
    echo "$label archive root must contain exactly one direct .app bundle" >&2
    exit 1
  fi
  printf '%s\n' "$app"
}

unsigned_app=$(direct_app "$verify_root/unsigned" unsigned 1)
direct_app "$verify_root/detached" signature-bundle 0 >/dev/null
if [ ! -f "$verify_root/detached/eidola-placement.json" ] ||
  [ -L "$verify_root/detached/eidola-placement.json" ]; then
  echo "signature-bundle archive root must contain a regular eidola-placement.json" >&2
  exit 1
fi

cargo run -q -p release-tool -- apple apply "$unsigned_app" "$verify_root/detached"
echo "Structurally parsed embedded claims (not independent Apple trust):"
cargo run -q -p release-tool -- apple inspect "$unsigned_app"
