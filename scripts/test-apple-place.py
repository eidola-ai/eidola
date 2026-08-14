#!/usr/bin/env python3
"""Regression for stale detached material in the measured placement consumer."""

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

SCRIPT = Path(__file__).with_name("apple-place.py")
SPEC = importlib.util.spec_from_file_location("apple_place", SCRIPT)
APPLE_PLACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APPLE_PLACE)
FIXTURE = Path(__file__).parent / "fixtures/apple-roundtrip/synthetic-universal"


def assert_refused(bundle, detached, expected):
    executable = bundle / "Contents/MacOS/Fixture"
    before = executable.read_bytes()
    try:
        APPLE_PLACE.place(str(bundle), str(detached))
    except SystemExit as error:
        if expected not in str(error):
            raise AssertionError(f"wrong refusal: {error}") from error
    else:
        raise AssertionError(f"accepted invalid detached input: {detached}")
    assert executable.read_bytes() == before


def tree_snapshot(root):
    entries = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_dir():
            entries.append(("dir", relative))
        else:
            entries.append(("file", relative, path.read_bytes()))
    return entries


def repacked_12_14_input():
    source = bytearray(
        (FIXTURE / "settled/Fixture.app/Contents/MacOS/Fixture").read_bytes()
    )
    x86_source, x86_size, x86_target = 16_384, 22_608, 4_096
    arm_source, arm_size, arm_target = 49_152, 34_896, 32_768
    output = bytearray(arm_target + arm_size)
    output[:48] = source[:48]
    output[x86_target : x86_target + x86_size] = source[
        x86_source : x86_source + x86_size
    ]
    output[arm_target : arm_target + arm_size] = source[
        arm_source : arm_source + arm_size
    ]
    struct.pack_into(">I", output, 16, x86_target)
    struct.pack_into(">I", output, 24, 12)
    struct.pack_into(">I", output, 36, arm_target)
    return bytes(output)


def main():
    with tempfile.TemporaryDirectory(prefix="eidola-apple-place-test.") as temporary:
        root = Path(temporary)

        stale_bundle = root / "stale-ticket/Fixture.app"
        stale_detached = root / "stale-ticket/detached"
        shutil.copytree(FIXTURE / "settled/Fixture.app", stale_bundle)
        shutil.copytree(FIXTURE / "detached", stale_detached)
        stale_ticket = stale_bundle / "Contents/CodeResources"
        stale_ticket.write_bytes(b"stale ticket")
        stale_record_path = stale_detached / "eidola-placement.json"
        stale_record = json.loads(stale_record_path.read_text())
        stale_record["inputs"]["Contents/CodeResources"] = (
            "sha256:" + hashlib.sha256(stale_ticket.read_bytes()).hexdigest()
        )
        stale_record_path.write_text(json.dumps(stale_record, indent=2))
        APPLE_PLACE.place(str(stale_bundle), str(stale_detached))
        assert tree_snapshot(stale_bundle) == tree_snapshot(
            FIXTURE / "signed/Fixture.app"
        )

        bundle = root / "Fixture.app"
        detached = root / "detached"
        shutil.copytree(FIXTURE / "settled/Fixture.app", bundle)
        shutil.copytree(FIXTURE / "detached", detached)
        extra = detached / "Fixture.app/Contents/MacOS/Fixture.stalesign"
        extra.write_bytes(b"stale")
        assert_refused(bundle, detached, "Fixture.stalesign: unexpected detached material")
        extra.unlink()

        sibling = detached / "stale.sign"
        sibling.write_bytes(b"stale")
        assert_refused(bundle, detached / "Fixture.app", "stale.sign: unexpected detached material")
        sibling.unlink()

        payload = detached / "payload"
        payload.mkdir()
        assert_refused(bundle, detached, "payload: detached archive root contains an unexpected directory")
        payload.rmdir()

        external = root / "sentinel"
        external.write_bytes(b"sentinel")
        link = detached / "payload-link"
        link.symlink_to(external)
        assert_refused(bundle, detached, "payload-link: detached archive root contains a symbolic link")
        assert external.read_bytes() == b"sentinel"
        link.unlink()

        record_path = detached / "eidola-placement.json"
        record = json.loads(record_path.read_text())
        record["machos"]["Contents/MacOS/Fixture"]["output_len"] = 16 * 1024 * 1024 * 1024
        record_path.write_text(json.dumps(record, indent=2))
        assert_refused(
            bundle,
            detached,
            "recorded output length 17179869184 does not equal reconstructed end 84480",
        )

        record = json.loads(
            (FIXTURE / "detached/eidola-placement.json").read_text()
        )
        entry = record["machos"]["Contents/MacOS/Fixture"]
        arm64 = next(slice for slice in entry["slices"] if slice["arch"] == "arm64")
        shifted_offset = 16 * 1024 * 1024 * 1024
        delta = shifted_offset - arm64["header_offset"]
        arm64["header_offset"] = shifted_offset
        arm64["fat_offset"] = shifted_offset
        arm64["code_signature"]["lc_offset"] += delta
        arm64["linkedit"]["vmsize_field_offset"] += delta
        arm64["linkedit"]["fileoff_field_offset"] += delta
        arm64["linkedit"]["filesize_field_offset"] += delta
        entry["output_len"] = shifted_offset + arm64["fat_size"]
        record_path.write_text(json.dumps(record, indent=2))
        assert_refused(
            bundle,
            detached,
            "arm64: fat slice starts at 17179869184, canonical packing requires 49152",
        )

        record = json.loads(
            (FIXTURE / "detached/eidola-placement.json").read_text()
        )
        record["machos"]["Contents/MacOS/Fixture"]["slices"][0]["fat_align"] = 999
        record_path.write_text(json.dumps(record, indent=2))
        assert_refused(
            bundle,
            detached,
            "x86_64: fat alignment exponent 999 exceeds input maximum 14",
        )

        record = json.loads(
            (FIXTURE / "detached/eidola-placement.json").read_text()
        )
        entry = record["machos"]["Contents/MacOS/Fixture"]
        x86 = entry["slices"][0]
        x86_offset = 1 << 34
        x86_delta = x86_offset - x86["header_offset"]
        x86["header_offset"] = x86_offset
        x86["fat_offset"] = x86_offset
        x86["fat_align"] = 34
        x86["code_signature"]["lc_offset"] += x86_delta
        x86["linkedit"]["vmsize_field_offset"] += x86_delta
        x86["linkedit"]["fileoff_field_offset"] += x86_delta
        x86["linkedit"]["filesize_field_offset"] += x86_delta
        arm64 = entry["slices"][1]
        arm_offset = x86_offset + 32_768
        arm_delta = arm_offset - arm64["header_offset"]
        arm64["header_offset"] = arm_offset
        arm64["fat_offset"] = arm_offset
        arm64["code_signature"]["lc_offset"] += arm_delta
        arm64["linkedit"]["vmsize_field_offset"] += arm_delta
        arm64["linkedit"]["fileoff_field_offset"] += arm_delta
        arm64["linkedit"]["filesize_field_offset"] += arm_delta
        entry["output_len"] = arm_offset + arm64["fat_size"]
        record_path.write_text(json.dumps(record, indent=2))
        assert_refused(
            bundle,
            detached,
            "x86_64: fat alignment exponent 34 exceeds input maximum 14",
        )

        repacked = repacked_12_14_input()
        executable = bundle / "Contents/MacOS/Fixture"
        executable.write_bytes(repacked)
        record = json.loads(
            (FIXTURE / "detached/eidola-placement.json").read_text()
        )
        digest = hashlib.sha256(repacked).hexdigest()
        record["machos"]["Contents/MacOS/Fixture"]["input_sha256"] = digest
        record["inputs"]["Contents/MacOS/Fixture"] = f"sha256:{digest}"
        record_path.write_text(json.dumps(record, indent=2))
        APPLE_PLACE.place(str(bundle), str(detached))
        assert executable.read_bytes() == (
            FIXTURE / "signed/Fixture.app/Contents/MacOS/Fixture"
        ).read_bytes()


if __name__ == "__main__":
    main()
