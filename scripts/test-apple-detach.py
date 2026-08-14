#!/usr/bin/env python3
"""Regression for source-overlapping apple-detach destinations."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

SCRIPT = Path(__file__).with_name("apple-detach.py")
SPEC = importlib.util.spec_from_file_location("apple_detach", SCRIPT)
APPLE_DETACH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APPLE_DETACH)
PLACE_SCRIPT = Path(__file__).with_name("apple-place.py")
PLACE_SPEC = importlib.util.spec_from_file_location("apple_place", PLACE_SCRIPT)
APPLE_PLACE = importlib.util.module_from_spec(PLACE_SPEC)
PLACE_SPEC.loader.exec_module(APPLE_PLACE)
FIXTURE = Path(__file__).parent / "fixtures/apple-roundtrip/synthetic-universal"


def snapshot(root):
    entries = []
    for directory, dirs, files in os.walk(root):
        relative_dir = Path(directory).relative_to(root)
        plain_dirs = []
        for name in sorted(dirs):
            path = Path(directory, name)
            relative = str(relative_dir / name)
            if stat.S_ISLNK(path.lstat().st_mode):
                entries.append(("link", relative, os.readlink(path)))
            else:
                entries.append(("dir", relative))
                plain_dirs.append(name)
        dirs[:] = plain_dirs
        for name in sorted(files):
            path = Path(directory, name)
            relative = str(relative_dir / name)
            if stat.S_ISLNK(path.lstat().st_mode):
                entries.append(("link", relative, os.readlink(path)))
            else:
                entries.append(("file", relative, path.read_bytes()))
    return entries


def main():
    with tempfile.TemporaryDirectory(prefix="eidola-apple-detach-test.") as temporary:
        root = Path(temporary)
        signed_parent = root / "signed"
        unsigned_parent = root / "unsigned"
        signed = signed_parent / "Fixture.app"
        unsigned = unsigned_parent / "Fixture.app"
        for bundle, marker in ((signed, b"signed"), (unsigned, b"unsigned")):
            (bundle / "Contents" / "MacOS").mkdir(parents=True)
            (bundle / "Contents" / "sentinel").write_bytes(marker)
        signed_before = snapshot(signed)
        unsigned_before = snapshot(unsigned)

        def assert_destination_refused(destination, expected):
            destination_before = snapshot(destination)
            try:
                APPLE_DETACH.detach(str(signed), str(destination), str(unsigned))
            except SystemExit as error:
                if expected not in str(error):
                    raise AssertionError(
                        f"wrong refusal for {destination}: {error}"
                    ) from error
            else:
                raise AssertionError(f"accepted inexact output root: {destination}")
            assert snapshot(signed) == signed_before
            assert snapshot(unsigned) == unsigned_before
            assert snapshot(destination) == destination_before

        unrelated_file = root / "unrelated-file"
        unrelated_file.mkdir()
        (unrelated_file / "keep.txt").write_bytes(b"keep")
        assert_destination_refused(unrelated_file, "unexpected detached output entry")

        unrelated_directory = root / "unrelated-directory"
        (unrelated_directory / "keep").mkdir(parents=True)
        assert_destination_refused(unrelated_directory, "unexpected detached output entry")

        unrelated_symlink = root / "unrelated-symlink"
        unrelated_symlink.mkdir()
        symlink_target = root / "symlink-target"
        symlink_target.write_bytes(b"keep")
        (unrelated_symlink / "keep-link").symlink_to(symlink_target)
        assert_destination_refused(unrelated_symlink, "unexpected detached output entry")
        assert symlink_target.read_bytes() == b"keep"

        missing_record = root / "missing-record"
        (missing_record / "Old.app/Contents").mkdir(parents=True)
        (missing_record / "Old.app/Contents/keep").write_bytes(b"keep")
        assert_destination_refused(missing_record, "unexpected detached output entry")

        corrupt_record = root / "corrupt-record"
        (corrupt_record / "Old.app/Contents").mkdir(parents=True)
        (corrupt_record / "Old.app/Contents/keep").write_bytes(b"keep")
        (corrupt_record / "eidola-placement.json").write_text("not json")
        assert_destination_refused(corrupt_record, "invalid previous placement record")

        destinations = (
            signed_parent,
            unsigned_parent,
            signed / "Contents" / "detached-output",
            signed_parent / "missing" / "..",
        )
        for destination in destinations:
            try:
                APPLE_DETACH.detach(str(signed), str(destination), str(unsigned))
            except SystemExit as error:
                if "overlaps source" not in str(error):
                    raise AssertionError(f"wrong refusal for {destination}: {error}") from error
            else:
                raise AssertionError(f"accepted source-overlapping output: {destination}")
            assert snapshot(signed) == signed_before, f"signed source changed: {destination}"
            assert snapshot(unsigned) == unsigned_before, f"unsigned source changed: {destination}"
            assert not (signed_parent / "missing").exists()

        stale_output = root / "stale-output"
        nested_signed = stale_output / "Old.app/Nested/New.app"
        nested_unsigned = root / "nested-unsigned/New.app"
        for bundle, marker in (
            (nested_signed, b"nested-signed"),
            (nested_unsigned, b"nested-unsigned"),
        ):
            (bundle / "Contents/MacOS").mkdir(parents=True)
            (bundle / "Contents/sentinel").write_bytes(marker)
        record = stale_output / "eidola-placement.json"
        record.write_text(json.dumps({"bundle": "Old.app"}))
        stale_before = snapshot(stale_output)
        try:
            APPLE_DETACH.detach(
                str(nested_signed), str(stale_output), str(nested_unsigned)
            )
        except SystemExit as error:
            if "previous detached root overlaps source" not in str(error):
                raise AssertionError(f"wrong stale-root refusal: {error}") from error
        else:
            raise AssertionError("accepted stale cleanup root containing signed source")
        assert snapshot(stale_output) == stale_before

        reusable = root / "reusable"
        (reusable / "Old.app/Contents").mkdir(parents=True)
        (reusable / "Old.app/Contents/stale").write_bytes(b"stale")
        (reusable / "eidola-placement.json").write_text(
            json.dumps({"bundle": "Old.app"})
        )
        produced = Path(
            APPLE_DETACH.detach(
                str(FIXTURE / "signed/Fixture.app"),
                str(reusable),
                str(FIXTURE / "settled/Fixture.app"),
            )
        )
        assert produced == reusable / "Fixture.app"
        assert sorted(path.name for path in reusable.iterdir()) == [
            "Fixture.app",
            "eidola-placement.json",
        ]

        roundtrip_signed = root / "roundtrip/signed/Fixture.app"
        roundtrip_unsigned = root / "roundtrip/unsigned/Fixture.app"
        shutil.copytree(FIXTURE / "signed/Fixture.app", roundtrip_signed)
        shutil.copytree(FIXTURE / "settled/Fixture.app", roundtrip_unsigned)
        (roundtrip_signed / "Contents/_CodeSignature/CodeResources").unlink()
        for bundle in (roundtrip_signed, roundtrip_unsigned):
            auxiliary = bundle / "Contents/_CodeSignature/requirements"
            auxiliary.parent.mkdir(parents=True, exist_ok=True)
            auxiliary.write_bytes(b"bound auxiliary input")
        roundtrip_detached = root / "roundtrip/detached"
        APPLE_DETACH.detach(
            str(roundtrip_signed),
            str(roundtrip_detached),
            str(roundtrip_unsigned),
        )
        reconstructed = root / "roundtrip/reconstructed/Fixture.app"
        shutil.copytree(roundtrip_unsigned, reconstructed)
        APPLE_PLACE.place(str(reconstructed), str(roundtrip_detached))
        assert snapshot(reconstructed) == snapshot(roundtrip_signed)


if __name__ == "__main__":
    main()
