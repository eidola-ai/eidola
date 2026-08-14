use std::fs;
use std::path::{Path, PathBuf};

use eidola_apple::{
    ApplyError, DetachError, GuardError, InspectError, PlacementRecord, apply, detach, inspect,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/fixtures/apple-roundtrip/synthetic-universal")
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    let mut entries: Vec<_> = fs::read_dir(source)
        .unwrap()
        .map(|entry| entry.unwrap())
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn assert_tree_equal(actual: &Path, expected: &Path) {
    let mut actual_entries: Vec<_> = fs::read_dir(actual)
        .unwrap()
        .map(|entry| entry.unwrap())
        .collect();
    let mut expected_entries: Vec<_> = fs::read_dir(expected)
        .unwrap()
        .map(|entry| entry.unwrap())
        .collect();
    actual_entries.sort_by_key(|entry| entry.file_name());
    expected_entries.sort_by_key(|entry| entry.file_name());
    assert_eq!(
        actual_entries
            .iter()
            .map(|entry| entry.file_name())
            .collect::<Vec<_>>(),
        expected_entries
            .iter()
            .map(|entry| entry.file_name())
            .collect::<Vec<_>>(),
        "directory entries differ at {}",
        actual.display()
    );
    for (actual_entry, expected_entry) in actual_entries.iter().zip(expected_entries.iter()) {
        let actual_type = actual_entry.file_type().unwrap();
        let expected_type = expected_entry.file_type().unwrap();
        assert_eq!(actual_type.is_dir(), expected_type.is_dir());
        if actual_type.is_dir() {
            assert_tree_equal(&actual_entry.path(), &expected_entry.path());
        } else {
            assert_eq!(
                fs::read(actual_entry.path()).unwrap(),
                fs::read(expected_entry.path()).unwrap(),
                "file bytes differ at {}",
                actual_entry.path().display()
            );
        }
    }
}

fn prepared_bundle() -> (tempfile::TempDir, PathBuf) {
    let temporary = tempfile::tempdir().unwrap();
    let bundle = temporary.path().join("Fixture.app");
    copy_tree(&fixtures().join("settled/Fixture.app"), &bundle);
    (temporary, bundle)
}

fn assert_detach_destination_unchanged(
    signed: &Path,
    unsigned: &Path,
    output: &Path,
    expected_reason: &str,
) {
    let snapshots = tempfile::tempdir().unwrap();
    let signed_before = snapshots.path().join("signed");
    let unsigned_before = snapshots.path().join("unsigned");
    let output_before = snapshots.path().join("output");
    copy_tree(signed, &signed_before);
    copy_tree(unsigned, &unsigned_before);
    copy_tree(output, &output_before);

    let error = detach(signed, unsigned, output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::InvalidDestination { ref reason, .. }
            if reason.contains(expected_reason)
    ));
    assert_tree_equal(signed, &signed_before);
    assert_tree_equal(unsigned, &unsigned_before);
    assert_tree_equal(output, &output_before);
}

fn repacked_12_14_input() -> Vec<u8> {
    let source = fs::read(fixtures().join("settled/Fixture.app/Contents/MacOS/Fixture")).unwrap();
    let x86_source = 16_384usize;
    let x86_size = 22_608usize;
    let arm_source = 49_152usize;
    let arm_size = 34_896usize;
    let x86_target = 4_096usize;
    let arm_target = 32_768usize;
    let mut output = vec![0; arm_target + arm_size];
    output[..48].copy_from_slice(&source[..48]);
    output[x86_target..x86_target + x86_size]
        .copy_from_slice(&source[x86_source..x86_source + x86_size]);
    output[arm_target..arm_target + arm_size]
        .copy_from_slice(&source[arm_source..arm_source + arm_size]);
    output[16..20].copy_from_slice(&(x86_target as u32).to_be_bytes());
    output[24..28].copy_from_slice(&12u32.to_be_bytes());
    output[36..40].copy_from_slice(&(arm_target as u32).to_be_bytes());
    output
}

#[test]
fn golden_universal_bundle_is_byte_exact() {
    let (_temporary, bundle) = prepared_bundle();
    apply(&bundle, &fixtures().join("detached")).unwrap();
    assert_tree_equal(&bundle, &fixtures().join("signed/Fixture.app"));
}

#[test]
fn input_wide_alignment_allows_x86_12_to_14_repacking() {
    let (_temporary, bundle) = prepared_bundle();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let repacked = repacked_12_14_input();
    fs::write(&executable, &repacked).unwrap();

    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    let hash = sha256(&repacked);
    record
        .machos
        .get_mut("Contents/MacOS/Fixture")
        .unwrap()
        .input_sha256 = hash.clone();
    record
        .inputs
        .insert("Contents/MacOS/Fixture".into(), format!("sha256:{hash}"));
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();

    apply(&bundle, detached_temp.path()).unwrap();
    assert_eq!(
        fs::read(executable).unwrap(),
        fs::read(fixtures().join("signed/Fixture.app/Contents/MacOS/Fixture")).unwrap()
    );
}

#[test]
fn wrong_input_is_typed_and_names_the_macho() {
    let (_temporary, bundle) = prepared_bundle();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let mut bytes = fs::read(&executable).unwrap();
    bytes[0x100] ^= 1;
    fs::write(&executable, &bytes).unwrap();

    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::WrongInput { ref path, .. }
            if path == Path::new("Contents/MacOS/Fixture")
    ));
    assert_eq!(fs::read(executable).unwrap(), bytes);
}

#[test]
fn modified_info_plist_is_refused_before_mutation() {
    let (_temporary, bundle) = prepared_bundle();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();
    fs::write(bundle.join("Contents/Info.plist"), b"modified").unwrap();

    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsignedInputHash { ref path, .. }
            if path == Path::new("Contents/Info.plist")
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn modified_resource_is_refused_before_mutation() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let resource = bundle.join("Contents/Resources/model.txt");
    fs::create_dir_all(resource.parent().unwrap()).unwrap();
    fs::write(&resource, b"original").unwrap();
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record.inputs.insert(
        "Contents/Resources/model.txt".into(),
        format!("sha256:{}", sha256(b"original")),
    );
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    fs::write(&resource, b"modified").unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsignedInputHash { ref path, .. }
            if path == Path::new("Contents/Resources/model.txt")
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn missing_and_unexpected_inputs_are_refused_by_relative_path() {
    let (_temporary, bundle) = prepared_bundle();
    fs::remove_file(bundle.join("Contents/Info.plist")).unwrap();
    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsignedInputMissing { ref path }
            if path == Path::new("Contents/Info.plist")
    ));

    let (_temporary, bundle) = prepared_bundle();
    let extra = bundle.join("Contents/MacOS/Unexpected");
    fs::write(&extra, b"executable").unwrap();
    let before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();
    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsignedInputUnexpected { ref path }
            if path == Path::new("Contents/MacOS/Unexpected")
    ));
    assert_eq!(
        fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap(),
        before
    );
}

#[test]
fn incompatible_plain_file_leaf_types_are_refused_before_macho_writes() {
    let assert_unchanged = |bundle: &Path, before: &[u8], error: ApplyError, expected: &Path| {
        assert!(matches!(
            error,
            ApplyError::PlainFileTarget { ref path, .. } if path == expected
        ));
        assert_eq!(
            fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap(),
            before
        );
    };

    let (_temporary, bundle) = prepared_bundle();
    let executable_before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();
    fs::create_dir_all(bundle.join("Contents/_CodeSignature/CodeResources")).unwrap();
    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert_unchanged(
        &bundle,
        &executable_before,
        error,
        Path::new("Contents/_CodeSignature/CodeResources"),
    );

    let (_temporary, bundle) = prepared_bundle();
    let executable_before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();
    fs::create_dir(bundle.join("Contents/CodeResources")).unwrap();
    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert_unchanged(
        &bundle,
        &executable_before,
        error,
        Path::new("Contents/CodeResources"),
    );

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    let ticket = b"ticket";
    record.files.insert(
        "Contents/CodeResources".into(),
        format!("sha256:{}", sha256(ticket)),
    );
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    fs::write(
        detached_temp
            .path()
            .join("Fixture.app/Contents/CodeResources"),
        ticket,
    )
    .unwrap();
    fs::create_dir(bundle.join("Contents/CodeResources")).unwrap();
    let executable_before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();
    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert_unchanged(
        &bundle,
        &executable_before,
        error,
        Path::new("Contents/CodeResources"),
    );

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record.files.clear();
    record.inputs.insert(
        "Contents/_CodeSignature".into(),
        format!("sha256:{}", sha256(b"bound stale seal")),
    );
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    fs::remove_file(
        detached_temp
            .path()
            .join("Fixture.app/Contents/_CodeSignature/CodeResources"),
    )
    .unwrap();
    fs::write(bundle.join("Contents/_CodeSignature"), b"bound stale seal").unwrap();
    let executable_before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();
    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert_unchanged(
        &bundle,
        &executable_before,
        error,
        Path::new("Contents/_CodeSignature"),
    );
}

#[cfg(unix)]
#[test]
fn readonly_recorded_plain_files_are_prepared_before_reconstruction() {
    use std::os::unix::fs::PermissionsExt;

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();

    let old_seal = b"old seal";
    let old_ticket = b"old ticket";
    let new_ticket = b"new ticket";
    record.inputs.insert(
        "Contents/_CodeSignature/CodeResources".into(),
        format!("sha256:{}", sha256(old_seal)),
    );
    record.inputs.insert(
        "Contents/CodeResources".into(),
        format!("sha256:{}", sha256(old_ticket)),
    );
    record.files.insert(
        "Contents/CodeResources".into(),
        format!("sha256:{}", sha256(new_ticket)),
    );
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();

    let detached_ticket = detached_temp
        .path()
        .join("Fixture.app/Contents/CodeResources");
    fs::write(&detached_ticket, new_ticket).unwrap();

    let seal = bundle.join("Contents/_CodeSignature/CodeResources");
    let ticket = bundle.join("Contents/CodeResources");
    fs::create_dir_all(seal.parent().unwrap()).unwrap();
    fs::write(&seal, old_seal).unwrap();
    fs::write(&ticket, old_ticket).unwrap();
    for path in [&seal, &ticket] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o444)).unwrap();
    }
    for path in [
        bundle.join("Contents"),
        bundle.join("Contents/_CodeSignature"),
    ] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o555)).unwrap();
    }

    let executable = bundle.join("Contents/MacOS/Fixture");
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o555)).unwrap();
    apply(&bundle, detached_temp.path()).unwrap();
    assert_eq!(
        fs::read(&executable).unwrap(),
        fs::read(fixtures().join("signed/Fixture.app/Contents/MacOS/Fixture")).unwrap()
    );
    assert_eq!(
        fs::read(&seal).unwrap(),
        fs::read(
            detached_temp
                .path()
                .join("Fixture.app/Contents/_CodeSignature/CodeResources")
        )
        .unwrap()
    );
    assert_eq!(fs::read(ticket).unwrap(), new_ticket);
}

#[cfg(unix)]
#[test]
fn readonly_signature_removal_targets_are_prepared_before_reconstruction() {
    use std::os::unix::fs::PermissionsExt;

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();

    let old_seal = b"old seal";
    let old_ticket = b"old ticket";
    record.files.clear();
    record.inputs.insert(
        "Contents/_CodeSignature/CodeResources".into(),
        format!("sha256:{}", sha256(old_seal)),
    );
    record.inputs.insert(
        "Contents/CodeResources".into(),
        format!("sha256:{}", sha256(old_ticket)),
    );
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    fs::remove_file(
        detached_temp
            .path()
            .join("Fixture.app/Contents/_CodeSignature/CodeResources"),
    )
    .unwrap();

    let seal = bundle.join("Contents/_CodeSignature/CodeResources");
    let ticket = bundle.join("Contents/CodeResources");
    fs::create_dir_all(seal.parent().unwrap()).unwrap();
    fs::write(&seal, old_seal).unwrap();
    fs::write(&ticket, old_ticket).unwrap();
    for path in [&seal, &ticket] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o444)).unwrap();
    }
    for path in [
        bundle.join("Contents"),
        bundle.join("Contents/_CodeSignature"),
    ] {
        fs::set_permissions(path, fs::Permissions::from_mode(0o555)).unwrap();
    }

    apply(&bundle, detached_temp.path()).unwrap();
    assert!(!seal.exists());
    assert!(seal.parent().unwrap().is_dir());
    assert!(!ticket.exists());
}

#[cfg(unix)]
#[test]
fn readonly_creation_parent_is_prepared_before_reconstruction() {
    use std::os::unix::fs::PermissionsExt;

    let (_temporary, bundle) = prepared_bundle();
    let contents = bundle.join("Contents");
    fs::set_permissions(&contents, fs::Permissions::from_mode(0o555)).unwrap();

    apply(&bundle, &fixtures().join("detached")).unwrap();
    assert_eq!(
        fs::read(bundle.join("Contents/_CodeSignature/CodeResources")).unwrap(),
        fs::read(fixtures().join("signed/Fixture.app/Contents/_CodeSignature/CodeResources"))
            .unwrap()
    );
}

#[test]
fn enormous_recorded_output_is_typed_and_does_not_mutate_the_bundle() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record
        .machos
        .get_mut("Contents/MacOS/Fixture")
        .unwrap()
        .output_len = u64::MAX;
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Placement { ref path, ref arch, ref reason }
            if path == Path::new("Contents/MacOS/Fixture")
                && arch == "all"
                && reason == "recorded output length 18446744073709551615 does not equal reconstructed end 84480"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn addressable_oversized_output_is_refused_before_allocation_or_mutation() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record
        .machos
        .get_mut("Contents/MacOS/Fixture")
        .unwrap()
        .output_len = 16 * 1024 * 1024 * 1024;
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Placement { ref path, ref arch, ref reason }
            if path == Path::new("Contents/MacOS/Fixture")
                && arch == "all"
                && reason == "recorded output length 17179869184 does not equal reconstructed end 84480"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn addressable_shifted_fat_slice_is_refused_before_allocation_or_mutation() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    let entry = record.machos.get_mut("Contents/MacOS/Fixture").unwrap();
    let slice = entry
        .slices
        .iter_mut()
        .find(|slice| slice.arch == "arm64")
        .unwrap();
    let shifted_offset = 16 * 1024 * 1024 * 1024u64;
    let delta = shifted_offset - slice.header_offset;
    slice.header_offset = shifted_offset;
    slice.fat_offset = Some(shifted_offset);
    slice.code_signature.as_mut().unwrap().lc_offset += delta;
    let linkedit = slice.linkedit.as_mut().unwrap();
    linkedit.vmsize_field_offset += delta;
    linkedit.fileoff_field_offset += delta;
    linkedit.filesize_field_offset += delta;
    entry.output_len = shifted_offset + slice.fat_size.unwrap();
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Placement { ref path, ref arch, ref reason }
            if path == Path::new("Contents/MacOS/Fixture")
                && arch == "arm64"
                && reason == "fat slice starts at 17179869184, canonical packing requires 49152"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn oversized_fat_alignment_is_refused_before_allocation_or_mutation() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record
        .machos
        .get_mut("Contents/MacOS/Fixture")
        .unwrap()
        .slices[0]
        .fat_align = Some(u32::MAX);
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Placement { ref path, ref arch, ref reason }
            if path == Path::new("Contents/MacOS/Fixture")
                && arch == "x86_64"
                && reason == "fat alignment exponent 4294967295 exceeds input maximum 14"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn canonical_target_alignment_above_input_maximum_is_refused_before_allocation() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    let entry = record.machos.get_mut("Contents/MacOS/Fixture").unwrap();
    let x86_offset = 1u64 << 34;
    let x86 = &mut entry.slices[0];
    let x86_delta = x86_offset - x86.header_offset;
    x86.header_offset = x86_offset;
    x86.fat_offset = Some(x86_offset);
    x86.fat_align = Some(34);
    x86.code_signature.as_mut().unwrap().lc_offset += x86_delta;
    let linkedit = x86.linkedit.as_mut().unwrap();
    linkedit.vmsize_field_offset += x86_delta;
    linkedit.fileoff_field_offset += x86_delta;
    linkedit.filesize_field_offset += x86_delta;

    let arm_offset = x86_offset + 32_768;
    let arm = &mut entry.slices[1];
    let arm_delta = arm_offset - arm.header_offset;
    arm.header_offset = arm_offset;
    arm.fat_offset = Some(arm_offset);
    arm.code_signature.as_mut().unwrap().lc_offset += arm_delta;
    let linkedit = arm.linkedit.as_mut().unwrap();
    linkedit.vmsize_field_offset += arm_delta;
    linkedit.fileoff_field_offset += arm_delta;
    linkedit.filesize_field_offset += arm_delta;
    entry.output_len = arm_offset + arm.fat_size.unwrap();
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::Placement { ref path, ref arch, ref reason }
            if path == Path::new("Contents/MacOS/Fixture")
                && arch == "x86_64"
                && reason == "fat alignment exponent 34 exceeds input maximum 14"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[test]
fn mutated_superblob_is_typed_and_names_the_macho() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let signature = detached_temp
        .path()
        .join("Fixture.app/Contents/MacOS/Fixture.arm64sign");
    let mut signature_bytes = fs::read(&signature).unwrap();
    signature_bytes[128] ^= 1;
    fs::write(&signature, signature_bytes).unwrap();
    let before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::DetachedSignature { ref path, ref arch, .. }
            if path == Path::new("Contents/MacOS/Fixture") && arch == "arm64"
    ));
    assert_eq!(
        fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap(),
        before,
        "validation failure mutated the input"
    );
}

#[test]
fn unexpected_detached_files_are_refused_before_bundle_mutation() {
    for relative in [
        "Fixture.app/Contents/MacOS/Fixture.stalesign",
        "Fixture.app/unrelated.txt",
    ] {
        let (_temporary, bundle) = prepared_bundle();
        let detached_temp = tempfile::tempdir().unwrap();
        copy_tree(&fixtures().join("detached"), detached_temp.path());
        let extra = detached_temp.path().join(relative);
        fs::create_dir_all(extra.parent().unwrap()).unwrap();
        fs::write(&extra, b"stale").unwrap();
        let executable = bundle.join("Contents/MacOS/Fixture");
        let before = fs::read(&executable).unwrap();

        let error = apply(&bundle, detached_temp.path()).unwrap_err();
        assert!(matches!(
            error,
            ApplyError::DetachedInputUnexpected { ref path }
                if path == relative.strip_prefix("Fixture.app/").unwrap()
        ));
        assert_eq!(fs::read(executable).unwrap(), before);
    }
}

#[test]
fn unexpected_detached_root_file_is_refused_for_both_input_forms() {
    for pass_app_path in [false, true] {
        let (_temporary, bundle) = prepared_bundle();
        let detached_temp = tempfile::tempdir().unwrap();
        copy_tree(&fixtures().join("detached"), detached_temp.path());
        fs::write(detached_temp.path().join("stale.sign"), b"stale").unwrap();
        let executable = bundle.join("Contents/MacOS/Fixture");
        let before = fs::read(&executable).unwrap();
        let detached = if pass_app_path {
            detached_temp.path().join("Fixture.app")
        } else {
            detached_temp.path().to_path_buf()
        };

        let error = apply(&bundle, &detached).unwrap_err();
        assert!(matches!(
            error,
            ApplyError::DetachedInputUnexpected { ref path }
                if path == Path::new("stale.sign")
        ));
        assert_eq!(fs::read(executable).unwrap(), before);
    }
}

#[test]
fn unexpected_detached_root_directory_is_refused_before_bundle_mutation() {
    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    fs::create_dir(detached_temp.path().join("payload")).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::DetachedInputInvalid { ref path, ref reason }
            if path == Path::new("payload") && reason == "unexpected directory"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
}

#[cfg(unix)]
#[test]
fn unexpected_detached_root_symlink_is_refused_before_bundle_mutation() {
    use std::os::unix::fs::symlink;

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let external_temp = tempfile::tempdir().unwrap();
    let external = external_temp.path().join("sentinel");
    fs::write(&external, b"sentinel").unwrap();
    symlink(&external, detached_temp.path().join("payload-link")).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::DetachedInputInvalid { ref path, ref reason }
            if path == Path::new("payload-link") && reason == "symbolic link"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
    assert_eq!(fs::read(external).unwrap(), b"sentinel");
}

#[cfg(unix)]
#[test]
fn unexpected_detached_symlink_is_refused_before_bundle_mutation() {
    use std::os::unix::fs::symlink;

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let external_temp = tempfile::tempdir().unwrap();
    let external = external_temp.path().join("external");
    fs::write(&external, b"sentinel").unwrap();
    let link = detached_temp.path().join("Fixture.app/unexpected-link");
    symlink(&external, &link).unwrap();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let before = fs::read(&executable).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::DetachedInputInvalid { ref path, ref reason }
            if path == Path::new("unexpected-link") && reason == "symbolic link"
    ));
    assert_eq!(fs::read(executable).unwrap(), before);
    assert_eq!(fs::read(external).unwrap(), b"sentinel");
}

#[test]
fn unsigned_slice_is_refused_by_path_and_architecture() {
    let (_temporary, bundle) = prepared_bundle();
    let executable = bundle.join("Contents/MacOS/Fixture");
    let unsettled = fs::read(fixtures().join("unsettled.macho")).unwrap();
    fs::write(&executable, &unsettled).unwrap();

    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record
        .machos
        .get_mut("Contents/MacOS/Fixture")
        .unwrap()
        .input_sha256 = sha256(&unsettled);
    record.inputs.insert(
        "Contents/MacOS/Fixture".into(),
        format!("sha256:{}", sha256(&unsettled)),
    );
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsignedSlice { ref path, ref arch }
            if path == Path::new("Contents/MacOS/Fixture") && arch == "x86_64"
    ));
}

#[test]
fn x86_64_linkedit_target_is_the_recorded_codesign_value() {
    let record: PlacementRecord = serde_json::from_slice(
        &fs::read(fixtures().join("detached/eidola-placement.json")).unwrap(),
    )
    .unwrap();
    let executable = &record.machos["Contents/MacOS/Fixture"];
    let slice = executable
        .slices
        .iter()
        .find(|slice| slice.arch == "x86_64")
        .unwrap();
    let linkedit = slice.linkedit.as_ref().unwrap();

    assert_eq!(linkedit.filesize, 18_944);
    assert_eq!(linkedit.vmsize, 32_768);
    assert_eq!(linkedit.vmsize_field_offset, 16_752);
    assert_ne!(linkedit.vmsize, (linkedit.filesize + 4095) & !4095);
}

#[test]
fn detach_reproduces_the_committed_layout_and_record() {
    let output = tempfile::tempdir().unwrap();
    let signed = fixtures().join("signed/Fixture.app");
    let unsigned = fixtures().join("settled/Fixture.app");

    let material_root = detach(&signed, &unsigned, output.path()).unwrap();
    assert_eq!(material_root, output.path().join("Fixture.app"));
    assert_eq!(
        fs::read(material_root.join("Contents/MacOS/Fixture.x86_64sign")).unwrap(),
        fs::read(fixtures().join("detached/Fixture.app/Contents/MacOS/Fixture.x86_64sign"))
            .unwrap()
    );
    assert_eq!(
        fs::read(material_root.join("Contents/MacOS/Fixture.arm64sign")).unwrap(),
        fs::read(fixtures().join("detached/Fixture.app/Contents/MacOS/Fixture.arm64sign")).unwrap()
    );
    assert_eq!(
        fs::read(material_root.join("Contents/_CodeSignature/CodeResources")).unwrap(),
        fs::read(fixtures().join("detached/Fixture.app/Contents/_CodeSignature/CodeResources"))
            .unwrap()
    );
    let actual: PlacementRecord =
        serde_json::from_slice(&fs::read(output.path().join("eidola-placement.json")).unwrap())
            .unwrap();
    let expected: PlacementRecord = serde_json::from_slice(
        &fs::read(fixtures().join("detached/eidola-placement.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn detach_apply_preserves_auxiliary_code_signature_inputs_without_a_seal() {
    let temporary = tempfile::tempdir().unwrap();
    let signed = temporary.path().join("signed/Fixture.app");
    let unsigned = temporary.path().join("unsigned/Fixture.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    fs::remove_file(signed.join("Contents/_CodeSignature/CodeResources")).unwrap();
    for bundle in [&signed, &unsigned] {
        let auxiliary = bundle.join("Contents/_CodeSignature/requirements");
        fs::create_dir_all(auxiliary.parent().unwrap()).unwrap();
        fs::write(auxiliary, b"bound auxiliary input").unwrap();
    }

    let detached = temporary.path().join("detached");
    detach(&signed, &unsigned, &detached).unwrap();
    let record: PlacementRecord =
        serde_json::from_slice(&fs::read(detached.join("eidola-placement.json")).unwrap()).unwrap();
    assert!(
        !record
            .files
            .contains_key("Contents/_CodeSignature/CodeResources")
    );
    assert!(
        record
            .inputs
            .contains_key("Contents/_CodeSignature/requirements")
    );

    let reconstructed = temporary.path().join("reconstructed/Fixture.app");
    copy_tree(&unsigned, &reconstructed);
    apply(&reconstructed, &detached).unwrap();
    assert_tree_equal(&reconstructed, &signed);
}

#[test]
fn detach_refuses_changed_auxiliary_code_signature_input_without_output() {
    let temporary = tempfile::tempdir().unwrap();
    let signed = temporary.path().join("signed/Fixture.app");
    let unsigned = temporary.path().join("unsigned/Fixture.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    fs::remove_file(signed.join("Contents/_CodeSignature/CodeResources")).unwrap();
    for (bundle, contents) in [
        (&signed, b"changed auxiliary".as_slice()),
        (&unsigned, b"unsigned auxiliary".as_slice()),
    ] {
        let auxiliary = bundle.join("Contents/_CodeSignature/requirements");
        fs::create_dir_all(auxiliary.parent().unwrap()).unwrap();
        fs::write(auxiliary, contents).unwrap();
    }
    let signed_before = temporary.path().join("signed-before");
    let unsigned_before = temporary.path().join("unsigned-before");
    copy_tree(&signed, &signed_before);
    copy_tree(&unsigned, &unsigned_before);
    let detached = temporary.path().join("detached");

    let error = detach(&signed, &unsigned, &detached).unwrap_err();
    assert!(matches!(
        error,
        DetachError::IncompatibleInput { ref path, .. }
            if path == Path::new("Contents/_CodeSignature/requirements")
    ));
    assert_tree_equal(&signed, &signed_before);
    assert_tree_equal(&unsigned, &unsigned_before);
    assert!(!detached.exists());
}

#[test]
fn detach_refuses_unsigned_slice_by_path_and_architecture() {
    let (_temporary, signed) = prepared_bundle();
    fs::write(
        signed.join("Contents/MacOS/Fixture"),
        fs::read(fixtures().join("unsettled.macho")).unwrap(),
    )
    .unwrap();
    let output = tempfile::tempdir().unwrap();

    let error = detach(
        &signed,
        &fixtures().join("settled/Fixture.app"),
        output.path(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DetachError::UnsignedSlice { ref path, ref arch }
            if path == Path::new("Contents/MacOS/Fixture") && arch == "x86_64"
    ));
}

#[test]
fn detach_refuses_wrong_unsigned_macho_before_creating_output() {
    let temporary = tempfile::tempdir().unwrap();
    let unsigned = temporary.path().join("Fixture.app");
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    let executable = unsigned.join("Contents/MacOS/Fixture");
    let mut bytes = fs::read(&executable).unwrap();
    bytes[0x5000] ^= 1;
    fs::write(&executable, bytes).unwrap();
    let output = temporary.path().join("new-detached");

    let error = detach(&fixtures().join("signed/Fixture.app"), &unsigned, &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::IncompatibleInput { ref path, ref arch, .. }
            if path == Path::new("Contents/MacOS/Fixture") && arch == "all"
    ));
    assert!(!output.exists());
}

#[test]
fn detach_refuses_changed_signed_resource_before_creating_output() {
    let temporary = tempfile::tempdir().unwrap();
    let signed = temporary.path().join("Fixture.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    fs::write(
        signed.join("Contents/Info.plist"),
        b"changed signed identity",
    )
    .unwrap();
    let output = temporary.path().join("new-detached");

    let error = detach(&signed, &fixtures().join("settled/Fixture.app"), &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::IncompatibleInput { ref path, ref arch, .. }
            if path == Path::new("Contents/Info.plist") && arch == "all"
    ));
    assert!(!output.exists());
}

#[test]
fn detach_refuses_unsigned_source_slice_without_changing_destination() {
    let temporary = tempfile::tempdir().unwrap();
    let unsigned = temporary.path().join("Fixture.app");
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    fs::write(
        unsigned.join("Contents/MacOS/Fixture"),
        fs::read(fixtures().join("unsettled.macho")).unwrap(),
    )
    .unwrap();
    let output = temporary.path().join("detached");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("sentinel"), b"keep").unwrap();

    let error = detach(&fixtures().join("signed/Fixture.app"), &unsigned, &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::UnsignedSlice { ref path, ref arch }
            if path == Path::new("Contents/MacOS/Fixture") && arch == "x86_64"
    ));
    assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
fn detach_binds_complete_unsigned_tree_before_creating_output() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let unsigned = temporary.path().join("Fixture.app");
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    let info = unsigned.join("Contents/Info.plist");
    let external = temporary.path().join("external-info");
    fs::copy(&info, &external).unwrap();
    fs::remove_file(&info).unwrap();
    symlink(&external, &info).unwrap();
    let output = temporary.path().join("new-detached");

    let error = detach(&fixtures().join("signed/Fixture.app"), &unsigned, &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::UnsafePath(GuardError::Symlink(ref path)) if path == &info
    ));
    assert!(!output.exists());
    assert!(external.exists());
}

#[test]
fn detach_refuses_source_overlapping_destinations_without_mutation() {
    let temporary = tempfile::tempdir().unwrap();
    let signed_parent = temporary.path().join("signed");
    let unsigned_parent = temporary.path().join("unsigned");
    let signed = signed_parent.join("Fixture.app");
    let unsigned = unsigned_parent.join("Fixture.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    let signed_backup = temporary.path().join("signed-backup");
    let unsigned_backup = temporary.path().join("unsigned-backup");
    copy_tree(&signed, &signed_backup);
    copy_tree(&unsigned, &unsigned_backup);

    for output in [
        signed_parent.clone(),
        unsigned_parent.clone(),
        signed.join("Contents/detached-output"),
    ] {
        let error = detach(&signed, &unsigned, &output).unwrap_err();
        assert!(matches!(error, DetachError::InvalidDestination { .. }));
        assert_tree_equal(&signed, &signed_backup);
        assert_tree_equal(&unsigned, &unsigned_backup);
    }

    let unresolved_alias = signed_parent.join("missing/..");
    let error = detach(&signed, &unsigned, &unresolved_alias).unwrap_err();
    assert!(matches!(
        error,
        DetachError::InvalidDestination { ref reason, .. }
            if reason.contains("overlaps source")
    ));
    assert!(!signed_parent.join("missing").exists());
    assert_tree_equal(&signed, &signed_backup);
    assert_tree_equal(&unsigned, &unsigned_backup);
}

#[test]
fn detach_refuses_stale_cleanup_root_containing_source() {
    let temporary = tempfile::tempdir().unwrap();
    let output = temporary.path().join("out");
    let signed = output.join("Old.app/Nested/New.app");
    let unsigned = temporary.path().join("unsigned/New.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    fs::write(
        output.join("eidola-placement.json"),
        br#"{"bundle":"Old.app"}"#,
    )
    .unwrap();
    let backup = temporary.path().join("output-backup");
    copy_tree(&output, &backup);

    let error = detach(&signed, &unsigned, &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::InvalidDestination { ref reason, .. }
            if reason.contains("previous detached root")
    ));
    assert_tree_equal(&output, &backup);
}

#[test]
fn detach_refuses_inexact_existing_output_without_mutation() {
    let temporary = tempfile::tempdir().unwrap();
    let signed = temporary.path().join("signed/Fixture.app");
    let unsigned = temporary.path().join("unsigned/Fixture.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);

    let unrelated_file = temporary.path().join("unrelated-file");
    fs::create_dir(&unrelated_file).unwrap();
    fs::write(unrelated_file.join("keep.txt"), b"keep").unwrap();
    assert_detach_destination_unchanged(
        &signed,
        &unsigned,
        &unrelated_file,
        "unexpected detached output entry",
    );

    let unrelated_directory = temporary.path().join("unrelated-directory");
    fs::create_dir_all(unrelated_directory.join("keep")).unwrap();
    assert_detach_destination_unchanged(
        &signed,
        &unsigned,
        &unrelated_directory,
        "unexpected detached output entry",
    );

    let missing_record = temporary.path().join("missing-record");
    fs::create_dir_all(missing_record.join("Old.app/Contents")).unwrap();
    fs::write(missing_record.join("Old.app/Contents/keep"), b"keep").unwrap();
    assert_detach_destination_unchanged(
        &signed,
        &unsigned,
        &missing_record,
        "unexpected detached output entry",
    );

    let corrupt_record = temporary.path().join("corrupt-record");
    fs::create_dir_all(corrupt_record.join("Old.app/Contents")).unwrap();
    fs::write(corrupt_record.join("Old.app/Contents/keep"), b"keep").unwrap();
    fs::write(corrupt_record.join("eidola-placement.json"), b"not json").unwrap();
    assert_detach_destination_unchanged(
        &signed,
        &unsigned,
        &corrupt_record,
        "invalid previous placement record",
    );
}

#[cfg(unix)]
#[test]
fn detach_refuses_unrelated_output_symlink_without_mutation() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let signed = temporary.path().join("signed/Fixture.app");
    let unsigned = temporary.path().join("unsigned/Fixture.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    let output = temporary.path().join("output");
    fs::create_dir(&output).unwrap();
    let external = temporary.path().join("external");
    fs::write(&external, b"keep").unwrap();
    let link = output.join("keep-link");
    symlink(&external, &link).unwrap();
    let signed_before = fs::read(signed.join("Contents/MacOS/Fixture")).unwrap();
    let unsigned_before = fs::read(unsigned.join("Contents/MacOS/Fixture")).unwrap();

    let error = detach(&signed, &unsigned, &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::UnsafePath(GuardError::Symlink(ref path)) if path == &link
    ));
    assert_eq!(fs::read_link(&link).unwrap(), external);
    assert_eq!(fs::read(&external).unwrap(), b"keep");
    assert_eq!(
        fs::read(signed.join("Contents/MacOS/Fixture")).unwrap(),
        signed_before
    );
    assert_eq!(
        fs::read(unsigned.join("Contents/MacOS/Fixture")).unwrap(),
        unsigned_before
    );
    assert_eq!(fs::read_dir(output).unwrap().count(), 1);
}

#[test]
fn detach_replaces_one_complete_parseable_previous_output() {
    let output = tempfile::tempdir().unwrap();
    fs::create_dir_all(output.path().join("Old.app/Contents")).unwrap();
    fs::write(output.path().join("Old.app/Contents/stale"), b"stale").unwrap();
    fs::write(
        output.path().join("eidola-placement.json"),
        br#"{"bundle":"Old.app"}"#,
    )
    .unwrap();

    let material = detach(
        &fixtures().join("signed/Fixture.app"),
        &fixtures().join("settled/Fixture.app"),
        output.path(),
    )
    .unwrap();
    assert_eq!(material, output.path().join("Fixture.app"));
    let mut names: Vec<_> = fs::read_dir(output.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    names.sort();
    assert_eq!(names, ["Fixture.app", "eidola-placement.json"]);
}

#[cfg(unix)]
#[test]
fn detach_refuses_stale_cleanup_root_alias_containing_source() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let output = temporary.path().join("out");
    let physical_signed = output.join("Old.app/Nested/New.app");
    let unsigned = temporary.path().join("unsigned/New.app");
    copy_tree(&fixtures().join("signed/Fixture.app"), &physical_signed);
    copy_tree(&fixtures().join("settled/Fixture.app"), &unsigned);
    fs::write(
        output.join("eidola-placement.json"),
        br#"{"bundle":"Old.app"}"#,
    )
    .unwrap();
    let alias = temporary.path().join("signed-alias");
    symlink(output.join("Old.app"), &alias).unwrap();
    let signed = alias.join("Nested/New.app");
    let before = fs::read(physical_signed.join("Contents/MacOS/Fixture")).unwrap();

    let error = detach(&signed, &unsigned, &output).unwrap_err();
    assert!(matches!(
        error,
        DetachError::InvalidDestination { ref reason, .. }
            if reason.contains("previous detached root")
    ));
    assert_eq!(
        fs::read(physical_signed.join("Contents/MacOS/Fixture")).unwrap(),
        before
    );
    assert_eq!(
        fs::read(output.join("eidola-placement.json")).unwrap(),
        br#"{"bundle":"Old.app"}"#
    );
}

#[test]
fn inspect_reads_claims_without_platform_tools() {
    let facts = inspect(&fixtures().join("signed/Fixture.app")).unwrap();
    assert_eq!(facts.team_id, None);
    assert_eq!(facts.identifier, "ai.eidola.fixture");
    assert!(facts.hardened_runtime);
    assert!(matches!(
        facts.entitlements_sha256.as_deref(),
        Some(hash) if hash.len() == 64
    ));
    assert!(!facts.has_notarization_ticket);
}

#[test]
fn inspect_reports_ticket_presence_and_unsigned_slice() {
    let (_temporary, bundle) = prepared_bundle();
    fs::write(bundle.join("Contents/CodeResources"), b"ticket").unwrap();
    let facts = inspect(&bundle).unwrap();
    assert!(!facts.hardened_runtime);
    assert_eq!(facts.entitlements_sha256, None);
    assert!(facts.has_notarization_ticket);

    fs::write(
        bundle.join("Contents/MacOS/Fixture"),
        fs::read(fixtures().join("unsettled.macho")).unwrap(),
    )
    .unwrap();
    let error = inspect(&bundle).unwrap_err();
    assert!(matches!(
        error,
        InspectError::UnsignedSlice { ref path, ref arch }
            if path == Path::new("Contents/MacOS/Fixture") && arch == "x86_64"
    ));
}

#[cfg(unix)]
#[test]
fn bundle_symlink_cannot_redirect_an_in_place_write() {
    use std::os::unix::fs::symlink;

    let (_temporary, bundle) = prepared_bundle();
    let external = bundle.parent().unwrap().join("external-macho");
    let original = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();
    fs::write(&external, &original).unwrap();
    fs::remove_file(bundle.join("Contents/MacOS/Fixture")).unwrap();
    symlink(&external, bundle.join("Contents/MacOS/Fixture")).unwrap();

    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsafePath(GuardError::Symlink(ref path))
            if path == &bundle.join("Contents/MacOS/Fixture")
    ));
    assert_eq!(fs::read(external).unwrap(), original);
}

#[cfg(unix)]
#[test]
fn detached_symlinks_are_refused_before_external_material_is_read() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let (_temporary, bundle) = prepared_bundle();
    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let signature = detached_temp
        .path()
        .join("Fixture.app/Contents/MacOS/Fixture.arm64sign");
    let external_temp = tempfile::tempdir().unwrap();
    let external = external_temp.path().join("external-signature");
    let bytes = fs::read(&signature).unwrap();
    fs::write(&external, &bytes).unwrap();
    let mut permissions = fs::metadata(&external).unwrap().permissions();
    permissions.set_mode(0o000);
    fs::set_permissions(&external, permissions).unwrap();
    fs::remove_file(&signature).unwrap();
    symlink(&external, &signature).unwrap();
    let before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::DetachedInputInvalid { ref path, ref reason }
            if path == Path::new("Contents/MacOS/Fixture.arm64sign")
                && reason == "symbolic link"
    ));
    assert_eq!(
        fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap(),
        before
    );

    let mut permissions = fs::metadata(&external).unwrap().permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(&external, permissions).unwrap();
    assert_eq!(fs::read(external).unwrap(), bytes);
}

#[cfg(unix)]
#[test]
fn seal_symlink_cannot_redirect_a_plain_file_write() {
    use std::os::unix::fs::symlink;

    let (_temporary, bundle) = prepared_bundle();
    let external = bundle.parent().unwrap().join("external-seal");
    let sentinel = b"external sentinel";
    fs::write(&external, sentinel).unwrap();
    fs::create_dir_all(bundle.join("Contents/_CodeSignature")).unwrap();
    let seal = bundle.join("Contents/_CodeSignature/CodeResources");
    symlink(&external, &seal).unwrap();

    let error = apply(&bundle, &fixtures().join("detached")).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsafePath(GuardError::Symlink(ref path)) if path == &seal
    ));
    assert_eq!(fs::read(external).unwrap(), sentinel);
}

#[cfg(unix)]
#[test]
fn stale_signature_directory_symlink_is_refused_before_removal() {
    use std::os::unix::fs::symlink;

    let (_temporary, bundle) = prepared_bundle();
    let external = bundle
        .parent()
        .unwrap()
        .join("external-signature-directory");
    fs::create_dir(&external).unwrap();
    fs::write(external.join("sentinel"), b"keep").unwrap();
    symlink(&external, bundle.join("Contents/_CodeSignature")).unwrap();

    let detached_temp = tempfile::tempdir().unwrap();
    copy_tree(&fixtures().join("detached"), detached_temp.path());
    let record_path = detached_temp.path().join("eidola-placement.json");
    let mut record: PlacementRecord =
        serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
    record.files.clear();
    fs::write(&record_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    fs::remove_file(
        detached_temp
            .path()
            .join("Fixture.app/Contents/_CodeSignature/CodeResources"),
    )
    .unwrap();
    let before = fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap();

    let error = apply(&bundle, detached_temp.path()).unwrap_err();
    assert!(matches!(
        error,
        ApplyError::UnsafePath(GuardError::Symlink(ref path))
            if path == &bundle.join("Contents/_CodeSignature")
    ));
    assert_eq!(fs::read(external.join("sentinel")).unwrap(), b"keep");
    assert_eq!(
        fs::read(bundle.join("Contents/MacOS/Fixture")).unwrap(),
        before
    );
}

#[cfg(unix)]
#[test]
fn detach_and_inspect_refuse_bundle_symlinks() {
    use std::os::unix::fs::symlink;

    let (_temporary, signed) = prepared_bundle();
    let executable = signed.join("Contents/MacOS/Fixture");
    let external = signed.parent().unwrap().join("external-input");
    let bytes = fs::read(&executable).unwrap();
    fs::write(&external, &bytes).unwrap();
    fs::remove_file(&executable).unwrap();
    symlink(&external, &executable).unwrap();
    let output = tempfile::tempdir().unwrap();

    let detach_error = detach(
        &signed,
        &fixtures().join("settled/Fixture.app"),
        output.path(),
    )
    .unwrap_err();
    assert!(matches!(
        detach_error,
        DetachError::UnsafePath(GuardError::Symlink(ref path)) if path == &executable
    ));
    let inspect_error = inspect(&signed).unwrap_err();
    assert!(matches!(
        inspect_error,
        InspectError::UnsafePath(GuardError::Symlink(ref path)) if path == &executable
    ));
    assert_eq!(fs::read(external).unwrap(), bytes);
}

#[cfg(unix)]
#[test]
fn detach_refuses_a_stale_output_bundle_symlink() {
    use std::os::unix::fs::symlink;

    let output = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    fs::write(external.path().join("sentinel"), b"keep").unwrap();
    symlink(external.path(), output.path().join("Fixture.app")).unwrap();

    let error = detach(
        &fixtures().join("signed/Fixture.app"),
        &fixtures().join("settled/Fixture.app"),
        output.path(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DetachError::UnsafePath(GuardError::Symlink(ref path))
            if path == &output.path().join("Fixture.app")
    ));
    assert_eq!(fs::read(external.path().join("sentinel")).unwrap(), b"keep");
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
