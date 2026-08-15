//! Disposal of untouched spaces.
//!
//! A space is created when its window opens, so an abandoned new window (and
//! every launch that opens a blank one) leaves a durable empty conversation
//! behind. `AppCore::discard_if_pristine` and the startup sweep reap those,
//! under one law: **reap only what is provably pristine; when unsure, keep.**
//! A wrongly-kept orphan costs a Library row; a wrongly-reaped space costs
//! someone's work, so every test here is written to fail in the direction that
//! keeps.
//!
//! Three things are pinned:
//!
//! 1. **Teeth, both ways** — an untouched space is deleted *with its whole
//!    footprint* (membership rows and owned participants, not just the row a
//!    listing reads), and every door that changes a space stops it being
//!    reaped.
//! 2. **The door sweep is one parameterized test** over the write surface, so a
//!    future door that forgets to mark its space fails the suite rather than
//!    quietly reaping someone's configuration.
//! 3. **The write surface is enumerated where the writes live** — a source scan
//!    over `db.rs` (`the_stamp_ledger_covers_every_space_write`) that fails on a
//!    statement against the space's three tables which no ledger entry
//!    accounts for.

use eidola_app_core::db::HUMAN_PARTICIPANT_ID;
use eidola_app_core::{
    AppCore, ExpectedScope, MembershipRole, NewParticipant, ParticipantOverride, ParticipantUpdate,
    changes::Change,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn make_core_in(dir: &std::path::Path) -> AppCore {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    AppCore::new(dir.to_path_buf(), dir.join("data")).expect("open core")
}

fn make_core() -> (AppCore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let core = make_core_in(dir.path());
    (core, dir)
}

fn run_in_thread<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
    }
    out
}

/// A fresh untouched space, exactly as ⌘N leaves one: the default template
/// instantiated under a client-minted id, with no title.
fn blank_space(core: &AppCore) -> String {
    let id = eidola_app_core::new_space_id();
    core.runtime()
        .block_on(core.create_space_with_id(id.clone(), None))
        .expect("create space");
    id
}

/// The space's own agent — the copy the default template's instantiation made.
fn owned_agent(core: &AppCore, space_id: &str) -> String {
    let members = core
        .runtime()
        .block_on(core.list_space_participants(space_id.to_string()))
        .expect("roster");
    members
        .into_iter()
        .find(|p| p.source == "owned")
        .expect("the default template owns an agent")
        .id
}

// ---------------------------------------------------------------------------
// Teeth: an untouched space goes, footprint and all
// ---------------------------------------------------------------------------

#[test]
fn an_untouched_space_is_discarded_with_its_whole_footprint() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let space = blank_space(&core);
        let mut rx = core.subscribe_changes();

        // It arrives with rows of both kinds: the human's membership, and the
        // template's agent copied into a space-owned participant.
        let (spaces, memberships, owned) = core
            .runtime()
            .block_on(core.test_space_footprint(space.clone()))
            .unwrap();
        assert_eq!(spaces, 1);
        assert!(memberships >= 1, "the shared human is referenced in");
        assert!(owned >= 1, "the default template's agent is copied in");

        assert!(
            core.runtime()
                .block_on(core.discard_if_pristine(space.clone()))
                .unwrap(),
            "an untouched space is discarded"
        );

        assert_eq!(
            core.runtime()
                .block_on(core.test_space_footprint(space.clone()))
                .unwrap(),
            (0, 0, 0),
            "the row, its memberships and its own participants all go"
        );
        assert!(
            !core
                .runtime()
                .block_on(core.list_spaces(true))
                .unwrap()
                .iter()
                .any(|s| s.id == space),
            "and the Library shows no residue"
        );
        assert_eq!(
            drain(&mut rx),
            vec![Change::SpaceIndex],
            "a delete announces the listing it changed, and nothing else"
        );
    });
}

#[test]
fn the_shared_library_survives_a_disposal() {
    // The delete takes the space's *memberships*, never the globals they
    // reference: those are the shared library, and one of them is the human
    // every other space is built around.
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let keep = blank_space(&core);
        core.runtime()
            .block_on(core.post("a thought".into(), Some(keep.clone())))
            .unwrap();

        let doomed = blank_space(&core);
        assert!(
            core.runtime()
                .block_on(core.discard_if_pristine(doomed))
                .unwrap()
        );

        let roster = core
            .runtime()
            .block_on(core.list_space_participants(keep))
            .unwrap();
        assert!(
            roster.iter().any(|p| p.id == HUMAN_PARTICIPANT_ID),
            "the shared human is still referenced by the space that kept it"
        );
    });
}

#[test]
fn discarding_an_unknown_space_is_a_plain_no() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let mut rx = core.subscribe_changes();
        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine("no-such-space".into()))
                .unwrap()
        );
        assert!(drain(&mut rx).is_empty(), "nothing happened, nothing said");
    });
}

// ---------------------------------------------------------------------------
// The door sweep — the load-bearing test
// ---------------------------------------------------------------------------

/// Every door that changes a space, and the assertion is the same for all of
/// them: after this, the space is **not** reapable and survives an explicit
/// disposal.
///
/// This is the test the marking design rests on. A door added without marking
/// its space would reap someone's work silently in production; here it fails
/// loudly, on the row that names it. Adding a door means adding a row.
type Door = (&'static str, fn(&AppCore, &str));

const DOORS: &[Door] = &[
    ("post", |core, space| {
        core.runtime()
            .block_on(core.post("a saved thought".into(), Some(space.to_string())))
            .unwrap();
    }),
    ("edit_post", |core, space| {
        let posted = core
            .runtime()
            .block_on(core.post("first".into(), Some(space.to_string())))
            .unwrap();
        core.runtime()
            .block_on(core.edit_post(posted.action_id, "second".into()))
            .unwrap();
    }),
    ("rename_space", |core, space| {
        core.runtime()
            .block_on(core.rename_space(space.to_string(), "Tides".into()))
            .unwrap();
    }),
    ("archive_space", |core, space| {
        core.runtime()
            .block_on(core.archive_space(space.to_string()))
            .unwrap();
    }),
    ("set_space_cascade_limit", |core, space| {
        core.runtime()
            .block_on(core.set_space_cascade_limit(space.to_string(), 2))
            .unwrap();
    }),
    ("set_space_router_model", |core, space| {
        core.runtime()
            .block_on(core.set_space_router_model(space.to_string(), None))
            .unwrap();
    }),
    ("add_space_participant", |core, space| {
        core.runtime()
            .block_on(core.add_space_participant(
                space.to_string(),
                NewParticipant {
                    label: "Ada".into(),
                    model_ref: Some("some-model".into()),
                    system_prompt: None,
                    notify_policy: "explicit".into(),
                },
            ))
            .unwrap();
    }),
    ("update_space_participant", |core, space| {
        let agent = owned_agent(core, space);
        core.runtime()
            .block_on(core.update_space_participant(
                agent,
                ParticipantUpdate {
                    label: Some("Renamed".into()),
                    ..Default::default()
                },
                ExpectedScope::SpaceOwned {
                    space_id: space.to_string(),
                },
            ))
            .unwrap();
    }),
    ("set_space_participant_override", |core, space| {
        core.runtime()
            .block_on(core.set_space_participant_override(
                space.to_string(),
                HUMAN_PARTICIPANT_ID.to_string(),
                ParticipantOverride {
                    label: Some(Some("Mike".into())),
                    ..Default::default()
                },
            ))
            .unwrap();
    }),
    ("remove_space_participant", |core, space| {
        let agent = owned_agent(core, space);
        core.runtime()
            .block_on(core.remove_space_participant(space.to_string(), agent))
            .unwrap();
    }),
    ("grant_space_membership", |core, space| {
        // Sharing an agent out of one space and granting it into another
        // changes both: the home space loses an owned participant, and the
        // destination gains a member.
        let donor = blank_space(core);
        let agent = owned_agent(core, &donor);
        core.runtime()
            .block_on(core.promote_participant(agent.clone(), None, None))
            .unwrap();
        core.runtime()
            .block_on(core.grant_space_membership(
                space.to_string(),
                agent,
                MembershipRole::Observer,
            ))
            .unwrap();
    }),
    ("add_global_participant", |core, space| {
        let donor = blank_space(core);
        let agent = owned_agent(core, &donor);
        core.runtime()
            .block_on(core.promote_participant(agent.clone(), None, None))
            .unwrap();
        core.runtime()
            .block_on(core.add_global_participant(space.to_string(), agent, None))
            .unwrap();
    }),
    ("promote_participant", |core, space| {
        let agent = owned_agent(core, space);
        core.runtime()
            .block_on(core.promote_participant(agent, None, None))
            .unwrap();
    }),
];

#[test]
fn every_door_that_changes_a_space_stops_it_being_reaped() {
    run_in_thread(|| {
        let (core, _dir) = make_core();
        for (name, door) in DOORS {
            let space = blank_space(&core);
            assert!(
                core.runtime()
                    .block_on(core.test_space_is_pristine(space.clone()))
                    .unwrap(),
                "{name}: a fresh space starts pristine (the premise of the case)"
            );

            door(&core, &space);

            assert!(
                !core
                    .runtime()
                    .block_on(core.test_space_is_pristine(space.clone()))
                    .unwrap(),
                "{name} changed the space, so it is no longer pristine"
            );
            assert!(
                !core
                    .runtime()
                    .block_on(core.discard_if_pristine(space.clone()))
                    .unwrap(),
                "{name} must keep its space out of the reaper"
            );
            let (spaces, ..) = core
                .runtime()
                .block_on(core.test_space_footprint(space.clone()))
                .unwrap();
            assert_eq!(spaces, 1, "{name}: the space is still there");
        }
    });
}

#[test]
fn a_space_with_participants_configured_and_no_posts_is_kept() {
    // The deliberately-decided ambiguous case, stated on its own so it cannot
    // be lost inside the sweep: configuring who is in a conversation is work,
    // and work is kept whether or not anything was ever said.
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let space = blank_space(&core);
        core.runtime()
            .block_on(core.add_space_participant(
                space.clone(),
                NewParticipant {
                    label: "Ada".into(),
                    model_ref: Some("some-model".into()),
                    system_prompt: Some("Be exact.".into()),
                    notify_policy: "human".into(),
                },
            ))
            .unwrap();

        assert!(
            core.runtime()
                .block_on(core.get_space_tree(space.clone()))
                .unwrap()
                .is_empty(),
            "no posts at all"
        );
        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine(space))
                .unwrap(),
            "and it is kept anyway"
        );
    });
}

#[test]
fn a_titled_creation_is_born_touched_and_an_untitled_one_is_not() {
    // A caller-supplied title is a human saying what the conversation is for,
    // which is the one thing an instantiation can carry that is not birth. The
    // GUI's ⌘N and its "New Space from Template" both pass none, so both are
    // reaped alike — the uniform rule for the two creation doors.
    run_in_thread(|| {
        let (core, _dir) = make_core();

        let named = core
            .runtime()
            .block_on(core.create_space(Some("Tides".into())))
            .unwrap()
            .id;
        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine(named))
                .unwrap(),
            "a space someone named is kept"
        );

        let from_template = core
            .runtime()
            .block_on(core.create_space_from_template(
                eidola_app_core::config::DEFAULT_TEMPLATE_ID.to_string(),
                None,
            ))
            .unwrap()
            .id;
        assert!(
            core.runtime()
                .block_on(core.discard_if_pristine(from_template))
                .unwrap(),
            "an untouched template instantiation is reaped like any other blank"
        );
    });
}

#[test]
fn a_notebook_is_never_reaped() {
    // A notebook exists only for the agent that owns it and is the residence of
    // that agent's core memory. It is minted empty by a promotion, which makes
    // it the most obviously "pristine" space in the database and the one whose
    // loss would cost the most.
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let home = blank_space(&core);
        let agent = owned_agent(&core, &home);
        let outcome = core
            .runtime()
            .block_on(core.promote_participant(agent, None, None))
            .unwrap();
        let notebook = outcome.notebook_space_id;

        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine(notebook.clone()))
                .unwrap(),
            "a notebook is not an ordinary conversation"
        );
        let (spaces, ..) = core
            .runtime()
            .block_on(core.test_space_footprint(notebook))
            .unwrap();
        assert_eq!(spaces, 1);
    });
}

#[test]
fn a_write_landing_before_the_disposal_keeps_its_space() {
    // The predicate is re-asked inside the deleting transaction, so a caller
    // holding a "pristine" reading from a moment ago cannot talk the reaper
    // into acting on it.
    run_in_thread(|| {
        let (core, _dir) = make_core();
        let space = blank_space(&core);

        // The reading the close trigger would have acted on.
        assert!(
            core.runtime()
                .block_on(core.test_space_is_pristine(space.clone()))
                .unwrap()
        );

        // …and the write that lands between it and the disposal.
        core.runtime()
            .block_on(core.post("wait — one more thing".into(), Some(space.clone())))
            .unwrap();

        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine(space.clone()))
                .unwrap(),
            "the transaction decides for itself, not from the caller's reading"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_tree(space))
                .unwrap()
                .len(),
            1,
            "and the post is still there"
        );
    });
}

// ---------------------------------------------------------------------------
// The startup sweep
// ---------------------------------------------------------------------------

#[test]
fn the_startup_sweep_takes_the_orphans_and_leaves_the_rest() {
    // A session that ended without closing its windows fires no close hook, so
    // its blanks are still here at the next open. The sweep runs inside the
    // database's first-open initializer, which is what makes it provably
    // earlier than any read of the index.
    run_in_thread(|| {
        let dir = tempfile::tempdir().unwrap();

        let (orphan, spoken_in, named) = {
            let core = make_core_in(dir.path());
            let orphan = blank_space(&core);

            let spoken_in = blank_space(&core);
            core.runtime()
                .block_on(core.post("hello".into(), Some(spoken_in.clone())))
                .unwrap();

            let named = blank_space(&core);
            core.runtime()
                .block_on(core.rename_space(named.clone(), "Tides".into()))
                .unwrap();

            (orphan, spoken_in, named)
        };

        // The next launch.
        let core = make_core_in(dir.path());
        let mut rx = core.subscribe_changes();
        let listed = core.runtime().block_on(core.list_spaces(true)).unwrap();

        assert!(
            !listed.iter().any(|s| s.id == orphan),
            "the untouched orphan is gone before anything reads the index"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.test_space_footprint(orphan))
                .unwrap(),
            (0, 0, 0),
            "footprint and all"
        );
        assert!(
            listed.iter().any(|s| s.id == spoken_in),
            "a space someone posted in survives"
        );
        assert!(
            listed.iter().any(|s| s.id == named),
            "so does one someone named"
        );
        assert!(
            drain(&mut rx).contains(&Change::SpaceIndex),
            "and the sweep announced the listing it changed"
        );
    });
}

#[test]
fn a_sweep_with_nothing_to_take_says_nothing() {
    run_in_thread(|| {
        let dir = tempfile::tempdir().unwrap();
        {
            let core = make_core_in(dir.path());
            let space = blank_space(&core);
            core.runtime()
                .block_on(core.post("hello".into(), Some(space)))
                .unwrap();
        }

        let core = make_core_in(dir.path());
        let mut rx = core.subscribe_changes();
        let listed = core.runtime().block_on(core.list_spaces(true)).unwrap();
        assert_eq!(listed.len(), 1);
        assert!(
            !drain(&mut rx).contains(&Change::SpaceIndex),
            "an empty sweep is not an invalidation"
        );
    });
}

// ---------------------------------------------------------------------------
// The write-surface ledger
// ---------------------------------------------------------------------------

/// Every function in `db.rs` whose statements write a space's configuration
/// footprint — its own row, its `space_participant` rows, or the `participant`
/// rows it owns — and what each does about the pristineness stamp.
///
/// This is the enumeration the marking design rests on, kept where the writes
/// live rather than in a caller's memory. The scan below fails on any statement
/// against those three tables inside a function this list does not name, so a
/// new write door is a test failure until someone has decided what it means for
/// pristineness.
const STAMP_LEDGER: &[(&str, &str)] = &[
    // -- the stamp itself --
    ("touch_space", "writes the stamp"),
    ("touch_space_of_participant", "writes the stamp"),
    // -- the disposal --
    (
        "discard_space_if_pristine_body",
        "deletes a space that never had one",
    ),
    // -- space row writes, stamp folded into the statement --
    ("set_space_cascade_limit", "stamps in the same statement"),
    ("set_space_router_model", "stamps in the same statement"),
    ("archive_space", "stamps in the same statement"),
    ("update_space_title", "stamps in the same statement"),
    (
        "retire_participant_tx_body",
        "archives the retired agent's notebook; stamps in the same statement",
    ),
    // -- births --
    (
        "insert_notebook_space",
        "born stamped: a promotion minted it",
    ),
    (
        "instantiate_template_body",
        "resets the stamp last, inside its own transaction: birth is not a \
         change, except for a caller-supplied title",
    ),
    (
        "insert_space",
        "raw helper with no production caller (asserted below)",
    ),
    // -- participant + membership writes --
    (
        "insert_participant",
        "stamps its owner space for a space-owned row",
    ),
    ("update_participant_config", "stamps before the write"),
    (
        "soft_remove_participant",
        "retires a global, which owns no space; the space-scoped removal is \
         remove_space_participant_tx_body",
    ),
    (
        "promote_participant_tx_body",
        "stamps the home space it takes an owned agent out of",
    ),
    (
        "insert_participant_ref",
        "table is interpolated; stamps when it is space_participant",
    ),
    ("ensure_space_participant", "stamps before the write"),
    (
        "update_space_participant_override",
        "stamps before the write",
    ),
    ("leave_space_participant", "stamps before the write"),
    (
        "remove_space_participant_tx_body",
        "stamps inside its own transaction, either branch",
    ),
    // -- writes that reach `participant` but never a space --
    (
        "delete_template_owned_participants",
        "template-owned rows only; templates are never reaped",
    ),
    (
        "ensure_default_participants",
        "seeds the globals (the shared human, Eidola); owns no space",
    ),
    (
        "ensure_default_template_tx_body",
        "seeds the Default template's own agent; owns no space",
    ),
];

#[test]
fn the_stamp_ledger_covers_every_space_write() {
    let source = include_str!("../src/db.rs");
    // The in-module tests write these tables freely; the ledger is about the
    // production write surface.
    let production = source
        .split_once("\n#[cfg(test)]\nmod tests {")
        .map(|(before, _)| before)
        .expect("db.rs ends in its test module");

    let is_write = |line: &str| {
        for verb in ["UPDATE ", "INTO ", "DELETE FROM "] {
            let mut rest = line;
            while let Some(at) = rest.find(verb) {
                let tail = &rest[at + verb.len()..];
                // A table named by an interpolated `{…}` cannot be decided
                // lexically, so it is declared instead: every one counts as a
                // write and the ledger says which tables it can reach.
                if tail.starts_with('{') {
                    return true;
                }
                let table: String = tail
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if matches!(
                    table.as_str(),
                    "space" | "space_participant" | "participant"
                ) {
                    return true;
                }
                rest = tail;
            }
        }
        false
    };

    let mut current_fn = "<file scope>";
    let mut found: std::collections::BTreeSet<&str> = Default::default();
    for line in production.lines() {
        let trimmed = line.trim_start();
        for head in ["pub async fn ", "async fn ", "pub fn ", "fn "] {
            if let Some(rest) = trimmed.strip_prefix(head) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                // Borrow from the source so the set can hold `&'static str`.
                let start = line.len() - rest.len();
                current_fn = &line[start..start + name.len()];
                break;
            }
        }
        if is_write(line) {
            found.insert(current_fn);
        }
    }

    let ledgered: std::collections::BTreeSet<&str> =
        STAMP_LEDGER.iter().map(|(name, _)| *name).collect();

    let unaccounted: Vec<_> = found.difference(&ledgered).collect();
    assert!(
        unaccounted.is_empty(),
        "these functions in db.rs write a space's configuration footprint and \
         are not in STAMP_LEDGER: {unaccounted:?}\n\nDecide what the write \
         means for pristineness (stamp it, or say why it is a birth), then add \
         a ledger row. A door that writes without stamping reaps someone's \
         work."
    );
    let stale: Vec<_> = ledgered.difference(&found).collect();
    assert!(
        stale.is_empty(),
        "STAMP_LEDGER names functions that no longer write those tables: {stale:?}"
    );
}

/// **A separate-statement stamp comes before every statement in its function.**
///
/// The ledger above pins *coverage* — that each write door marks its space at
/// all. This pins the *order*, which is the half that decides whether a mark is
/// worth anything: turso autocommits each statement, so a stamp written after
/// the row it describes leaves a window in which the row is durable and its
/// space still reads untouched — long enough for a disposal to take both, and
/// the stamp then updates nothing. Marking first can only over-mark, which
/// keeps a space nothing changed; the other order loses the change.
///
/// The rule is lexical because it can be: a function that calls `touch_space` /
/// `touch_space_of_participant` must do so before it runs any statement of its
/// own. Doors that fold the stamp into the statement, and the transactions that
/// hold several statements atomically, call neither and are not asked.
#[test]
fn a_stamp_is_written_before_the_statements_it_covers() {
    let source = include_str!("../src/db.rs");
    let production = source
        .split_once("\n#[cfg(test)]\nmod tests {")
        .map(|(before, _)| before)
        .expect("db.rs ends in its test module");
    let lines: Vec<&str> = production.lines().collect();

    // Line -> the function it belongs to.
    let mut owner: Vec<&str> = Vec::with_capacity(lines.len());
    let mut current = "<file scope>";
    for line in &lines {
        let trimmed = line.trim_start();
        for head in ["pub async fn ", "async fn ", "pub fn ", "fn "] {
            if let Some(rest) = trimmed.strip_prefix(head) {
                let name_len = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .count();
                let start = line.len() - rest.len();
                current = &line[start..start + name_len];
                break;
            }
        }
        owner.push(current);
    }

    let is_stamp_call =
        |l: &str| l.contains("touch_space(conn") || l.contains("touch_space_of_participant(conn");
    let is_statement =
        |l: &str| l.contains(".execute(") || l.contains(".query(") || l.contains(".prepare(");

    let mut offenders: Vec<String> = Vec::new();
    let stampers: std::collections::BTreeSet<&str> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| is_stamp_call(l))
        .map(|(i, _)| owner[i])
        .collect();

    for func in stampers {
        // `touch_space` is itself the stamp; its own statement is the mark.
        if func == "touch_space" || func == "touch_space_of_participant" {
            continue;
        }
        let first_stamp = lines
            .iter()
            .enumerate()
            .position(|(i, l)| owner[i] == func && is_stamp_call(l))
            .expect("the function was found by this predicate");
        for (i, line) in lines.iter().enumerate() {
            if owner[i] == func && is_statement(line) && i < first_stamp {
                offenders.push(format!("{func} (statement at db.rs:{})", i + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these functions run a statement before they stamp the space it \n         changes, so a disposal interleaving between the two takes the change \n         with the space: {offenders:?}"
    );
}

#[test]
fn the_raw_space_insert_has_no_production_caller() {
    // `insert_space` writes a `space` row with no stamp and no template behind
    // it, which would be a hole in the ledger the moment production used it —
    // every real creation goes through `instantiate_template`.
    let source = include_str!("../src/db.rs");
    let (production, tests) = source
        .split_once("\n#[cfg(test)]\nmod tests {")
        .expect("db.rs ends in its test module");
    assert!(
        tests.contains("insert_space(&conn"),
        "the helper is exercised by the in-module tests"
    );
    let calls = production.matches("insert_space(&conn").count()
        + production.matches("insert_space(conn").count();
    assert_eq!(
        calls, 0,
        "insert_space must stay test-only; a production caller needs a stamp \
         rule of its own (see STAMP_LEDGER)"
    );
    let other_crates = include_str!("../src/lib.rs");
    assert!(
        !other_crates.contains("insert_space("),
        "and app-core must not reach for it either"
    );
}
