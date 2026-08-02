//! Seeds make a *fresh* database usable; they never re-assert state over a
//! user's later edits.
//!
//! The seeded "Default" template owns one agent participant. Its seed used to
//! be an `INSERT OR IGNORE` on a well-known id, run on **every** database open
//! — which cannot tell "this row has never existed" from "the user removed it".
//! Since `update_template` replaces a template's owned agents wholesale (hard
//! delete + re-insert with fresh ids), *any* save to the Default template — a
//! removal, an addition, even a plain rename — left no row at the well-known
//! id, so the next launch injected the seeded agent back into the template and
//! from there into every new space (task 41).
//!
//! These tests model a restart the way `db_lock.rs` does: drop the first core
//! (releasing the advisory lock), then open a second one over the same dirs.

use eidola_app_core::{AppCore, NewTemplateParticipant, SpaceTemplateInfo};

/// Run a test body on its own thread — `AppCore` owns a tokio runtime and the
/// tests drive it with `block_on`.
fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

fn open(config_dir: &std::path::Path, data_dir: &std::path::Path) -> AppCore {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let client = reqwest::Client::builder().build().expect("client");
    AppCore::with_test_http_client(config_dir.to_path_buf(), data_dir.to_path_buf(), client)
        .expect("open core")
}

fn default_template(core: &AppCore) -> SpaceTemplateInfo {
    core.runtime()
        .block_on(core.list_space_templates())
        .expect("list templates")
        .into_iter()
        .find(|t| t.id == eidola_app_core::DEFAULT_TEMPLATE_ID)
        .expect("the Default template is seeded")
}

fn agent(label: &str, model: &str) -> NewTemplateParticipant {
    NewTemplateParticipant {
        label: label.into(),
        model_ref: Some(model.into()),
        system_prompt: None,
        notify_policy: "human".into(),
    }
}

fn labels(t: &SpaceTemplateInfo) -> Vec<String> {
    t.participants.iter().map(|p| p.label.clone()).collect()
}

/// A fresh database still gets a usable Default template — one agent on the
/// compiled default model. This is the seed's whole job.
#[test]
fn a_fresh_database_is_seeded_with_a_usable_default_template() {
    run(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let core = open(dir.path(), &dir.path().join("data"));

        let t = default_template(&core);
        assert_eq!(t.title, "Default");
        assert_eq!(t.participants.len(), 1, "one seeded agent");
        assert_eq!(
            t.participants[0].model_ref.as_deref(),
            Some(eidola_app_core::config::DEFAULT_MODEL)
        );
    });
}

/// Re-opening an untouched database re-runs the seeds and changes nothing.
#[test]
fn reopening_an_untouched_database_does_not_duplicate_the_seeded_agent() {
    run(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let (config_dir, data_dir) = (dir.path().to_path_buf(), dir.path().join("data"));

        let core = open(&config_dir, &data_dir);
        let before = labels(&default_template(&core));
        drop(core);

        let core = open(&config_dir, &data_dir);
        assert_eq!(labels(&default_template(&core)), before);
    });
}

/// The reported bug (task 41): the user removes the seeded agent from the
/// Default template, keeping another. It must stay removed across a restart.
#[test]
fn a_removed_default_agent_stays_removed_across_a_restart() {
    run(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let (config_dir, data_dir) = (dir.path().to_path_buf(), dir.path().join("data"));

        let core = open(&config_dir, &data_dir);
        let seeded = default_template(&core);
        assert_eq!(seeded.participants.len(), 1);
        let seeded_label = seeded.participants[0].label.clone();

        // Add a second agent alongside the seeded one…
        core.runtime()
            .block_on(core.update_template(
                seeded.id.clone(),
                None,
                None,
                Some(vec![
                    agent(&seeded_label, eidola_app_core::config::DEFAULT_MODEL),
                    agent("Gemma 4 E2B", "gemma4-e2b@local"),
                ]),
            ))
            .expect("add a second agent");

        // …then remove the seeded one, keeping the local model.
        core.runtime()
            .block_on(core.update_template(
                seeded.id.clone(),
                None,
                None,
                Some(vec![agent("Gemma 4 E2B", "gemma4-e2b@local")]),
            ))
            .expect("remove the seeded agent");
        assert_eq!(
            labels(&default_template(&core)),
            vec!["Gemma 4 E2B".to_string()],
            "the edit saves within the session"
        );
        drop(core);

        // Restart.
        let core = open(&config_dir, &data_dir);
        assert_eq!(
            labels(&default_template(&core)),
            vec!["Gemma 4 E2B".to_string()],
            "a seed must not re-inject an agent the user deliberately removed"
        );

        // …and a new space instantiated after the restart carries only what the
        // template says, which is where the user actually saw the resurrection.
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("create space");
        let members = core
            .runtime()
            .block_on(core.list_space_participants(space.id.clone()))
            .expect("list participants");
        let agents: Vec<String> = members
            .iter()
            .filter(|p| p.kind == "agent")
            .map(|p| p.label.clone())
            .collect();
        assert_eq!(agents, vec!["Gemma 4 E2B".to_string()]);
    });
}

/// The sharper half of the same mechanism: nothing was *removed* at all. A
/// plain rename replaces the owned-agent set (fresh ids), so a seed keyed on a
/// well-known id sees "absent" and injects a second agent on the next launch.
#[test]
fn a_renamed_default_agent_is_not_duplicated_by_a_restart() {
    run(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let (config_dir, data_dir) = (dir.path().to_path_buf(), dir.path().join("data"));

        let core = open(&config_dir, &data_dir);
        let seeded = default_template(&core);
        core.runtime()
            .block_on(core.update_template(
                seeded.id.clone(),
                None,
                None,
                Some(vec![agent(
                    "My assistant",
                    eidola_app_core::config::DEFAULT_MODEL,
                )]),
            ))
            .expect("rename the seeded agent");
        drop(core);

        let core = open(&config_dir, &data_dir);
        assert_eq!(
            labels(&default_template(&core)),
            vec!["My assistant".to_string()],
            "a rename must not leave the pre-rename agent behind on the next launch"
        );
    });
}
