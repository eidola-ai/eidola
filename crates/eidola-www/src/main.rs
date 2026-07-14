//! CLI for the website generator. Run from anywhere in the repo:
//!
//! ```text
//! eidola-www build [--out <dir>] [--drafts]   # default out: target/www
//! eidola-www serve [--addr <host:port>]       # dev server, drafts included
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use eidola_www::{BuildOptions, build, serve::serve};

/// Locate the repo root: walk up from CWD looking for `www/` + `docs/`.
fn find_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("www").is_dir() && dir.join("docs").is_dir() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut command = None;
    let mut out = None;
    let mut addr = "127.0.0.1:8000".to_string();
    let mut drafts = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "build" | "serve" if command.is_none() => command = Some(args[i].clone()),
            "--out" if i + 1 < args.len() => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--addr" if i + 1 < args.len() => {
                i += 1;
                addr = args[i].clone();
            }
            "--drafts" => drafts = true,
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!(
                    "usage: eidola-www build [--out <dir>] [--drafts] | serve [--addr <host:port>]"
                );
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let Some(root) = find_root() else {
        eprintln!("could not find the repo root (a directory containing www/ and docs/)");
        return ExitCode::FAILURE;
    };
    let out = out.unwrap_or_else(|| root.join("target/www"));
    let opts = BuildOptions {
        root,
        out,
        include_drafts: drafts,
    };

    let result = match command.as_deref() {
        Some("serve") => serve(
            BuildOptions {
                include_drafts: true,
                ..opts
            },
            &addr,
        ),
        _ => build(&opts).map(|stats| {
            println!(
                "built {} pages, {} posts, {} docs -> {}",
                stats.pages,
                stats.posts,
                stats.docs,
                opts.out.display()
            );
        }),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
