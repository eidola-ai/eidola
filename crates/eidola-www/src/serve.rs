//! A tiny std-only dev server: serves the built site and rebuilds when a
//! source file changed (checked per request via an mtime sweep — the site
//! is small enough that this is instant). Not for production; the deployed
//! site is plain static files on GitHub Pages.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

use crate::{BuildOptions, Error, build};

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "woff2" => "font/woff2",
        "xml" => "application/xml; charset=utf-8",
        "txt" => "text/plain; charset=utf-8",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

/// Newest mtime under the content sources; drives the rebuild check.
fn newest_mtime(root: &Path) -> SystemTime {
    fn walk(dir: &Path, newest: &mut SystemTime) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, newest);
            } else if let Ok(meta) = entry.metadata()
                && let Ok(mtime) = meta.modified()
                && mtime > *newest
            {
                *newest = mtime;
            }
        }
    }
    let mut newest = SystemTime::UNIX_EPOCH;
    walk(&root.join("www"), &mut newest);
    walk(&root.join("docs"), &mut newest);
    newest
}

fn handle(mut stream: TcpStream, out: &Path) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    let path = path.split(['?', '#']).next().unwrap_or("/");
    // Reject traversal; map directory routes to index.html.
    let mut file = out.to_path_buf();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                respond(&mut stream, 400, "text/plain", b"bad request");
                return;
            }
            _ => file.push(seg),
        }
    }
    if file.is_dir() {
        file.push("index.html");
    }
    match std::fs::read(&file) {
        Ok(body) => respond(&mut stream, 200, content_type(&file), &body),
        Err(_) => respond(&mut stream, 404, "text/plain", b"not found"),
    }
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

pub fn serve(opts: BuildOptions, addr: &str) -> Result<(), Error> {
    build(&opts)?;
    let listener = TcpListener::bind(addr)?;
    println!(
        "serving {} at http://{addr}/ (drafts included)",
        opts.out.display()
    );
    let last_built = Mutex::new(SystemTime::now());
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        // Rebuild if sources changed since the last build.
        {
            let mut last = last_built.lock().expect("serve mutex poisoned");
            if newest_mtime(&opts.root) > *last {
                match build(&opts) {
                    Ok(_) => println!("rebuilt"),
                    Err(e) => eprintln!("rebuild failed: {e}"),
                }
                *last = SystemTime::now();
            }
        }
        let out = opts.out.clone();
        std::thread::spawn(move || handle(stream, &out));
    }
    Ok(())
}
