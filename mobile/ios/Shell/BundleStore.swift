// The bundle store, its provisional activation, the digest, and serving the selected bundle
// (shell-contract.md → Origin and serving, Bundle store, Bundle digest).
import CryptoKit
import Foundation
import WebKit

enum Origin {
    static let scheme = "eidola"
    static let host = "app"
    static let root = URL(string: "eidola://app/")!
}

let embedded = "embedded"

/// What the shell keeps durably (`state.json`).
struct StoreState: Codable, Equatable {
    var active = embedded
    var confirmed = true
    var tried = false
    var previous = embedded
    var failed: String?
    /// Sealed versions and their digests.
    var sealed: [String: String] = [:]
}

/// The bundle one document is served from.
struct Serving {
    let version: String
    let root: URL
    var confirmed: Bool
    var digest: String?
}

final class BundleStore {
    /// Files and the store share this queue: one request at a time, in arrival order.
    let queue = DispatchQueue(label: "ai.eidola.shell.store")
    private let dir: URL
    private let versions: URL
    private let embeddedRoot = Bundle.main.resourceURL!.appendingPathComponent("bundle", isDirectory: true)
    private var state = StoreState()
    /// Read by the scheme handler on the main thread, so behind a lock rather than the queue.
    private let lock = NSLock()
    private var serving: Serving

    init(dir: URL) throws {
        self.dir = dir
        versions = dir.appendingPathComponent("versions", isDirectory: true)
        try makePrivateDirectory(versions)
        if let data = try? Data(contentsOf: dir.appendingPathComponent("state.json")),
            let s = try? JSONDecoder().decode(StoreState.self, from: data)
        {
            state = s
        }
        serving = Serving(version: embedded, root: embeddedRoot, confirmed: true)
    }

    private func root(_ version: String) -> URL {
        version == embedded ? embeddedRoot : versions.appendingPathComponent(version, isDirectory: true)
    }

    private func persist(_ s: StoreState) throws {
        let tmp = dir.appendingPathComponent("state.json.tmp").path
        let data = try JSONEncoder().encode(s)
        let fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        guard fd >= 0 else { throw ShellError.errno("open state") }
        defer { close(fd) }
        let n = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        guard n == data.count else { throw ShellError.errno("write state") }
        try Durable.file(fd, tmp)
        guard rename(tmp, dir.appendingPathComponent("state.json").path) == 0 else { throw ShellError.errno("rename state") }
        try Durable.directory(dir.path)
        state = s
    }

    // MARK: Selection

    /// Picks the bundle a new document boots (the provisional-activation rules) and returns it.
    func selectForDocument() -> Serving {
        let picked: Serving = queue.sync {
            var s = state
            if s.active != embedded && !s.confirmed {
                if !s.tried {
                    s.tried = true
                } else {
                    s.failed = s.active
                    s.active = s.previous
                    s.previous = embedded
                    s.confirmed = true
                    s.tried = false
                }
            }
            if s.active != embedded
                && !FileManager.default.fileExists(atPath: root(s.active).appendingPathComponent("index.html").path)
            {
                s.failed = s.active
                s.active = embedded
                s.previous = embedded
                s.confirmed = true
                s.tried = false
            }
            if s != state {
                do { try persist(s) } catch {
                    // A trial that cannot be recorded is not started.
                    return Serving(version: embedded, root: embeddedRoot, confirmed: true)
                }
            }
            return Serving(version: s.active, root: root(s.active), confirmed: s.confirmed)
        }
        lock.withLock { serving = picked }
        queue.async {
            let digest = try? Self.digest(of: picked.root).digest
            self.lock.withLock { if self.serving.root == picked.root { self.serving.digest = digest } }
        }
        return picked
    }

    /// The serving bundle, once its digest is computed.
    func servingInfo() -> (serving: Serving, active: String) {
        queue.sync { (lock.withLock { serving }, state.active) }
    }

    func servingRoot() -> URL { lock.withLock { serving.root } }

    // MARK: Requests

    private func version(_ p: Params) throws -> String {
        let v = try p.string("version")
        guard Names.isVersion(v) else { throw ShellError.bad("version: not a valid name") }
        return v
    }

    private func isStaging(_ v: String) -> Bool {
        v != embedded && state.sealed[v] == nil
            && FileManager.default.fileExists(atPath: root(v).path)
    }

    func handle(_ method: String, _ p: Params) throws -> [String: Any] {
        switch method {
        case "bundle.begin":
            let v = try version(p)
            guard v != embedded, state.sealed[v] == nil else { throw ShellError.state("sealed or embedded") }
            try? FileManager.default.removeItem(at: root(v))
            try FileManager.default.createDirectory(at: root(v), withIntermediateDirectories: true)
            return [:]
        case "bundle.write":
            let v = try version(p)
            let path = try p.string("path")
            let offset = try p.int("offset")
            let data = try p.bytes("data")
            guard Names.isBundlePath(path) else { throw ShellError.bad("path: not a valid bundle path") }
            guard isStaging(v) else { throw ShellError.state("not staging") }
            let file = root(v).appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let fd = open(file.path, O_WRONLY | O_CREAT, 0o600)
            guard fd >= 0 else { throw ShellError.errno("open") }
            defer { close(fd) }
            try data.withUnsafeBytes { buf in
                var done = 0
                while done < buf.count {
                    let n = pwrite(fd, buf.baseAddress! + done, buf.count - done, off_t(offset) + off_t(done))
                    if n < 0 { throw ShellError.errno("write") }
                    done += n
                }
            }
            return [:]
        case "bundle.seal":
            let v = try version(p)
            guard isStaging(v) else { throw ShellError.state("not staging") }
            // The version's files, its directories and its own name, before the state says sealed.
            try Self.syncTree(root(v))
            try Durable.directory(versions.path)
            let d = try Self.digest(of: root(v))
            var s = state
            s.sealed[v] = d.digest
            try persist(s)
            return ["digest": d.digest, "files": d.files, "bytes": d.bytes]
        case "bundle.activate":
            let v = try version(p)
            var s = state
            if v == embedded {
                s.active = embedded
                s.previous = embedded
                s.confirmed = true
                s.tried = false
            } else {
                guard s.sealed[v] != nil else {
                    throw isStaging(v) ? ShellError.state("not sealed") : ShellError.notFound("no such version")
                }
                if !(s.active == v && s.confirmed) {
                    if s.confirmed { s.previous = s.active }
                    s.active = v
                    s.confirmed = false
                    s.tried = false
                }
            }
            try persist(s)
            return stateResult()
        case "bundle.confirm":
            let current = lock.withLock { serving }
            if current.version == state.active && !state.confirmed {
                var s = state
                s.confirmed = true
                s.tried = false
                s.previous = embedded
                try persist(s)
                lock.withLock { serving.confirmed = true }
            }
            return stateResult()
        case "bundle.remove":
            let v = try version(p)
            let current = lock.withLock { serving.version }
            guard v != embedded, v != state.active, v != state.previous, v != current else {
                throw ShellError.state("embedded, active, previous or serving")
            }
            guard FileManager.default.fileExists(atPath: root(v).path) || state.sealed[v] != nil else {
                throw ShellError.notFound("no such version")
            }
            var s = state
            s.sealed[v] = nil
            try persist(s)
            try? FileManager.default.removeItem(at: root(v))
            return [:]
        case "bundle.state":
            return stateResult()
        default:
            throw ShellError(code: "unknown_method", message: method)
        }
    }

    private func stateResult() -> [String: Any] {
        let current = lock.withLock { serving }
        let names = Set((try? FileManager.default.contentsOfDirectory(atPath: versions.path)) ?? [])
            .union(state.sealed.keys).filter(Names.isVersion).sorted()
        return [
            "serving": [
                "version": current.version, "digest": orNull(current.digest),
                "confirmed": current.confirmed,
            ],
            "active": state.active, "confirmed": state.confirmed, "previous": state.previous,
            "failed": orNull(state.failed),
            "versions": names.map { v -> [String: Any] in
                ["version": v, "sealed": state.sealed[v] != nil, "digest": orNull(state.sealed[v])]
            },
        ]
    }

    private func orNull(_ s: String?) -> Any { s.map { $0 as Any } ?? NSNull() }

    // MARK: Digest

    /// Every servable file of a bundle by its bundle path, sorted by bytes.
    static func paths(of root: URL) throws -> [String] {
        let fm = FileManager.default
        return try fm.subpathsOfDirectory(atPath: root.path).filter { p in
            var isDir: ObjCBool = false
            return Names.isBundlePath(p) && fm.fileExists(atPath: root.appendingPathComponent(p).path, isDirectory: &isDir)
                && !isDir.boolValue
        }.sorted { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }
    }

    /// SHA-256 over the `sha256sum`-format manifest of the bundle's files.
    static func digest(of root: URL) throws -> (digest: String, files: Int, bytes: Int64) {
        var manifest = ""
        var bytes: Int64 = 0
        let paths = try paths(of: root)
        for p in paths {
            guard let h = FileHandle(forReadingAtPath: root.appendingPathComponent(p).path) else {
                throw ShellError.io("open \(p)")
            }
            defer { try? h.close() }
            var sha = SHA256()
            while let chunk = try h.read(upToCount: 1 << 20), !chunk.isEmpty {
                sha.update(data: chunk)
                bytes += Int64(chunk.count)
            }
            manifest += hex(sha.finalize()) + "  " + p + "\n"
        }
        return (hex(SHA256.hash(data: Data(manifest.utf8))), paths.count, bytes)
    }

    static func hex<D: Sequence>(_ d: D) -> String where D.Element == UInt8 {
        d.map { String(format: "%02x", $0) }.joined()
    }

    /// Makes every file and directory under `root` durable.
    static func syncTree(_ root: URL) throws {
        for p in try FileManager.default.subpathsOfDirectory(atPath: root.path) + [""] {
            let path = root.appendingPathComponent(p).path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw ShellError.errno("open") }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw ShellError.errno("stat") }
            if st.st_mode & S_IFMT == S_IFDIR {
                try Durable.directory(path)
            } else {
                try Durable.file(fd, path)
            }
        }
    }
}

/// Serves the selected bundle on the shell's origin.
final class SchemeHandler: NSObject, WKURLSchemeHandler {
    let store: BundleStore
    private let io = DispatchQueue(label: "ai.eidola.shell.serve", attributes: .concurrent)
    /// Tasks started and neither finished nor stopped. Main thread only.
    private var pending = Set<ObjectIdentifier>()

    init(store: BundleStore) { self.store = store }

    static let csp =
        "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; "
        + "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; "
        + "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

    static let types: [String: String] = [
        "html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8",
        "mjs": "text/javascript; charset=utf-8", "wasm": "application/wasm", "css": "text/css; charset=utf-8",
        "json": "application/json", "map": "application/json", "txt": "text/plain; charset=utf-8",
        "svg": "image/svg+xml", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "ico": "image/x-icon", "woff2": "font/woff2", "woff": "font/woff",
        "ttf": "font/ttf", "otf": "font/otf",
    ]

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        let url = task.request.url!
        var path = url.path(percentEncoded: true)
        if path == "/" { path = "/index.html" }
        path.removeFirst()
        guard url.host() == Origin.host else { return respond(task, 403) }
        guard task.request.httpMethod == "GET" else { return respond(task, 405) }
        guard Names.isBundlePath(path) else { return respond(task, 404) }
        let file = store.servingRoot().appendingPathComponent(path)
        let ext = (path as NSString).pathExtension.lowercased()
        let key = ObjectIdentifier(task)
        pending.insert(key)
        io.async {
            let data = try? Data(contentsOf: file, options: .alwaysMapped)
            DispatchQueue.main.async {
                guard self.pending.remove(key) != nil else { return }
                guard let data else { return self.respond(task, 404) }
                self.respond(task, 200, type: Self.types[ext] ?? "application/octet-stream", body: data)
            }
        }
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {
        pending.remove(ObjectIdentifier(task))
    }

    private func respond(_ task: WKURLSchemeTask, _ status: Int, type: String = "text/plain; charset=utf-8", body: Data = Data()) {
        let headers = [
            "Content-Type": type, "Content-Length": String(body.count), "Content-Security-Policy": Self.csp,
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        ]
        task.didReceive(HTTPURLResponse(url: task.request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!)
        if !body.isEmpty { task.didReceive(body) }
        task.didFinish()
    }
}
