// The app's file directory (shell-contract.md → Files). Runs on the store queue.
import Foundation

/// Every flush the shell makes goes through here. Debug builds record each one (as a path
/// relative to `base`) for the contract test page to check (`debug.syncLog`). Store queue only.
enum Durable {
    static var base = ""
    #if DEBUG
        static var log: [String] = []
    #endif

    private static func record(_ path: String) {
        #if DEBUG
            log.append(path.hasPrefix(base + "/") ? String(path.dropFirst(base.count + 1)) : path)
        #endif
    }

    /// `F_FULLFSYNC`, falling back to `fsync` where a file system lacks it. On Apple platforms
    /// `fsync` stops at the drive's write cache; `F_FULLFSYNC` also flushes that (fcntl(2)),
    /// for a directory's entries as for a file's contents.
    private static func flush(_ fd: Int32) -> Bool {
        fcntl(fd, F_FULLFSYNC) != -1 || fsync(fd) != -1
    }

    /// Flushes the open file `path` to stable storage.
    static func file(_ fd: Int32, _ path: String) throws {
        guard flush(fd) else { throw ShellError.errno("sync") }
        record(path)
    }

    /// Flushes a directory's entries, so a created, renamed or removed name is durable.
    static func directory(_ path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw ShellError.errno("open directory") }
        defer { close(fd) }
        guard flush(fd) else { throw ShellError.errno("sync directory") }
        record(path)
    }
}

/// Creates `url` as a private directory: excluded from backup, protected until first unlock.
func makePrivateDirectory(_ url: URL) throws {
    try FileManager.default.createDirectory(
        at: url, withIntermediateDirectories: true,
        attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication])
    var u = url
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try u.setResourceValues(values)
}

final class Files {
    let dir: String

    init(dir: URL) throws {
        try makePrivateDirectory(dir)
        self.dir = dir.path
    }

    private func path(_ p: Params, _ key: String = "name") throws -> String {
        let name = try p.string(key)
        guard Names.isFileName(name) else { throw ShellError.bad("\(key): not a valid file name") }
        return dir + "/" + name
    }

    func handle(_ method: String, _ p: Params) throws -> [String: Any] {
        switch method {
        case "files.read":
            let path = try path(p)
            guard FileManager.default.fileExists(atPath: path) else { return ["data": NSNull()] }
            guard let data = FileManager.default.contents(atPath: path) else { throw ShellError.io("read failed") }
            return ["data": data.base64EncodedString()]
        case "files.write":
            let path = try path(p)
            let data = try p.bytes("data")
            let offset = try p.optInt("offset")
            let append = try p.bool("append")
            guard (offset != nil) != append else { throw ShellError.bad("exactly one of offset and append") }
            let size = try withFile(path, sync: try p.bool("sync")) { fd in
                var at = offset ?? Int64(lseek(fd, 0, SEEK_END))
                try data.withUnsafeBytes { buf in
                    var done = 0
                    while done < buf.count {
                        let n = pwrite(fd, buf.baseAddress! + done, buf.count - done, off_t(at))
                        if n < 0 { throw ShellError.errno("write") }
                        done += n
                        at += Int64(n)
                    }
                }
                return Int64(lseek(fd, 0, SEEK_END))
            }
            return ["size": size]
        case "files.truncate":
            let path = try path(p)
            let size = try p.int("size")
            _ = try withFile(path, sync: try p.bool("sync")) { fd in
                if ftruncate(fd, off_t(size)) == -1 { throw ShellError.errno("truncate") }
                return size
            }
            return [:]
        case "files.sync":
            let path = try path(p)
            let fd = open(path, O_RDONLY)
            if fd >= 0 {
                defer { close(fd) }
                try Durable.file(fd, path)
            } else if errno != ENOENT {
                throw ShellError.errno("open")
            }
            try Durable.directory(dir)
            return [:]
        case "files.rename":
            let from = try path(p, "from")
            let to = try path(p, "to")
            // The contents first: a durable rename promises the file now named `to` is whole.
            let fd = open(from, O_RDONLY)
            guard fd >= 0 else {
                throw errno == ENOENT ? ShellError.notFound("no such file") : ShellError.errno("open")
            }
            defer { close(fd) }
            try Durable.file(fd, from)
            if rename(from, to) == -1 { throw ShellError.errno("rename") }
            try Durable.directory(dir)
            return [:]
        case "files.remove":
            if unlink(try path(p)) == -1 && errno != ENOENT { throw ShellError.errno("remove") }
            try Durable.directory(dir)
            return [:]
        case "files.list":
            let names = try FileManager.default.contentsOfDirectory(atPath: dir)
                .filter(Names.isFileName).sorted()
            return ["names": names]
        default:
            throw ShellError(code: "unknown_method", message: method)
        }
    }

    /// Opens (creating) `path` for writing, runs `body`, and with `sync` makes the result durable:
    /// the contents, then the directory, whether or not this call created the name (an earlier
    /// unsynced write may have).
    private func withFile(_ path: String, sync: Bool, _ body: (Int32) throws -> Int64) throws -> Int64 {
        let fd = open(path, O_WRONLY | O_CREAT, 0o600)
        guard fd >= 0 else { throw ShellError.errno("open") }
        defer { close(fd) }
        let result = try body(fd)
        if sync {
            try Durable.file(fd, path)
            try Durable.directory(dir)
        }
        return result
    }
}
