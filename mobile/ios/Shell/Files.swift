// The app's file directory (shell-contract.md → Files). Runs on the store queue.
import Foundation

/// Flushes `fd` to stable storage: `F_FULLFSYNC`, which plain `fsync` is not on Apple platforms.
func fullSync(_ fd: Int32) throws {
    if fcntl(fd, F_FULLFSYNC) == -1 && fsync(fd) == -1 { throw ShellError.errno("sync") }
}

/// Flushes a directory's entries, so a created, renamed or removed name is durable.
func syncDirectory(_ path: String) throws {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { throw ShellError.errno("open directory") }
    defer { close(fd) }
    if fsync(fd) == -1 { throw ShellError.errno("sync directory") }
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
                try fullSync(fd)
            } else if errno != ENOENT {
                throw ShellError.errno("open")
            }
            try syncDirectory(dir)
            return [:]
        case "files.rename":
            let from = try path(p, "from")
            let to = try path(p, "to")
            if rename(from, to) == -1 {
                if errno == ENOENT { throw ShellError.notFound("no such file") }
                throw ShellError.errno("rename")
            }
            try syncDirectory(dir)
            return [:]
        case "files.remove":
            if unlink(try path(p)) == -1 && errno != ENOENT { throw ShellError.errno("remove") }
            try syncDirectory(dir)
            return [:]
        case "files.list":
            let names = try FileManager.default.contentsOfDirectory(atPath: dir)
                .filter(Names.isFileName).sorted()
            return ["names": names]
        default:
            throw ShellError(code: "unknown_method", message: method)
        }
    }

    /// Opens (creating) `path` for writing, runs `body`, and with `sync` makes the result durable.
    private func withFile(_ path: String, sync: Bool, _ body: (Int32) throws -> Int64) throws -> Int64 {
        var fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        let created = fd >= 0
        if !created {
            guard errno == EEXIST else { throw ShellError.errno("open") }
            fd = open(path, O_WRONLY)
            guard fd >= 0 else { throw ShellError.errno("open") }
        }
        defer { close(fd) }
        let result = try body(fd)
        if sync {
            try fullSync(fd)
            if created { try syncDirectory(dir) }
        }
        return result
    }
}
