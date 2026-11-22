// The bridge: one JSON request in, one JSON reply out (shell-contract.md → The bridge).
import Foundation
import UIKit
import WebKit

let contractVersion = 2
let shellVersion = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0"

struct ShellError: Error {
    let code: String
    let message: String

    static func bad(_ m: String) -> ShellError { ShellError(code: "bad_request", message: m) }
    static func notFound(_ m: String) -> ShellError { ShellError(code: "not_found", message: m) }
    static func state(_ m: String) -> ShellError { ShellError(code: "invalid_state", message: m) }
    static func io(_ m: String) -> ShellError { ShellError(code: "io", message: m) }
    /// The current `errno` as an `io` error.
    static func errno(_ what: String) -> ShellError {
        ShellError(code: "io", message: "\(what): \(String(cString: strerror(Foundation.errno)))")
    }
}

typealias Result_ = Result<[String: Any], ShellError>
typealias Done = (Result_) -> Void

/// Names and paths the contract admits.
enum Names {
    static func isSegment(_ s: Substring) -> Bool {
        !s.isEmpty && !s.hasPrefix(".")
            && s.unicodeScalars.allSatisfy { c in
                (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || (c >= "0" && c <= "9")
                    || c == "." || c == "_" || c == "-"
            }
    }
    static func isFileName(_ s: String) -> Bool { s.utf8.count <= 255 && isSegment(Substring(s)) }
    static func isVersion(_ s: String) -> Bool { s.utf8.count <= 64 && isSegment(Substring(s)) }
    static func isBundlePath(_ s: String) -> Bool {
        s.split(separator: "/", omittingEmptySubsequences: false).allSatisfy(isSegment)
    }
}

/// Typed access to a request's params.
struct Params {
    let raw: [String: Any]

    func string(_ k: String) throws -> String {
        guard let v = raw[k] as? String else { throw ShellError.bad("\(k): expected a string") }
        return v
    }
    func optInt(_ k: String) throws -> Int64? {
        guard let v = raw[k], !(v is NSNull) else { return nil }
        guard let n = v as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else {
            throw ShellError.bad("\(k): expected an integer")
        }
        let d = n.doubleValue
        guard d >= 0, d <= 9_007_199_254_740_991, d == d.rounded() else {
            throw ShellError.bad("\(k): expected a non-negative integer")
        }
        return Int64(d)
    }
    func int(_ k: String) throws -> Int64 {
        guard let v = try optInt(k) else { throw ShellError.bad("\(k): missing") }
        return v
    }
    func bool(_ k: String) throws -> Bool {
        guard let v = raw[k], !(v is NSNull) else { return false }
        guard let n = v as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else {
            throw ShellError.bad("\(k): expected a boolean")
        }
        return n.boolValue
    }
    /// Base64 bytes, at most 8 MiB decoded.
    func bytes(_ k: String, limit: Int = 8 << 20) throws -> Data {
        let s = try string(k)
        guard s.count <= (limit / 3 + 1) * 4, let d = Data(base64Encoded: s) else {
            throw ShellError.bad("\(k): expected base64 of at most \(limit) bytes")
        }
        return d
    }
}

final class Bridge: NSObject, WKScriptMessageHandlerWithReply {
    let pipe: Pipe
    let files: Files
    let store: BundleStore

    static let methods = [
        "bundle.activate", "bundle.begin", "bundle.confirm", "bundle.remove", "bundle.seal",
        "bundle.state", "bundle.write", "files.list", "files.read", "files.remove", "files.rename",
        "files.sync", "files.truncate", "files.write", "shell.hello", "shell.openExternal", "tcp.close", "tcp.connect",
        "tcp.receive", "tcp.send", "tcp.shutdown",
    ]

    init(pipe: Pipe, files: Files, store: BundleStore) {
        self.pipe = pipe
        self.files = files
        self.store = store
    }

    func userContentController(
        _ controller: WKUserContentController, didReceive message: WKScriptMessage,
        replyHandler: @escaping (Any?, String?) -> Void
    ) {
        let origin = message.frameInfo.securityOrigin
        guard message.frameInfo.isMainFrame, origin.protocol == Origin.scheme, origin.host == Origin.host
        else { return replyHandler(nil, "refused") }
        var id: Any = NSNull()
        let send: Done = { result in
            var reply: [String: Any] = ["id": id]
            switch result {
            case .success(let r):
                reply["ok"] = true
                reply["result"] = r
            case .failure(let e):
                reply["ok"] = false
                reply["error"] = ["code": e.code, "message": e.message]
            }
            let text = (try? JSONSerialization.data(withJSONObject: reply)).flatMap {
                String(data: $0, encoding: .utf8)
            }
            DispatchQueue.main.async { replyHandler(text, nil) }
        }
        guard let body = message.body as? String,
            let obj = try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any]
        else { return send(.failure(.bad("not a JSON object"))) }
        id = obj["id"] ?? NSNull()
        guard let method = obj["method"] as? String else { return send(.failure(.bad("method: missing"))) }
        let params = Params(raw: obj["params"] as? [String: Any] ?? [:])
        dispatch(method, params, send)
    }

    /// An absolute `https:` URL with a host, in printable ASCII; anything else is refused.
    static func externalURL(_ p: Params) throws -> URL {
        let s = try p.string("url")
        guard s.unicodeScalars.allSatisfy({ $0.value >= 0x21 && $0.value <= 0x7e }),
            let c = URLComponents(string: s), c.scheme?.lowercased() == "https",
            let host = c.host, !host.isEmpty, let url = c.url
        else { throw ShellError.bad("url: not an absolute https URL") }
        return url
    }

    func dispatch(_ method: String, _ p: Params, _ done: @escaping Done) {
        if method == "shell.hello" {
            return done(.success([
                "contract": contractVersion, "platform": "ios", "shell": shellVersion,
                "methods": Bridge.methods,
            ]))
        }
        if method == "shell.openExternal" {
            // The system browser, outside the app; never this webview, never an in-app browser.
            do {
                let url = try Self.externalURL(p)
                return UIApplication.shared.open(url) { opened in
                    done(opened ? .success([:]) : .failure(.io("the system did not open the URL")))
                }
            } catch let e as ShellError {
                return done(.failure(e))
            } catch { return done(.failure(.io("\(error)"))) }
        }
        #if DEBUG
            // A test hook, not part of the contract: the flushes made since the last call.
            if method == "debug.syncLog" {
                return store.queue.async {
                    done(.success(["syncs": Durable.log]))
                    Durable.log = []
                }
            }
        #endif
        if method.hasPrefix("tcp.") { return pipe.handle(method, p, done) }
        let handler: ((String, Params) throws -> [String: Any])?
        if method.hasPrefix("files.") {
            handler = files.handle
        } else if method.hasPrefix("bundle.") {
            handler = store.handle
        } else {
            handler = nil
        }
        guard let handler, Bridge.methods.contains(method) else {
            return done(.failure(ShellError(code: "unknown_method", message: method)))
        }
        // Files and the bundle store share one queue: requests apply one at a time, in order.
        store.queue.async {
            do { done(.success(try handler(method, p))) } catch let e as ShellError {
                done(.failure(e))
            } catch { done(.failure(.io("\(error)"))) }
        }
    }
}
