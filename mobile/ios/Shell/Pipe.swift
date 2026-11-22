// The raw TCP pipe (shell-contract.md → TCP pipe), on Network.framework. Bytes only, never TLS.
import Foundation
import Network

final class Pipe {
    private let queue = DispatchQueue(label: "ai.eidola.shell.pipe")
    static let connectDeadline: TimeInterval = 30

    private final class Socket {
        let conn: NWConnection
        var connecting: Done?
        var receiving = false
        var shut = false
        var eof = false
        var closed = false
        init(_ conn: NWConnection) { self.conn = conn }
    }

    /// Open sockets by the app's ids. Touched only on `queue`.
    private var sockets: [Int64: Socket] = [:]

    func handle(_ method: String, _ p: Params, _ done: @escaping Done) {
        queue.async {
            do { try self.apply(method, p, done) } catch let e as ShellError {
                done(.failure(e))
            } catch { done(.failure(.io("\(error)"))) }
        }
    }

    /// Closes every socket: a new document never inherits the last one's.
    func closeAll() {
        queue.async {
            for id in Array(self.sockets.keys) { self.close(id) }
        }
    }

    private func socket(_ p: Params) throws -> (Int64, Socket) {
        let id = try p.int("socket")
        guard let s = sockets[id] else { throw ShellError.notFound("no such socket") }
        return (id, s)
    }

    private func apply(_ method: String, _ p: Params, _ done: @escaping Done) throws {
        switch method {
        case "tcp.connect":
            let id = try p.int("socket")
            let host = try p.string("host")
            let port = try p.int("port")
            guard sockets[id] == nil else { throw ShellError.state("socket id in use") }
            guard !host.isEmpty, let nwPort = NWEndpoint.Port(rawValue: UInt16(clamping: port)), port >= 1, port <= 65535
            else { throw ShellError.bad("host or port out of range") }
            let tcp = NWProtocolTCP.Options()
            tcp.noDelay = true
            tcp.connectionTimeout = Int(Self.connectDeadline)
            let conn = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: NWParameters(tls: nil, tcp: tcp))
            let s = Socket(conn)
            s.connecting = done
            sockets[id] = s
            conn.stateUpdateHandler = { [weak self, weak s] state in
                if case .cancelled = state { conn.stateUpdateHandler = nil }
                guard let self, let s else { return }
                switch state {
                case .ready:
                    s.connecting?(.success([:]))
                    s.connecting = nil
                case .waiting(let e), .failed(let e):
                    // NWConnection waits (and would retry) on a refused or unreachable peer;
                    // the contract fails the connect instead.
                    if let c = s.connecting {
                        s.connecting = nil
                        c(.failure(ShellError(code: "connect_failed", message: "\(e)")))
                        if self.sockets[id] === s { self.sockets[id] = nil }
                        conn.cancel()
                    }
                default:
                    break
                }
            }
            conn.start(queue: queue)
            // One deadline for the whole connect, name resolution included: NWConnection's own
            // timeout covers the TCP handshake only.
            queue.asyncAfter(deadline: .now() + Self.connectDeadline) { [weak self, weak s] in
                guard let self, let s, let c = s.connecting else { return }
                s.connecting = nil
                c(.failure(ShellError(code: "connect_failed", message: "no connection within \(Int(Self.connectDeadline)) s")))
                if self.sockets[id] === s { self.sockets[id] = nil }
                s.conn.cancel()
            }
        case "tcp.send":
            let (_, s) = try socket(p)
            let data = try p.bytes("data")
            guard s.connecting == nil, !s.shut else { throw ShellError.state("not connected, or shut down") }
            s.conn.send(content: data, completion: .contentProcessed { [weak s] e in
                done(Self.outcome(e, closed: s?.closed ?? true))
            })
        case "tcp.receive":
            let (_, s) = try socket(p)
            let max = Int(try p.optInt("max") ?? 65536)
            guard max >= 1, max <= 1 << 20 else { throw ShellError.bad("max: 1 to 1048576") }
            guard s.connecting == nil, !s.receiving else { throw ShellError.state("not connected, or a receive is pending") }
            if s.eof { return done(.success(["eof": true])) }
            s.receiving = true
            s.conn.receive(minimumIncompleteLength: 1, maximumLength: max) { [weak s] data, _, complete, e in
                guard let s else { return done(.failure(ShellError(code: "closed", message: "closed"))) }
                s.receiving = false
                if let data, !data.isEmpty {
                    if complete { s.eof = true }
                    done(.success(["data": data.base64EncodedString()]))
                } else if complete {
                    s.eof = true
                    done(.success(["eof": true]))
                } else {
                    done(Self.outcome(e, closed: s.closed))
                }
            }
        case "tcp.shutdown":
            let (_, s) = try socket(p)
            guard s.connecting == nil else { throw ShellError.state("not connected") }
            s.shut = true
            s.conn.send(content: nil, contentContext: .finalMessage, isComplete: true,
                        completion: .contentProcessed { [weak s] e in done(Self.outcome(e, closed: s?.closed ?? true)) })
        case "tcp.close":
            close(try p.int("socket"))
            done(.success([:]))
        default:
            throw ShellError(code: "unknown_method", message: method)
        }
    }

    private func close(_ id: Int64) {
        guard let s = sockets.removeValue(forKey: id) else { return }
        s.closed = true
        s.connecting?(.failure(ShellError(code: "closed", message: "closed")))
        s.connecting = nil
        s.conn.cancel()
    }

    private static func outcome(_ e: NWError?, closed: Bool) -> Result_ {
        if closed { return .failure(ShellError(code: "closed", message: "closed")) }
        if let e { return .failure(.io("\(e)")) }
        return .success([:])
    }
}
