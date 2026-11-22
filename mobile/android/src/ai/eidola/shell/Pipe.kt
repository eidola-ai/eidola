// The raw TCP pipe (shell-contract.md → TCP pipe). Bytes only, never TLS.
package ai.eidola.shell

import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

class Pipe {
    private class Sock {
        val socket = Socket()
        @Volatile var connected = false
        @Volatile var closed = false
        @Volatile var shut = false
        @Volatile var eof = false
        val receiving = AtomicBoolean(false)
        /** Sends and the shutdown, in request order. */
        val writer: ExecutorService = Executors.newSingleThreadExecutor()
        val reader: ExecutorService = Executors.newSingleThreadExecutor()
    }

    private val sockets = ConcurrentHashMap<Long, Sock>()
    private val connector = Executors.newCachedThreadPool()

    private fun sock(p: Params): Sock = sockets[p.long("socket")] ?: throw ShellError.notFound("no such socket")

    private fun failure(s: Sock, e: Exception) = if (s.closed) ShellError.closed else ShellError.io(e.toString())

    fun handle(method: String, p: Params, done: Done) {
        when (method) {
            "tcp.connect" -> {
                val id = p.long("socket")
                val host = p.string("host")
                val port = p.long("port")
                if (host.isEmpty() || port < 1 || port > 65535) throw ShellError.bad("host or port out of range")
                val s = Sock()
                if (sockets.putIfAbsent(id, s) != null) throw ShellError.state("socket id in use")
                connector.execute {
                    try {
                        s.socket.tcpNoDelay = true
                        s.socket.connect(InetSocketAddress(host, port.toInt()), 30_000)
                        s.connected = true
                        done(JSONObject(), null)
                    } catch (e: Exception) {
                        if (s.closed) {
                            done(null, ShellError.closed)
                        } else {
                            sockets.remove(id, s)
                            done(null, ShellError("connect_failed", e.toString()))
                        }
                    }
                }
            }
            "tcp.send" -> {
                val s = sock(p)
                val data = p.bytes("data")
                if (!s.connected || s.shut) throw ShellError.state("not connected, or shut down")
                s.writer.execute {
                    try {
                        s.socket.getOutputStream().write(data)
                        done(JSONObject(), null)
                    } catch (e: IOException) {
                        done(null, failure(s, e))
                    }
                }
            }
            "tcp.receive" -> {
                val s = sock(p)
                val max = p.optLong("max") ?: 65536
                if (max < 1 || max > 1 shl 20) throw ShellError.bad("max: 1 to 1048576")
                if (!s.connected || !s.receiving.compareAndSet(false, true)) {
                    throw ShellError.state("not connected, or a receive is pending")
                }
                if (s.eof) {
                    s.receiving.set(false)
                    return done(JSONObject().put("eof", true), null)
                }
                s.reader.execute {
                    try {
                        val buf = ByteArray(max.toInt())
                        val n = s.socket.getInputStream().read(buf)
                        if (n < 0) s.eof = true
                        s.receiving.set(false)
                        done(if (n < 0) JSONObject().put("eof", true) else JSONObject().put("data", b64(buf, n)), null)
                    } catch (e: IOException) {
                        s.receiving.set(false)
                        done(null, failure(s, e))
                    }
                }
            }
            "tcp.shutdown" -> {
                val s = sock(p)
                if (!s.connected) throw ShellError.state("not connected")
                s.shut = true
                s.writer.execute {
                    try {
                        s.socket.shutdownOutput()
                        done(JSONObject(), null)
                    } catch (e: IOException) {
                        done(null, failure(s, e))
                    }
                }
            }
            "tcp.close" -> {
                close(p.long("socket"))
                done(JSONObject(), null)
            }
            else -> throw ShellError("unknown_method", method)
        }
    }

    /** Closes every socket: a new document never inherits the last one's. */
    fun closeAll() {
        for (id in sockets.keys.toList()) close(id)
    }

    private fun close(id: Long) {
        val s = sockets.remove(id) ?: return
        s.closed = true
        try {
            s.socket.close()
        } catch (_: IOException) {
        }
        s.writer.shutdown()
        s.reader.shutdown()
    }
}
