// The raw TCP pipe (shell-contract.md → TCP pipe). Bytes only, never TLS.
package ai.eidola.shell

import java.io.IOException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONObject

class Pipe {
    private class Sock {
        val socket = Socket()
        @Volatile var connected = false
        @Volatile var closed = false
        @Volatile var shut = false
        @Volatile var eof = false
        val receiving = AtomicBoolean(false)
        /** The pending connect's reply, taken by whichever answers first: the worker, the deadline, or a close. */
        val connecting = AtomicReference<Done?>()
        /** Sends and the shutdown, in request order. */
        val writer: ExecutorService = Executors.newSingleThreadExecutor()
        val reader: ExecutorService = Executors.newSingleThreadExecutor()
    }

    private val sockets = ConcurrentHashMap<Long, Sock>()
    private val connector = Executors.newCachedThreadPool()
    private val deadlines = Executors.newSingleThreadScheduledExecutor()

    /** Answers the pending connect, once; false when something already did. */
    private fun answer(s: Sock, result: JSONObject?, error: ShellError?): Boolean {
        val d = s.connecting.getAndSet(null) ?: return false
        d(result, error)
        return true
    }

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
                s.connecting.set(done)
                // One deadline for the whole connect, name resolution included.
                val giveUp = System.nanoTime() + CONNECT_DEADLINE_MS * 1_000_000
                val timer = deadlines.schedule({
                    if (answer(s, null, ShellError("connect_failed", "no connection within ${CONNECT_DEADLINE_MS / 1000} s"))) {
                        sockets.remove(id, s)
                        s.closed = true
                        closeQuietly(s)
                    }
                }, CONNECT_DEADLINE_MS, TimeUnit.MILLISECONDS)
                connector.execute {
                    try {
                        // Blocks for as long as the resolver takes; the deadline and a close answer
                        // without waiting for it, and a socket it returns too late is closed below.
                        val address = InetAddress.getByName(host)
                        val left = (giveUp - System.nanoTime()) / 1_000_000
                        if (left <= 0 || s.connecting.get() == null) throw IOException("abandoned")
                        s.socket.tcpNoDelay = true
                        s.socket.connect(InetSocketAddress(address, port.toInt()), left.toInt())
                        s.connected = true
                        timer.cancel(false)
                        if (!answer(s, JSONObject(), null)) closeQuietly(s)
                    } catch (e: Exception) {
                        timer.cancel(false)
                        if (answer(s, null, ShellError("connect_failed", e.toString()))) sockets.remove(id, s)
                        closeQuietly(s)
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
        answer(s, null, ShellError.closed)
        closeQuietly(s)
    }

    private fun closeQuietly(s: Sock) {
        try {
            s.socket.close()
        } catch (_: IOException) {
        }
        s.writer.shutdown()
        s.reader.shutdown()
    }

    companion object {
        const val CONNECT_DEADLINE_MS = 30_000L
    }
}
