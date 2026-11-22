// The bridge: one JSON request in, one JSON reply out (shell-contract.md → The bridge).
package ai.eidola.shell

import android.util.Base64
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

const val CONTRACT_VERSION = 1

class ShellError(val code: String, message: String) : Exception(message) {
    companion object {
        fun bad(m: String) = ShellError("bad_request", m)
        fun notFound(m: String) = ShellError("not_found", m)
        fun state(m: String) = ShellError("invalid_state", m)
        fun io(m: String) = ShellError("io", m)
        val closed get() = ShellError("closed", "closed")
    }
}

/** Replies to one request: a result, or an error. */
typealias Done = (JSONObject?, ShellError?) -> Unit

/** Names and paths the contract admits. */
object Names {
    private fun segment(s: String) = s.isNotEmpty() && s[0] != '.' &&
        s.all { it in 'a'..'z' || it in 'A'..'Z' || it in '0'..'9' || it == '.' || it == '_' || it == '-' }
    fun file(s: String) = s.length <= 255 && segment(s)
    fun version(s: String) = s.length <= 64 && segment(s)
    fun bundlePath(s: String) = s.split('/').all(::segment)
}

/** Typed access to a request's params. */
class Params(private val o: JSONObject) {
    fun string(k: String): String = o.opt(k) as? String ?: throw ShellError.bad("$k: expected a string")

    fun optLong(k: String): Long? {
        if (!o.has(k) || o.isNull(k)) return null
        val n = when (val v = o.get(k)) {
            is Int -> v.toLong()
            is Long -> v
            is Double -> if (v == Math.floor(v) && v >= 0 && v <= 9.007199254740991E15) v.toLong() else -1
            else -> -1
        }
        if (n < 0 || n > 9_007_199_254_740_991L) throw ShellError.bad("$k: expected a non-negative integer")
        return n
    }

    fun long(k: String): Long = optLong(k) ?: throw ShellError.bad("$k: missing")

    fun bool(k: String): Boolean {
        if (!o.has(k) || o.isNull(k)) return false
        return o.get(k) as? Boolean ?: throw ShellError.bad("$k: expected a boolean")
    }

    /** Base64 bytes, at most 8 MiB decoded. */
    fun bytes(k: String, limit: Int = 8 shl 20): ByteArray {
        val s = string(k)
        if (s.length > (limit / 3 + 1) * 4) throw ShellError.bad("$k: more than $limit bytes")
        return try {
            Base64.decode(s, Base64.NO_WRAP)
        } catch (e: IllegalArgumentException) {
            throw ShellError.bad("$k: expected base64")
        }
    }
}

fun b64(bytes: ByteArray, len: Int = bytes.size): String = Base64.encodeToString(bytes, 0, len, Base64.NO_WRAP)

class Bridge(
    private val pipe: Pipe,
    private val files: Files,
    private val store: BundleStore,
    private val shellVersion: String,
) {
    companion object {
        val METHODS = listOf(
            "bundle.activate", "bundle.begin", "bundle.confirm", "bundle.remove", "bundle.seal",
            "bundle.state", "bundle.write", "files.list", "files.read", "files.remove", "files.rename",
            "files.sync", "files.truncate", "files.write", "shell.hello", "tcp.close", "tcp.connect",
            "tcp.receive", "tcp.send", "tcp.shutdown",
        )
    }

    /** Handles one request; `send` carries its reply back on the channel it came from. */
    fun receive(text: String, send: (String) -> Unit) {
        var id: Any = JSONObject.NULL
        val done: Done = { result, error ->
            val reply = JSONObject().put("id", id)
            if (error == null) {
                reply.put("ok", true).put("result", result ?: JSONObject())
            } else {
                reply.put("ok", false).put("error", JSONObject().put("code", error.code).put("message", error.message))
            }
            send(reply.toString())
        }
        val o = try {
            JSONObject(text)
        } catch (e: JSONException) {
            return done(null, ShellError.bad("not a JSON object"))
        }
        id = o.opt("id") ?: JSONObject.NULL
        val method = o.opt("method") as? String ?: return done(null, ShellError.bad("method: missing"))
        val params = Params(o.optJSONObject("params") ?: JSONObject())
        // A test hook in debuggable builds, not part of the contract: the flushes made since the last call.
        if (method == "debug.syncLog" && Durable.log != null) {
            return store.queue.execute {
                val log = Durable.log!!
                done(JSONObject().put("syncs", JSONArray(log.toList())), null)
                log.clear()
            }
        }
        if (method !in METHODS) return done(null, ShellError("unknown_method", method))
        when {
            method == "shell.hello" -> done(
                JSONObject().put("contract", CONTRACT_VERSION).put("platform", "android")
                    .put("shell", shellVersion).put("methods", JSONArray(METHODS)),
                null,
            )
            method.startsWith("tcp.") -> guard(done) { pipe.handle(method, params, done) }
            // Files and the bundle store share one queue: requests apply one at a time, in order.
            else -> store.queue.execute {
                guard(done) {
                    done(if (method.startsWith("files.")) files.handle(method, params) else store.handle(method, params), null)
                }
            }
        }
    }

    private inline fun guard(done: Done, body: () -> Unit) {
        try {
            body()
        } catch (e: ShellError) {
            done(null, e)
        } catch (e: Exception) {
            done(null, ShellError.io(e.toString()))
        }
    }
}
