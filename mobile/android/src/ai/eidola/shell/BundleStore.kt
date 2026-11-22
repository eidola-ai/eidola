// The bundle store, its provisional activation, the digest, and the selected bundle's bytes
// (shell-contract.md → Origin and serving, Bundle store, Bundle digest).
package ai.eidola.shell

import android.content.res.AssetManager
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.InputStream
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.concurrent.Executors
import org.json.JSONArray
import org.json.JSONObject

const val EMBEDDED = "embedded"

/** What the shell keeps durably (`state.json`). */
data class StoreState(
    val active: String = EMBEDDED,
    val confirmed: Boolean = true,
    val tried: Boolean = false,
    val previous: String = EMBEDDED,
    val failed: String? = null,
    /** Sealed versions and their digests. */
    val sealed: Map<String, String> = emptyMap(),
) {
    fun toJson(): JSONObject = JSONObject().put("active", active).put("confirmed", confirmed).put("tried", tried)
        .put("previous", previous).put("failed", failed ?: JSONObject.NULL).put("sealed", JSONObject(sealed))

    companion object {
        fun fromJson(o: JSONObject): StoreState {
            val sealed = o.getJSONObject("sealed")
            return StoreState(
                o.getString("active"), o.getBoolean("confirmed"), o.getBoolean("tried"), o.getString("previous"),
                if (o.isNull("failed")) null else o.getString("failed"),
                sealed.keys().asSequence().associateWith { sealed.getString(it) },
            )
        }
    }
}

/** The bundle one document is served from. */
data class Serving(val version: String, val confirmed: Boolean, val digest: String? = null)

class BundleStore(private val dir: File, private val assets: AssetManager) {
    /** Files and the store share this queue: one request at a time, in arrival order. */
    val queue = Executors.newSingleThreadExecutor()
    private val versions = File(dir, "versions")
    private var state = StoreState()
    @Volatile private var serving = Serving(EMBEDDED, true)

    init {
        versions.mkdirs()
        try {
            state = StoreState.fromJson(JSONObject(File(dir, "state.json").readText()))
        } catch (_: Exception) {
        }
    }

    private fun root(v: String) = File(versions, v)

    private fun persist(s: StoreState) {
        val tmp = File(dir, "state.json.tmp")
        FileOutputStream(tmp).use {
            it.write(s.toJson().toString().toByteArray())
            it.fd.sync()
        }
        if (!tmp.renameTo(File(dir, "state.json"))) throw ShellError.io("rename state")
        syncDirectory(dir)
        state = s
    }

    // Selection

    /** Picks the bundle a new document boots (the provisional-activation rules). */
    fun selectForDocument(): Serving {
        val picked = queue.submit<Serving> {
            var s = state
            if (s.active != EMBEDDED && !s.confirmed) {
                s = if (!s.tried) s.copy(tried = true)
                else s.copy(failed = s.active, active = s.previous, previous = EMBEDDED, confirmed = true, tried = false)
            }
            if (s.active != EMBEDDED && !File(root(s.active), "index.html").isFile) {
                s = s.copy(failed = s.active, active = EMBEDDED, previous = EMBEDDED, confirmed = true, tried = false)
            }
            try {
                if (s != state) persist(s)
                Serving(s.active, s.confirmed)
            } catch (e: Exception) {
                // A trial that cannot be recorded is not started.
                Serving(EMBEDDED, true)
            }
        }.get()
        serving = picked
        queue.execute {
            val digest = try { digest(picked.version).first } catch (_: Exception) { null }
            if (serving.version == picked.version) serving = serving.copy(digest = digest)
        }
        return picked
    }

    /** The serving bundle (once its digest is computed) and the active version. */
    fun servingInfo(): Pair<Serving, String> = queue.submit<Pair<Serving, String>> { Pair(serving, state.active) }.get()

    /** A file of the serving bundle, or null. */
    fun open(path: String): InputStream? {
        val v = serving.version
        return if (v == EMBEDDED) {
            try { assets.open("bundle/$path") } catch (_: FileNotFoundException) { null }
        } else {
            File(root(v), path).takeIf { it.isFile }?.inputStream()
        }
    }

    // Requests

    private fun version(p: Params): String {
        val v = p.string("version")
        if (!Names.version(v)) throw ShellError.bad("version: not a valid name")
        return v
    }

    private fun isStaging(v: String) = v != EMBEDDED && v !in state.sealed && root(v).isDirectory

    fun handle(method: String, p: Params): JSONObject = when (method) {
        "bundle.begin" -> {
            val v = version(p)
            if (v == EMBEDDED || v in state.sealed) throw ShellError.state("sealed or embedded")
            root(v).deleteRecursively()
            if (!root(v).mkdirs()) throw ShellError.io("mkdir")
            JSONObject()
        }
        "bundle.write" -> {
            val v = version(p)
            val path = p.string("path")
            val offset = p.long("offset")
            val data = p.bytes("data")
            if (!Names.bundlePath(path)) throw ShellError.bad("path: not a valid bundle path")
            if (!isStaging(v)) throw ShellError.state("not staging")
            val f = File(root(v), path)
            f.parentFile!!.mkdirs()
            RandomAccessFile(f, "rw").use {
                it.seek(offset)
                it.write(data)
            }
            JSONObject()
        }
        "bundle.seal" -> {
            val v = version(p)
            if (!isStaging(v)) throw ShellError.state("not staging")
            root(v).walkBottomUp().forEach { f ->
                if (f.isFile) RandomAccessFile(f, "r").use { it.fd.sync() } else syncDirectory(f)
            }
            val (digest, files, bytes) = digest(v)
            persist(state.copy(sealed = state.sealed + (v to digest)))
            JSONObject().put("digest", digest).put("files", files).put("bytes", bytes)
        }
        "bundle.activate" -> {
            val v = version(p)
            val s = state
            persist(
                when {
                    v == EMBEDDED -> s.copy(active = EMBEDDED, previous = EMBEDDED, confirmed = true, tried = false)
                    v !in s.sealed -> throw if (isStaging(v)) ShellError.state("not sealed") else ShellError.notFound("no such version")
                    s.active == v && s.confirmed -> s
                    else -> s.copy(previous = if (s.confirmed) s.active else s.previous, active = v, confirmed = false, tried = false)
                },
            )
            stateJson()
        }
        "bundle.confirm" -> {
            if (serving.version == state.active && !state.confirmed) {
                persist(state.copy(confirmed = true, tried = false, previous = EMBEDDED))
                serving = serving.copy(confirmed = true)
            }
            stateJson()
        }
        "bundle.remove" -> {
            val v = version(p)
            if (v == EMBEDDED || v == state.active || v == state.previous || v == serving.version) {
                throw ShellError.state("embedded, active, previous or serving")
            }
            if (!root(v).exists() && v !in state.sealed) throw ShellError.notFound("no such version")
            persist(state.copy(sealed = state.sealed - v))
            root(v).deleteRecursively()
            JSONObject()
        }
        "bundle.state" -> stateJson()
        else -> throw ShellError("unknown_method", method)
    }

    private fun stateJson(): JSONObject {
        val s = serving
        val names = ((versions.list() ?: arrayOf()).toSet() + state.sealed.keys).filter(Names::version).sorted()
        return JSONObject()
            .put("serving", JSONObject().put("version", s.version).put("digest", s.digest ?: JSONObject.NULL).put("confirmed", s.confirmed))
            .put("active", state.active).put("confirmed", state.confirmed).put("previous", state.previous)
            .put("failed", state.failed ?: JSONObject.NULL)
            .put("versions", JSONArray(names.map { v ->
                JSONObject().put("version", v).put("sealed", v in state.sealed).put("digest", state.sealed[v] ?: JSONObject.NULL)
            }))
    }

    // Digest

    /** Every servable file of a version by its bundle path, sorted by bytes. */
    private fun paths(v: String): List<String> {
        val out = ArrayList<String>()
        if (v == EMBEDDED) {
            fun walk(dir: String, prefix: String) {
                for (name in assets.list(dir) ?: arrayOf()) {
                    val child = "$dir/$name"
                    if (assets.list(child)?.isNotEmpty() == true) walk(child, "$prefix$name/") else out.add("$prefix$name")
                }
            }
            walk("bundle", "")
        } else {
            val root = root(v)
            root.walk().filter { it.isFile }.forEach { out.add(it.relativeTo(root).invariantSeparatorsPath) }
        }
        return out.filter(Names::bundlePath).sorted()
    }

    /** SHA-256 over the `sha256sum`-format manifest of the version's files: digest, files, bytes. */
    fun digest(v: String): Triple<String, Int, Long> {
        val manifest = StringBuilder()
        var bytes = 0L
        val paths = paths(v)
        val buf = ByteArray(1 shl 20)
        for (path in paths) {
            val sha = MessageDigest.getInstance("SHA-256")
            val input = if (v == EMBEDDED) assets.open("bundle/$path") else File(root(v), path).inputStream()
            input.use {
                while (true) {
                    val n = it.read(buf)
                    if (n < 0) break
                    sha.update(buf, 0, n)
                    bytes += n
                }
            }
            manifest.append(hex(sha.digest())).append("  ").append(path).append('\n')
        }
        return Triple(hex(MessageDigest.getInstance("SHA-256").digest(manifest.toString().toByteArray())), paths.size, bytes)
    }

    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
}
