// The app's file directory (shell-contract.md → Files). Runs on the store queue.
package ai.eidola.shell

import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import java.io.File
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.RandomAccessFile
import org.json.JSONArray
import org.json.JSONObject

/**
 * Every flush the shell makes goes through here. Debuggable builds record each one (as a path
 * relative to `base`) for the contract test page to check (`debug.syncLog`). Store queue only.
 */
object Durable {
    lateinit var base: File
    /** Null unless the build is debuggable. */
    var log: MutableList<String>? = null

    private fun record(f: File) {
        log?.add(f.relativeTo(base).invariantSeparatorsPath)
    }

    /** Flushes the open file `f` to stable storage. */
    fun file(fd: FileDescriptor, f: File) {
        fd.sync()
        record(f)
    }

    /** Flushes a directory's entries, so a created, renamed or removed name is durable. */
    fun directory(dir: File) {
        val fd = Os.open(dir.path, OsConstants.O_RDONLY, 0)
        try {
            Os.fsync(fd)
        } finally {
            Os.close(fd)
        }
        record(dir)
    }
}

class Files(private val dir: File) {
    init {
        dir.mkdirs()
    }

    private fun file(p: Params, key: String = "name"): File {
        val name = p.string(key)
        if (!Names.file(name)) throw ShellError.bad("$key: not a valid file name")
        return File(dir, name)
    }

    fun handle(method: String, p: Params): JSONObject = when (method) {
        "files.read" -> {
            val f = file(p)
            JSONObject().put("data", if (f.exists()) b64(f.readBytes()) else JSONObject.NULL)
        }
        "files.write" -> {
            val f = file(p)
            val data = p.bytes("data")
            val offset = p.optLong("offset")
            val append = p.bool("append")
            if ((offset != null) == append) throw ShellError.bad("exactly one of offset and append")
            JSONObject().put("size", withFile(f, p.bool("sync")) { raf ->
                raf.seek(offset ?: raf.length())
                raf.write(data)
                raf.length()
            })
        }
        "files.truncate" -> {
            val f = file(p)
            val size = p.long("size")
            withFile(f, p.bool("sync")) { raf -> raf.setLength(size); size }
            JSONObject()
        }
        "files.sync" -> {
            val f = file(p)
            if (f.exists()) FileInputStream(f).use { Durable.file(it.fd, f) }
            Durable.directory(dir)
            JSONObject()
        }
        "files.rename" -> {
            val from = file(p, "from")
            val to = file(p, "to")
            // The contents first: a durable rename promises the file now named `to` is whole.
            if (!from.isFile) throw ShellError.notFound("no such file")
            FileInputStream(from).use { Durable.file(it.fd, from) }
            try {
                Os.rename(from.path, to.path)
            } catch (e: ErrnoException) {
                if (e.errno == OsConstants.ENOENT) throw ShellError.notFound("no such file")
                throw ShellError.io(e.toString())
            }
            Durable.directory(dir)
            JSONObject()
        }
        "files.remove" -> {
            val f = file(p)
            if (f.exists() && !f.delete()) throw ShellError.io("remove failed")
            Durable.directory(dir)
            JSONObject()
        }
        "files.list" -> JSONObject().put("names", JSONArray((dir.list() ?: arrayOf()).filter(Names::file).sorted()))
        else -> throw ShellError("unknown_method", method)
    }

    /**
     * Opens (creating) `f` for writing, runs `body`, and with `sync` makes the result durable: the
     * contents, then the directory, whether or not this call created the name (an earlier unsynced
     * write may have).
     */
    private fun withFile(f: File, sync: Boolean, body: (RandomAccessFile) -> Long): Long {
        val result = RandomAccessFile(f, "rw").use { raf ->
            val r = body(raf)
            if (sync) Durable.file(raf.fd, f)
            r
        }
        if (sync) Durable.directory(dir)
        return result
    }
}
