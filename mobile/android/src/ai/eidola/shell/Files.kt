// The app's file directory (shell-contract.md → Files). Runs on the store queue.
package ai.eidola.shell

import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile
import org.json.JSONArray
import org.json.JSONObject

/** Flushes a directory's entries, so a created, renamed or removed name is durable. */
fun syncDirectory(dir: File) {
    val fd = Os.open(dir.path, OsConstants.O_RDONLY, 0)
    try {
        Os.fsync(fd)
    } finally {
        Os.close(fd)
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
            if (f.exists()) FileInputStream(f).use { it.fd.sync() }
            syncDirectory(dir)
            JSONObject()
        }
        "files.rename" -> {
            val from = file(p, "from")
            val to = file(p, "to")
            try {
                Os.rename(from.path, to.path)
            } catch (e: ErrnoException) {
                if (e.errno == OsConstants.ENOENT) throw ShellError.notFound("no such file")
                throw ShellError.io(e.toString())
            }
            syncDirectory(dir)
            JSONObject()
        }
        "files.remove" -> {
            val f = file(p)
            if (f.exists() && !f.delete()) throw ShellError.io("remove failed")
            syncDirectory(dir)
            JSONObject()
        }
        "files.list" -> JSONObject().put("names", JSONArray((dir.list() ?: arrayOf()).filter(Names::file).sorted()))
        else -> throw ShellError("unknown_method", method)
    }

    /** Opens (creating) `f` for writing, runs `body`, and with `sync` makes the result durable. */
    private fun withFile(f: File, sync: Boolean, body: (RandomAccessFile) -> Long): Long {
        val created = !f.exists()
        val result = RandomAccessFile(f, "rw").use { raf ->
            val r = body(raf)
            if (sync) raf.fd.sync()
            r
        }
        if (sync && created) syncDirectory(dir)
        return result
    }
}
