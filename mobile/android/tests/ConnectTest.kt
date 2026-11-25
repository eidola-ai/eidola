// tcp.connect's dialling (src/ai/eidola/shell/Connect.kt) on a host JVM: run by test-connect.sh.
package ai.eidola.shell

import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket

private var failures = 0

private fun check(what: String, ok: Boolean) {
    println((if (ok) "ok   " else "FAIL ") + what)
    if (!ok) failures++
}

/** A listener on IPv4 loopback alone, so the same port on IPv6 loopback refuses. */
private fun listener(): ServerSocket = ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"))

private fun dial(addresses: List<InetAddress>, port: Int, deadlineMs: Long): Result<Socket> =
    runCatching { connectFirst(addresses, port, System.nanoTime() + deadlineMs * 1_000_000) { true } }

fun main() {
    val v4 = InetAddress.getByName("127.0.0.1")
    val v6 = InetAddress.getByName("::1")
    // TEST-NET-1 (RFC 5737): routed nowhere, so a connect there is dropped or refused by the network.
    val blackhole = InetAddress.getByName("192.0.2.1")

    listener().use { server ->
        val port = server.localPort

        val refusedFirst = dial(listOf(v6, v4), port, 5_000)
        check("a first address that refuses is passed over for one that answers",
            refusedFirst.getOrNull()?.inetAddress == v4)
        refusedFirst.getOrNull()?.close()

        val start = System.nanoTime()
        val silentFirst = dial(listOf(blackhole, v4), port, 4_000)
        val tookMs = (System.nanoTime() - start) / 1_000_000
        check("a first address that never answers leaves time for one that does (${tookMs} ms)",
            silentFirst.getOrNull()?.inetAddress == v4 && tookMs < 4_000)
        silentFirst.getOrNull()?.close()

        val first = dial(listOf(v4, v6), port, 5_000)
        check("the resolver's first address is dialled first", first.getOrNull()?.inetAddress == v4)
        first.getOrNull()?.close()
    }

    val closedPort = listener().use { it.localPort }
    val start = System.nanoTime()
    val none = dial(listOf(v6, blackhole, v4), closedPort, 2_000)
    val tookMs = (System.nanoTime() - start) / 1_000_000
    check("no address answering fails within the deadline (${tookMs} ms)",
        none.exceptionOrNull() is IOException && tookMs < 3_000)

    var dialled = 0
    val abandoned = runCatching {
        connectFirst(listOf(v6, v4), 9, System.nanoTime() + 2_000_000_000L) { dialled++; false }
    }
    check("an abandoned connect dials nothing more", abandoned.isFailure && dialled == 1)

    if (failures > 0) {
        println("$failures failed")
        kotlin.system.exitProcess(1)
    }
    println("all passed")
}
