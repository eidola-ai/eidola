// tcp.connect's dialling (shell-contract.md → TCP pipe): every address a name resolves to, within
// one deadline. Only java.net, so a host JVM runs it in test-connect.sh.
package ai.eidola.shell

import java.io.IOException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket

/**
 * Connects to the first of [addresses], in the resolver's order, that accepts a connection to
 * [port] before [giveUp] (a [System.nanoTime] instant), and returns that socket. A refused or
 * unreachable address is passed over at once; one that does not answer is given an equal share of
 * the time left among the addresses not yet tried, so a broken first route (an IPv6 path that drops
 * everything) still leaves time for a working one, and the last address gets whatever remains.
 *
 * [attempting] is handed each new socket before it dials, so a close can abort the dial in flight
 * by closing it; it answers false once the connect has been abandoned, and nothing more is dialled.
 * Throws the last attempt's [IOException] when none connects.
 */
internal fun connectFirst(
    addresses: List<InetAddress>,
    port: Int,
    giveUp: Long,
    attempting: (Socket) -> Boolean,
): Socket {
    var failure: IOException = IOException("the name resolved to no address")
    for ((i, address) in addresses.withIndex()) {
        val left = (giveUp - System.nanoTime()) / 1_000_000
        if (left <= 0) throw IOException("no connection within the deadline (last: $failure)")
        val share = maxOf(1L, left / (addresses.size - i))
        val socket = Socket()
        if (!attempting(socket)) {
            closeQuietly(socket)
            throw IOException("abandoned")
        }
        try {
            socket.tcpNoDelay = true
            socket.connect(InetSocketAddress(address, port), share.toInt())
            return socket
        } catch (e: IOException) {
            closeQuietly(socket)
            failure = IOException("$address: $e", e)
        }
    }
    throw failure
}

private fun closeQuietly(socket: Socket) {
    try {
        socket.close()
    } catch (_: IOException) {
    }
}
