#!/usr/bin/env python3
"""The contract test page's TCP peer: echoes every byte; when the client shuts down
its sending side, replies `bye <bytes received>\\n` and closes.

Usage: echo_server.py [PORT [HOST]]   (default 7777 on 127.0.0.1; the Android emulator
reaches that as 10.0.2.2; a device needs HOST 0.0.0.0). Nothing listens on PORT+1, so
connects there are refused.
"""
import socket
import socketserver
import sys


class Echo(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        total = 0
        while data := self.request.recv(65536):
            total += len(data)
            self.request.sendall(data)
        self.request.sendall(b"bye %d\n" % total)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7777
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    with Server((host, port), Echo) as server:
        print(f"echo server on {host}:{port}", flush=True)
        server.serve_forever()
