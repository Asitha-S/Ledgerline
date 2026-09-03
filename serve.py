"""
serve.py — local static server for web/, with caching turned off.

`python -m http.server` sends no Cache-Control, so browsers apply heuristic caching to
index.html. The page then keeps replaying a stale app.js from its own cache and edits
appear not to land, which is a confusing failure: the server is serving the new file
and the browser never asks for it.

This serves exactly the same directory and sends no-store on everything, so a plain
reload always shows what is on disk. It is a development convenience only — the
interface itself is a static export and needs no server component to be correct.

Run:  python serve.py [port]        (default 8000)
"""

from __future__ import annotations

import functools
import http.server
import os
import socket
import socketserver
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):        # one line per request is enough
        sys.stderr.write("  %s\n" % (fmt % args))


class DualStack(socketserver.ThreadingTCPServer):
    """Listen on IPv6 and IPv4 both.

    On Windows "localhost" resolves to ::1 first. A server bound only to 127.0.0.1
    leaves the browser hanging on http://localhost:PORT with no error worth reading,
    which is a miserable thing to debug. Bind :: with V6ONLY off so one socket answers
    both, and fall back to plain IPv4 where that is not permitted."""

    allow_reuse_address = True
    daemon_threads = True          # a browser opens several connections at once, and a
    address_family = socket.AF_INET6   # serialised server stalls the page mid-load

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not os.path.isdir(ROOT):
        sys.exit(f"{ROOT} not found — run export.py first")
    handler = functools.partial(NoCache, directory=ROOT)
    try:
        httpd = DualStack(("::", port), handler)
        stack = "IPv6 + IPv4"
    except OSError:
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        socketserver.ThreadingTCPServer.daemon_threads = True
        httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), handler)
        stack = "IPv4"
    with httpd:
        print(f"serving {ROOT} on http://localhost:{port}/  (no-store, {stack})")
        print("Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
