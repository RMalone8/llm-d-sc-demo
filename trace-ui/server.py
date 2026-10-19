#!/usr/bin/env python3
"""A dependency-free OpenAI relay with a small persisted request trace view."""
from __future__ import annotations

import gzip
import http.client
import json
import os
import sqlite3
import ssl
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit
from urllib.request import HTTPHandler, Request, build_opener, urlopen

DATABASE = Path(os.environ.get("TRACE_DB_PATH", "/data/traces.db"))
UPSTREAM = os.environ.get("PRAXIS_UPSTREAM", "http://praxis:8080").rstrip("/")
CPU_UPSTREAM = os.environ.get("CPU_UPSTREAM", "http://vllm-cpu-predictor:8080").rstrip("/")
GPU_UPSTREAM = os.environ.get("GPU_UPSTREAM", "http://vllm-gpu-predictor:8080").rstrip("/")
CONTEXT_PROBE_MARKER = os.environ.get("TRACE_CONTEXT_PROBE_MARKER", "")
MAX_TRACES = int(os.environ.get("TRACE_MAX_TRACES", "250"))
# Raw response bytes kept per trace while relaying (enough to parse an SSE stream),
# and the longest answer text stored from them.
RESPONSE_CAPTURE_BYTES = int(os.environ.get("TRACE_RESPONSE_CAPTURE_BYTES", str(512 * 1024)))
RESPONSE_MAX_CHARS = int(os.environ.get("TRACE_RESPONSE_MAX_CHARS", "16000"))
GPU_LABELS = {"COMPLEX", "REASONING"}
# Praxis's active health checks reach the predictors through these relays; GET
# HEALTH_PATH on a relay is recorded as a probe and feeds the pod health shown in
# the architecture diagram.
HEALTH_PATH = os.environ.get("TRACE_HEALTH_PATH", "/health")
HEALTH_ROUTES = ("vllm-cpu", "vllm-gpu")
HEALTH_KEEP = int(os.environ.get("TRACE_HEALTH_KEEP", "300"))
PROBE_FRESH_SECONDS = float(os.environ.get("TRACE_PROBE_FRESH_SECONDS", "20"))
# Cluster state for the diagram: polled from the Kubernetes API and stored whenever
# it changes, so a request can be shown against the state it was sent into.
STATE_POLL_SECONDS = float(os.environ.get("TRACE_STATE_POLL_SECONDS", "2"))
STATE_KEEP = int(os.environ.get("TRACE_STATE_KEEP", "2000"))
K8S_API = os.environ.get("KUBERNETES_API", "https://kubernetes.default.svc")
SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
# relay route -> (Deployment name, InferenceService label value)
POOLS = {
    "vllm-cpu": (os.environ.get("TRACE_CPU_DEPLOYMENT", "vllm-cpu-predictor"), os.environ.get("TRACE_CPU_ISVC", "vllm-cpu")),
    "vllm-gpu": (os.environ.get("TRACE_GPU_DEPLOYMENT", "vllm-gpu-predictor"), os.environ.get("TRACE_GPU_ISVC", "vllm-gpu")),
}
# Traffic generator the Simulate Traffic button scales between 0 and 1 replicas.
TRAFFIC_DEPLOYMENT = os.environ.get("TRACE_TRAFFIC_DEPLOYMENT", "praxis-traffic-gen")
TRAFFIC_LABEL = os.environ.get("TRACE_TRAFFIC_LABEL", TRAFFIC_DEPLOYMENT)
DB_LOCK = threading.Lock()
# The backend relays reach a predictor Service that resolves to its pod IPs, so the
# address a connection lands on says which replica answered. Capture it per thread.
_PEER = threading.local()


class _PeerConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        super().connect()
        try:
            _PEER.ip = self.sock.getpeername()[0]
        except OSError:
            _PEER.ip = None


class _PeerHandler(HTTPHandler):
    def http_open(self, req: Request):
        return self.do_open(_PeerConnection, req)


OPENER = build_opener(_PeerHandler)
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def db() -> sqlite3.Connection:
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.execute("""CREATE TABLE IF NOT EXISTS traces (
        id TEXT PRIMARY KEY, started_at REAL NOT NULL, completed_at REAL,
        method TEXT NOT NULL, path TEXT NOT NULL, request_id TEXT NOT NULL,
        status INTEGER, classification TEXT, route TEXT, fallback INTEGER NOT NULL,
        duration_ms INTEGER, response_headers TEXT NOT NULL DEFAULT '{}', error TEXT,
        question TEXT, response TEXT, response_tokens INTEGER, response_tokens_est INTEGER
    )""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(traces)")}
    if "question" not in columns:
        connection.execute("ALTER TABLE traces ADD COLUMN question TEXT")
    for column, kind in (("response", "TEXT"), ("response_tokens", "INTEGER"), ("response_tokens_est", "INTEGER")):
        if column not in columns:
            connection.execute(f"ALTER TABLE traces ADD COLUMN {column} {kind}")
    connection.execute("""CREATE TABLE IF NOT EXISTS health_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, route TEXT NOT NULL, checked_at REAL NOT NULL,
        status INTEGER, latency_ms INTEGER, ok INTEGER NOT NULL, error TEXT
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS health_checks_route ON health_checks (route, id)")
    hop_columns = {row[1] for row in connection.execute("PRAGMA table_info(hops)")}
    if hop_columns and "trace_id" not in hop_columns:
        # Earlier builds keyed hops by request_id, which a failover hop does not carry.
        # The rows are derived timing data and cannot be joined to traces, so start over.
        connection.execute("DROP TABLE hops")
    connection.execute("""CREATE TABLE IF NOT EXISTS hops (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL, route TEXT NOT NULL,
        arrived_at REAL NOT NULL, ttfb_ms INTEGER, total_ms INTEGER, status INTEGER,
        classifier_us INTEGER, error TEXT, replica TEXT, peer_ip TEXT
    )""")
    hop_columns = {row[1] for row in connection.execute("PRAGMA table_info(hops)")}
    for column in ("replica", "peer_ip"):
        if column not in hop_columns:
            connection.execute(f"ALTER TABLE hops ADD COLUMN {column} TEXT")
    connection.execute("CREATE INDEX IF NOT EXISTS hops_trace ON hops (trace_id)")
    connection.execute("CREATE TABLE IF NOT EXISTS arch_state (ts REAL PRIMARY KEY, state TEXT NOT NULL)")
    return connection


def extract_response(path: str, headers: dict[str, str], raw: bytes) -> tuple[str | None, int | None, bool]:
    """Pull (answer text, completion tokens, tokens_are_estimated) out of a captured chat completion.

    The token count is the model's own usage.completion_tokens when the response carries
    it (always for JSON; for SSE only with stream_options.include_usage). Otherwise it is
    estimated at roughly four characters per token.
    """
    none = (None, None, False)
    if path.split("?", 1)[0] != "/v1/chat/completions" or not raw:
        return none
    lowered = {key.lower(): value for key, value in headers.items()}
    if lowered.get("content-encoding", "").lower() == "gzip":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            return none
    text = raw.decode("utf-8", errors="replace")
    answer: str | None = None
    tokens: int | None = None
    try:
        body = json.loads(text)
        message = body["choices"][0]["message"]
        answer = message.get("content")
        if not answer and message.get("tool_calls"):
            answer = json.dumps(message["tool_calls"])
        tokens = (body.get("usage") or {}).get("completion_tokens")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        parts = []
        for line in text.splitlines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("usage"), dict) and chunk["usage"].get("completion_tokens") is not None:
                tokens = chunk["usage"]["completion_tokens"]
            try:
                delta = chunk["choices"][0].get("delta", {})
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
        answer = "".join(parts)
    if not isinstance(answer, str) or not answer.strip():
        return none
    answer = answer.strip()
    estimated = not isinstance(tokens, int)
    if estimated:
        tokens = max(1, round(len(answer) / 4))
    stored = answer if len(answer) <= RESPONSE_MAX_CHARS else answer[:RESPONSE_MAX_CHARS] + "…"
    return stored, tokens, estimated


def extract_question(path: str, body: bytes) -> str | None:
    if path.split("?", 1)[0] != "/v1/chat/completions" or not body:
        return None
    try:
        messages = json.loads(body).get("messages", [])
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n\n".join(parts) or None
    return None


def log_context_probe(path: str, body: bytes, request_id: str, stage: str) -> None:
    if not CONTEXT_PROBE_MARKER or path.split("?", 1)[0] != "/v1/chat/completions":
        return
    try:
        messages = json.loads(body).get("messages", [])
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return
    if not isinstance(messages, list):
        return
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    marker_present = any(
        CONTEXT_PROBE_MARKER in json.dumps(message.get("content", ""))
        for message in messages if isinstance(message, dict)
    )
    print(json.dumps({"event": "context_probe", "request_id": request_id,
                      "stage": stage, "message_count": len(messages),
                      "roles": roles, "marker_present": marker_present}), flush=True)


def create_trace(trace_id: str, method: str, path: str, request_id: str, question: str | None) -> None:
    with DB_LOCK, db() as connection:
        connection.execute(
            "INSERT INTO traces (id, started_at, method, path, request_id, fallback, question) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (trace_id, time.time(), method, path, request_id, question),
        )


def record_hop(request_id: str, route: str, classification: str | None) -> None:
    """Remember the first relay hop a request reaches: its destination and label.

    The destination is known the moment Praxis picks a cluster, so the UI can show
    it while the request is still in flight. Praxis starts a failover step with
    empty headers, so the hop that fails (for example a stopped CPU predictor) is
    the only one that carries the original label. The first value wins;
    complete_trace() falls back to them and replaces the route with the final one.
    """
    with DB_LOCK, db() as connection:
        connection.execute(
            "UPDATE traces SET classification=COALESCE(classification, ?), route=COALESCE(route, ?) "
            "WHERE request_id=? AND completed_at IS NULL",
            (classification, route, request_id),
        )


def replica_for(route: str, ip: str | None) -> str | None:
    """Name of the pod of this route's pool that owns ip, from the latest polled state."""
    if not ip:
        return None
    with STATE_LOCK:
        state = LIVE_STATE["state"] or {}
    for pod in state.get("pools", {}).get(route, {}).get("pods", []):
        if pod.get("ip") == ip:
            return pod["name"]
    return None


def record_hop_timing(trace_id: str, route: str, arrived_at: float, started: float, ttfb_ms: int | None,
                      status: int | None, classifier_us: str | None, error: str | None = None,
                      peer_ip: str | None = None) -> None:
    """Store how long one backend hop took: time to response headers and total time.

    A request that fails over has one row per backend it tried, including the
    failed one. trace_id comes from the _trace query parameter the ingress relay
    adds: Praxis drops headers when it fails over but keeps the URI. classifier_us is the llm_d_sc filter's own latency; Praxis only
    sets it on the first hop, because a failover step starts with empty headers.
    """
    try:
        classifier = int(classifier_us) if classifier_us else None
    except ValueError:
        classifier = None
    with DB_LOCK, db() as connection:
        connection.execute(
            "INSERT INTO hops (trace_id, route, arrived_at, ttfb_ms, total_ms, status, classifier_us, error, "
            "replica, peer_ip) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trace_id, route, arrived_at, ttfb_ms, round((time.monotonic() - started) * 1000),
             status, classifier, error, replica_for(route, peer_ip), peer_ip),
        )


def complete_trace(trace_id: str, status: int, headers: dict[str, str], started: float,
                   error: str | None = None,
                   response: tuple[str | None, int | None, bool] = (None, None, False)) -> None:
    classification, route, fallback = routing_signal(headers)
    response_text, response_tokens, tokens_estimated = response
    with DB_LOCK, db() as connection:
        if classification is None or route is None:
            row = connection.execute("SELECT classification, route FROM traces WHERE id=?", (trace_id,)).fetchone()
            if row:
                classification = classification or row[0]
                route = route or row[1]
        if classification and route:
            # The label's policy cluster differs from where the request actually went.
            if (classification.upper() in GPU_LABELS) != ("gpu" in route.lower()):
                fallback = True
        connection.execute("""UPDATE traces SET completed_at=?, status=?, classification=?, route=?, fallback=?,
            duration_ms=?, response_headers=?, error=?, response=?, response_tokens=?, response_tokens_est=?
            WHERE id=?""", (
            time.time(), status, classification, route, fallback,
            round((time.monotonic() - started) * 1000), json.dumps(headers), error,
            response_text, response_tokens, int(tokens_estimated), trace_id,
        ))
        connection.execute("DELETE FROM traces WHERE id NOT IN (SELECT id FROM traces ORDER BY started_at DESC LIMIT ?)", (MAX_TRACES,))
        connection.execute("DELETE FROM hops WHERE trace_id NOT IN (SELECT id FROM traces)")


def routing_signal(headers: dict[str, str]) -> tuple[str | None, str | None, bool]:
    """Accept current and future Praxis header names without coupling the relay to one build."""
    lowered = {key.lower(): value for key, value in headers.items()}
    classification = (lowered.get("x-demo-classification") or lowered.get("x-llm-d-sc-label")
                      or next((value for key, value in lowered.items()
                               if "classif" in key or key.endswith("-label")), None))
    route = (lowered.get("x-demo-route") or lowered.get("x-praxis-cluster")
             or lowered.get("x-praxis-route")
             or next((value for key, value in lowered.items()
                      if "cluster" in key or "route" in key or "upstream" in key), None))
    fallback = any("fallback" in key and value.lower() in {"true", "1", "yes"}
                   for key, value in lowered.items())
    if route is None and classification:
        route = "vllm-gpu" if classification.upper() in {"COMPLEX", "REASONING"} else "vllm-cpu"
    wants_gpu = classification is not None and classification.upper() in {"COMPLEX", "REASONING"}
    if wants_gpu and route and "gpu" not in route.lower():
        fallback = True
    return classification, route, fallback


def record_health(route: str, status: int | None, latency_ms: int, error: str | None) -> None:
    """Store one Praxis health probe as seen on a relay (status None = no response)."""
    with DB_LOCK, db() as connection:
        connection.execute(
            "INSERT INTO health_checks (route, checked_at, status, latency_ms, ok, error) VALUES (?, ?, ?, ?, ?, ?)",
            (route, time.time(), status, latency_ms, int(status == 200), error),
        )
        connection.execute(
            "DELETE FROM health_checks WHERE route=? AND id NOT IN "
            "(SELECT id FROM health_checks WHERE route=? ORDER BY id DESC LIMIT ?)",
            (route, route, HEALTH_KEEP),
        )


def probe_state(route: str) -> bool | None:
    """Newest Praxis health probe seen on a relay: True/False, or None if there is no recent one."""
    with DB_LOCK, db() as connection:
        row = connection.execute(
            "SELECT checked_at, ok FROM health_checks WHERE route=? ORDER BY id DESC LIMIT 1", (route,)
        ).fetchone()
    if row is None or time.time() - row[0] > PROBE_FRESH_SECONDS:
        return None
    return bool(row[1])


def k8s_get(path: str) -> dict:
    token = (SA_DIR / "token").read_text().strip()
    context = ssl.create_default_context(cafile=str(SA_DIR / "ca.crt"))
    request = Request(K8S_API + path, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urlopen(request, timeout=4, context=context) as response:
        return json.load(response)


def k8s_patch(path: str, body: dict) -> dict:
    token = (SA_DIR / "token").read_text().strip()
    context = ssl.create_default_context(cafile=str(SA_DIR / "ca.crt"))
    request = Request(K8S_API + path, data=json.dumps(body).encode(), method="PATCH",
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                               "Content-Type": "application/merge-patch+json"})
    with urlopen(request, timeout=4, context=context) as response:
        return json.load(response)


def traffic_state(desired: int | None, pods: list[dict]) -> str:
    """stopped / initializing / ready / terminating, from the desired replicas and the pods."""
    if desired:
        return "ready" if any(p["ready"] and not p["terminating"] for p in pods) else "initializing"
    return "terminating" if pods else "stopped"


def read_traffic() -> dict:
    """State of the traffic generator, from the Kubernetes API."""
    try:
        namespace = (SA_DIR / "namespace").read_text().strip()
        spec = k8s_get(f"/apis/apps/v1/namespaces/{namespace}/deployments/{TRAFFIC_DEPLOYMENT}")
        items = k8s_get(f"/api/v1/namespaces/{namespace}/pods?labelSelector=app%3D{TRAFFIC_LABEL}")["items"]
    except (OSError, ValueError, KeyError) as failure:  # URLError and HTTPError are OSErrors
        return {"state": "unavailable", "error": str(failure)[:160]}
    pods = []
    for item in items:
        status = item.get("status", {})
        if status.get("phase") not in ("Pending", "Running"):
            continue
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in status.get("conditions", []))
        pods.append({"ready": ready, "terminating": bool(item["metadata"].get("deletionTimestamp"))})
    desired = spec.get("spec", {}).get("replicas") or 0
    return {"state": traffic_state(desired, pods), "desired": desired, "pods": len(pods)}


def set_traffic(replicas: int) -> None:
    namespace = (SA_DIR / "namespace").read_text().strip()
    k8s_patch(f"/apis/apps/v1/namespaces/{namespace}/deployments/{TRAFFIC_DEPLOYMENT}/scale",
              {"spec": {"replicas": replicas}})


def read_pools() -> dict:
    """Desired replicas and per-pod readiness for each predictor, from the Kubernetes API."""
    namespace = (SA_DIR / "namespace").read_text().strip()
    pools = {}
    for route, (deployment, isvc) in POOLS.items():
        spec = k8s_get(f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}")
        items = k8s_get(
            f"/api/v1/namespaces/{namespace}/pods?labelSelector=serving.kserve.io%2Finferenceservice%3D{isvc}"
        )["items"]
        pods = []
        for item in sorted(items, key=lambda entry: entry["metadata"]["name"]):
            status = item.get("status", {})
            if status.get("phase") not in ("Pending", "Running"):
                continue
            ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in status.get("conditions", []))
            pods.append({"name": item["metadata"]["name"], "ip": status.get("podIP"), "ready": ready,
                         "terminating": bool(item["metadata"].get("deletionTimestamp"))})
        pools[route] = {"desired": spec.get("spec", {}).get("replicas"), "pods": pods}
    return pools


def read_state() -> dict:
    """Snapshot of both pools. Without Kubernetes access it degrades to probe results only."""
    error = None
    try:
        pools = read_pools()
    except (OSError, ValueError, KeyError) as failure:  # URLError and HTTPError are OSErrors
        pools, error = {route: {} for route in POOLS}, str(failure)[:160]
    for route, pool in pools.items():
        pool["probe"] = probe_state(route)
    state: dict = {"k8s": error is None, "pools": pools}
    if error:
        state["error"] = error
    return state


STATE_LOCK = threading.Lock()
LIVE_STATE: dict = {"state": None, "at": None}


def state_poller() -> None:
    """Keep the live state fresh and record a row whenever it changes."""
    last = None
    while True:
        try:
            state = read_state()
            key = json.dumps(state, sort_keys=True)
            now = time.time()
            with STATE_LOCK:
                LIVE_STATE.update(state=state, at=now)
            if key != last:
                with DB_LOCK, db() as connection:
                    connection.execute("INSERT OR REPLACE INTO arch_state (ts, state) VALUES (?, ?)", (now, key))
                    connection.execute(
                        "DELETE FROM arch_state WHERE ts NOT IN (SELECT ts FROM arch_state ORDER BY ts DESC LIMIT ?)",
                        (STATE_KEEP,),
                    )
                last = key
        except Exception as error:  # keep polling whatever goes wrong
            print(json.dumps({"event": "state_poll_error", "error": str(error)[:200]}), flush=True)
        time.sleep(STATE_POLL_SECONDS)


def state_payload(path: str) -> dict:
    """Live state, or the state in effect at ?at=<epoch seconds>."""
    at = parse_qs(urlsplit(path).query).get("at", [None])[0]
    with DB_LOCK, db() as connection:
        first = connection.execute("SELECT MIN(ts) FROM arch_state").fetchone()[0]
        row = None
        if at is not None:
            try:
                row = connection.execute(
                    "SELECT state FROM arch_state WHERE ts <= ? ORDER BY ts DESC LIMIT 1", (float(at),)
                ).fetchone()
            except ValueError:
                pass
    if at is None:
        with STATE_LOCK:
            return {"now": time.time(), "state": LIVE_STATE["state"], "history_starts": first}
    return {"at": at, "state": json.loads(row[0]) if row else None, "history_starts": first}


def traces() -> list[dict]:
    with DB_LOCK, db() as connection:
        rows = connection.execute("""SELECT id, started_at, completed_at, method, path, request_id,
            status, classification, route, fallback, duration_ms, response_headers, error, question, response,
            response_tokens, response_tokens_est
            FROM traces ORDER BY started_at DESC LIMIT ?""", (MAX_TRACES,)).fetchall()
    ids = [row[0] for row in rows]
    hops: dict[str, list[dict]] = {}
    if ids:
        with DB_LOCK, db() as connection:
            hop_rows = connection.execute(
                f"SELECT trace_id, route, arrived_at, ttfb_ms, total_ms, status, classifier_us, error, replica "
                f"FROM hops WHERE trace_id IN ({','.join('?' * len(ids))}) ORDER BY arrived_at", ids).fetchall()
        started = {row[0]: row[1] for row in rows}
        for trace_key, route, arrived_at, ttfb, total, status, classifier, error, replica in hop_rows:
            hops.setdefault(trace_key, []).append({
                "route": route, "offset_ms": round((arrived_at - started[trace_key]) * 1000),
                "ttfb_ms": ttfb, "total_ms": total, "status": status,
                "classifier_ms": round(classifier / 1000) if classifier is not None else None, "error": error, "replica": replica})
    keys = ("id", "started_at", "completed_at", "method", "path", "request_id", "status", "classification", "route", "fallback", "duration_ms", "response_headers", "error", "question", "response", "response_tokens", "response_tokens_est")
    return [{**dict(zip(keys, row)), "response_headers": json.loads(row[11]), "hops": hops.get(row[0], [])} for row in rows]


def clear_traces() -> None:
    with DB_LOCK, db() as connection:
        connection.execute("DELETE FROM traces")
        connection.execute("DELETE FROM hops")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PraxisTraceRelay/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.server.route:
            self.proxy()
        elif self.path in {"/", "/index.html", "/_trace", "/_trace/", "/_trace/index.html"}:
            self.html(200, PAGE)
        elif self.path == "/_trace/healthz":
            self.json(200, {"status": "ok"})
        elif self.path == "/_trace/api/traces":
            self.json(200, {"traces": traces(), "upstream": UPSTREAM, "retention": MAX_TRACES})
        elif self.path.split("?", 1)[0] == "/_trace/api/state":
            self.json(200, state_payload(self.path))
        elif self.path == "/_trace/api/traffic":
            self.json(200, read_traffic())
        else:
            self.proxy()

    def do_POST(self) -> None:
        if self.server.route is None and self.path == "/_trace/api/traffic":
            self.traffic_action()
        else:
            self.proxy()

    def traffic_action(self) -> None:
        """Start or stop the traffic generator. Needs a JSON body, which a cross-site form cannot send."""
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self.json(415, {"error": "Content-Type must be application/json"})
            return
        try:
            action = json.loads(raw or b"{}").get("action")
        except (ValueError, AttributeError):
            action = None
        if action not in ("start", "stop"):
            self.json(400, {"error": "action must be start or stop"})
            return
        try:
            set_traffic(1 if action == "start" else 0)
        except (OSError, ValueError) as failure:
            self.json(502, {"state": "unavailable", "error": str(failure)[:160]})
            return
        self.json(200, read_traffic())

    def do_PUT(self) -> None: self.proxy()
    def do_PATCH(self) -> None: self.proxy()
    def do_DELETE(self) -> None:
        if self.path == "/_trace/api/traces":
            clear_traces()
            self.json(200, {"cleared": True})
        else:
            self.proxy()
    def do_OPTIONS(self) -> None: self.proxy()

    def json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def html(self, status: int, body: str) -> None:
        raw = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def proxy(self) -> None:
        route = self.server.route
        upstream = self.server.upstream
        trace_id, started = str(uuid.uuid4()), time.monotonic()
        request_id = self.headers.get("X-Request-Id", trace_id)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        log_context_probe(self.path, body, request_id, route or "before-praxis")
        if route is None:
            create_trace(trace_id, self.command, self.path, request_id, extract_question(self.path, body))
        is_probe = route is not None and self.command == "GET" and self.path.split("?", 1)[0] == HEALTH_PATH
        if route is not None and not is_probe:
            # Record before contacting the predictor: the destination is known now,
            # and if the predictor is down this hop is the only place the original
            # label is visible.
            record_hop(request_id, route, self.headers.get("X-Llm-D-Sc-Label"))
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in HOP_BY_HOP and not key.lower().startswith("x-praxis-")}
        headers["X-Request-Id"] = request_id
        timed = route is not None and not is_probe
        # The ingress relay tags the URL it sends to Praxis with the trace ID. Praxis
        # keeps the URI when it fails over (it drops the headers), so every backend
        # hop can be tied to its trace. The backend relay strips the tag again.
        path, hop_trace_id = self.path, trace_id
        if route is None and self.command == "POST":
            path += ("&" if "?" in path else "?") + "_trace=" + trace_id
        elif route is not None:
            parts = urlsplit(path)
            query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "_trace"]
            tagged = dict(parse_qsl(parts.query)).get("_trace")
            hop_trace_id = tagged or trace_id
            path = parts.path + ("?" + urlencode(query) if query else "")
        arrived_at = time.time()
        classifier_us = self.headers.get("X-Llm-D-Sc-Latency-Us")
        upstream_request = Request(upstream + path, data=body if body else None, headers=headers, method=self.command)
        response = None
        _PEER.ip = None
        try:
            response = OPENER.open(upstream_request, timeout=5 if is_probe else 600)
        except HTTPError as error:
            response = error
        except (URLError, TimeoutError, OSError) as error:
            if is_probe:
                record_health(route, None, round((time.monotonic() - started) * 1000), str(error))
            if timed:
                record_hop_timing(hop_trace_id, route, arrived_at, started, None, None, classifier_us, str(error))
            if route is None:
                complete_trace(trace_id, 502, {}, started, str(error))
            self.json(502, {"error": {"message": "Upstream unavailable", "type": "upstream_error"}})
            return
        ttfb_ms = round((time.monotonic() - started) * 1000)
        peer_ip = getattr(_PEER, "ip", None)
        response_headers = dict(response.headers.items())
        if is_probe:
            record_health(route, response.status, round((time.monotonic() - started) * 1000), None)
        if route is not None:
            response_headers["X-Demo-Route"] = route
            classification = self.headers.get("X-Llm-D-Sc-Label")
            if classification:
                response_headers["X-Demo-Classification"] = classification
        self.send_response(response.status)
        for key, value in response_headers.items():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        if route is None:
            self.send_header("X-Praxis-Trace-Id", trace_id)
        self.send_header("Connection", "close")
        self.end_headers()
        captured = bytearray()
        try:
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
                if route is None and len(captured) < RESPONSE_CAPTURE_BYTES:
                    captured += chunk[:RESPONSE_CAPTURE_BYTES - len(captured)]
            if route is None:
                complete_trace(trace_id, response.status, response_headers, started,
                               response=extract_response(self.path, response_headers, bytes(captured)))
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            if route is None:
                complete_trace(trace_id, response.status, response_headers, started, str(error),
                               response=extract_response(self.path, response_headers, bytes(captured)))
        finally:
            if timed:
                record_hop_timing(hop_trace_id, route, arrived_at, started, ttfb_ms, response.status, classifier_us,
                                  peer_ip=peer_ip)


class RelayServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], upstream: str, route: str | None = None):
        super().__init__(address, Handler)
        self.upstream = upstream
        self.route = route


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Praxis request traces</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--panel:#0d1b2d;--line:#2b4865;--text:#e9f2ff;--muted:#9db1c8;--blue:#64b5ff;--cyan:#55e6d0;--violet:#ad9cff;--amber:#ffc76a;--active:#ff4d5e}*{box-sizing:border-box}body{margin:0;min-height:100vh;padding:24px;color:var(--text);background:radial-gradient(circle at 15% 5%,#17385a 0,transparent 30%),radial-gradient(circle at 90% 95%,#173c3b 0,transparent 28%),var(--bg);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}main{width:min(1040px,100%);margin:auto}header,.section-heading,.drawer-header{display:flex;align-items:center;justify-content:space-between;gap:16px}header{margin-bottom:22px}h1{margin:0;font-size:25px}h2{margin:0}.section-heading{margin-bottom:12px}header p{margin:5px 0 0;color:var(--muted)}button{border:1px solid #365775;border-radius:10px;padding:7px 12px;color:var(--text);background:#172b41;font:inherit;cursor:pointer}button:hover{border-color:var(--blue);background:#1c3855}button:disabled{cursor:wait;opacity:.6}.live{color:var(--cyan);font-weight:700}.live:before{content:'●';margin-right:7px}.graphic,.traces{overflow:hidden;border:1px solid #23415d;border-radius:24px;background:linear-gradient(145deg,rgba(13,27,45,.97),rgba(7,17,31,.97));box-shadow:0 24px 80px #00000059}.canvas{padding:26px 30px 18px;overflow-x:auto}svg{display:block;width:100%;height:auto}.label{fill:var(--muted);font-size:12px;font-weight:700;letter-spacing:.07em}.small{fill:var(--muted);font-size:12px}.title{fill:var(--text);font-size:18px;font-weight:700}.box{fill:var(--panel);stroke:var(--line);stroke-width:1.5}.box-blue{fill:#102d49;stroke:var(--blue)}.box-cyan{fill:#103b3c;stroke:var(--cyan)}.box-violet{fill:#28234b;stroke:var(--violet)}.box-amber{fill:#45331a;stroke:var(--amber)}.pill{fill:#172b41;stroke:#365775;stroke-width:1}.pill-text{fill:#c5d7ea;font-size:11px;font-weight:600}.flow{fill:none;stroke:var(--blue);stroke-width:2.5;marker-end:url(#arrow-blue)}.flow-cyan{fill:none;stroke:var(--cyan);stroke-width:2.5;stroke-dasharray:7 6;marker-end:url(#arrow-cyan)}.flow.selected,.flow-cyan.selected{color:var(--active);stroke:var(--active);stroke-width:5;marker-end:url(#arrow-red);filter:drop-shadow(0 0 4px var(--active)) drop-shadow(0 0 9px var(--active))}.cluster{fill:#10243a80;stroke:#31516f;stroke-dasharray:6 6;stroke-width:1.5}.cluster-label{fill:var(--text);font-size:15px;font-weight:700}.packet-path{fill:none;stroke:none}.packet{display:none;filter:drop-shadow(0 0 6px currentColor);pointer-events:none}.legend{display:flex;justify-content:center;gap:28px;padding:16px 24px 20px;border-top:1px solid #1d354c;color:var(--muted);font-size:12px}.legend span{display:inline-flex;align-items:center;gap:8px}.swatch{width:24px;height:3px;border-radius:4px;background:var(--blue)}.swatch.cyan{background:var(--cyan)}.traces{margin-top:24px;padding:20px 24px}.traces small{color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:11px 8px;border-bottom:1px solid #203b55;vertical-align:top}th{color:var(--muted);font-size:11px;text-transform:uppercase}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected-row{background:#132b42}.tag{display:inline-block;padding:2px 7px;border-radius:999px;background:#18324d;font-size:12px;font-weight:700}.cpu{color:var(--violet)}.gpu{color:var(--amber)}.unknown{color:var(--muted)}.fellback{display:block;margin-top:3px;color:var(--active);font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}.defaulted{display:block;margin-top:3px;color:var(--cyan);font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}details summary{cursor:pointer;color:var(--blue)}.trace-detail{display:grid;gap:4px;margin-top:8px;padding:9px 10px;border-left:2px solid var(--blue);background:#0a1726;color:var(--muted);font-size:12px}.trace-detail b{color:var(--text)}.question-button{margin-left:7px;padding:2px 7px;border-radius:7px;color:var(--blue);font-size:12px}.no-question{margin-left:7px;color:var(--muted);font-size:11px}tbody td:first-child,.question-button{white-space:nowrap}.drawer-backdrop{position:fixed;inset:0;z-index:20;background:#02071199;opacity:0;pointer-events:none;transition:opacity .2s}.drawer-backdrop.open{opacity:1;pointer-events:auto}.drawer{position:fixed;top:0;right:0;z-index:21;width:min(480px,92vw);height:100vh;padding:26px;background:#0b1929;border-left:1px solid #31516f;box-shadow:-24px 0 70px #0009;transform:translateX(102%);transition:transform .24s ease;overflow:auto}.drawer.open{transform:translateX(0)}.drawer-header{padding-bottom:18px;border-bottom:1px solid #203b55}.drawer-header h2{font-size:21px}.drawer-close{width:36px;height:36px;padding:0;font-size:22px}.drawer-meta{margin:16px 0;color:var(--muted);font-size:12px}.question-text{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--text);font:16px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}@media(prefers-reduced-motion:reduce){.packet{display:none!important}.drawer,.drawer-backdrop{transition:none}}@media(max-width:700px){body{padding:10px}.canvas{padding:14px 8px 12px}svg{min-width:920px}.legend{justify-content:flex-start;flex-wrap:wrap}.traces{padding:16px 12px;overflow:auto}th:nth-child(3),td:nth-child(3){display:none}.drawer{width:100%;padding:20px}}
.box-grey{fill:#131c2a;stroke:#3a4b61;stroke-width:1.5}.dashed{stroke-dasharray:6 5}.pod-title{fill:var(--text);font-size:12px;font-weight:700}.pod-title.dim{fill:#8da0b8}.pod-sub{fill:var(--muted);font-size:11px}#state-note{fill:var(--amber);font-size:12px;font-weight:700}
.tl{display:flex;height:10px;border-radius:4px;overflow:hidden;margin:6px 0;background:#13243a}.tl i{min-width:2px}.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}.k-classify{background:var(--violet)}.k-praxis{background:var(--blue)}.k-backend{background:var(--cyan)}.k-fail{background:var(--active)}.k-gap{background:var(--amber)}.tl-line{display:block;color:var(--muted)}.tl-line b{color:var(--text)}
.actions{display:flex;align-items:center;gap:10px}.traffic-status{display:inline-flex;align-items:center;gap:7px;color:var(--muted);font-size:13px}.traffic-status i{width:9px;height:9px;border-radius:50%;background:#6b7c90}.traffic-status.ready i{background:#55e6a0}.traffic-status.initializing i,.traffic-status.terminating i{background:var(--amber)}.traffic-status.unavailable i{background:var(--active)}
</style>
</head>
<body><main>
<header><div><h1>Praxis request traces</h1><p>Live semantic routing with the latest user question saved alongside each recent trace.</p></div><span class="live">LIVE</span></header>
<section class="graphic"><div class="canvas"><svg viewBox="0 0 980 690" role="img" aria-labelledby="diagram-title diagram-desc">
<title id="diagram-title">Live Praxis request routing</title><desc id="diagram-desc">Open WebUI sends a request to Praxis. Praxis calls llm-d-sc for classification and routes the request to either the CPU or GPU vLLM service.</desc>
<defs><marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L10,3 L0,6 z" fill="#64b5ff"/></marker><marker id="arrow-cyan" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L10,3 L0,6 z" fill="#55e6d0"/></marker><marker id="arrow-red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L10,3 L0,6 z" fill="#ff4d5e"/></marker></defs>
<path id="packet-source" class="packet-path" d="M490 124 V180"/><path id="packet-classify" class="packet-path" d="M623 225 H720"/><path id="packet-result" class="packet-path" d="M720 256 H623"/><path id="packet-cpu" class="packet-path" d="M490 292 V350 H245 V598"/><path id="packet-gpu" class="packet-path" d="M490 292 V350 H735 V598"/><path id="packet-pod" class="packet-path" d="M0 0"/><circle id="request-dot" class="packet" r="7" fill="#64b5ff"/>

<text x="0" y="30" class="label">REQUEST</text><rect x="290" y="48" width="400" height="76" rx="18" class="box-blue"/><text x="330" y="84" class="title">Open WebUI / client</text><text x="330" y="105" class="small">OpenAI-compatible POST /v1/chat/completions</text><path id="flow-source" d="M490 124 V180" class="flow"/>

<text x="0" y="160" class="label">SEMANTIC ROUTING</text><rect x="357" y="180" width="266" height="112" rx="18" class="box-blue"/><text x="385" y="218" class="title">Praxis / AI Gateway</text><text x="385" y="242" class="small">Classifies, selects, and proxies</text><rect x="720" y="180" width="210" height="112" rx="18" class="box-cyan"/><text x="750" y="220" class="title">llm-d-sc</text><text x="750" y="243" class="small">Returns complexity label</text><rect x="750" y="257" width="70" height="21" rx="10" class="pill"/><text x="785" y="271" text-anchor="middle" class="pill-text">SIMPLE</text><rect x="828" y="257" width="70" height="21" rx="10" class="pill"/><text x="863" y="271" text-anchor="middle" class="pill-text">COMPLEX</text><path id="flow-classify" d="M623 225 H720" class="flow-cyan"/><path id="flow-result" d="M720 256 H623" class="flow-cyan"/>

<path id="flow-cpu" d="M490 292 V350 H245 V382" class="flow cpu-path"/><path id="flow-gpu" d="M490 292 V350 H735 V382" class="flow gpu-path"/><text x="0" y="370" class="label">INFERENCE ROUTING</text><rect x="90" y="386" width="310" height="106" rx="18" class="box-violet"/><text x="116" y="425" class="title">vLLM CPU service</text><text x="116" y="450" class="small">SIMPLE / MEDIUM requests</text><rect x="580" y="386" width="310" height="106" rx="18" class="box-amber"/><text x="606" y="425" class="title">vLLM GPU service</text><text x="606" y="450" class="small">COMPLEX / REASONING requests</text><path id="flow-cpu-pod" d="M245 492 V550" class="flow cpu-path"/><path id="flow-gpu-pod" d="M735 492 V550" class="flow gpu-path"/>

<text x="0" y="536" class="label">OPENSHIFT WORKLOADS</text><text id="state-note" x="490" y="12" text-anchor="middle"></text><rect x="55" y="550" width="405" height="120" rx="20" class="cluster"/><text x="82" y="582" class="cluster-label">CPU pool</text><text id="pool-cpu-count" x="440" y="582" text-anchor="end" class="small"></text><g id="arrows-cpu"></g><g id="pods-cpu"></g><rect x="520" y="550" width="405" height="120" rx="20" class="cluster"/><text x="547" y="582" class="cluster-label">GPU pool</text><text id="pool-gpu-count" x="905" y="582" text-anchor="end" class="small"></text><g id="arrows-gpu"></g><g id="pods-gpu"></g>
</svg></div><footer class="legend"><span><i class="swatch"></i> request / routing path</span><span><i class="swatch cyan"></i> classifier request + result</span></footer></section>

<section class="traces"><div class="section-heading"><h2>Recent requests <small id="count"></small></h2><div class="actions"><span id="traffic-status" class="traffic-status unknown"><i></i>Traffic: …</span><button id="traffic-toggle" type="button" disabled>Simulate Traffic</button><button id="clear-recents" type="button">Clear Recents</button></div></div><table><thead><tr><th>Time</th><th>Request</th><th>Classification</th><th>Destination</th><th>Status</th><th>Duration</th><th>Response</th></tr></thead><tbody id="traces"><tr><td colspan="6">Waiting for a request…</td></tr></tbody></table></section>
</main><div id="drawer-backdrop" class="drawer-backdrop"></div><aside id="question-drawer" class="drawer" aria-hidden="true" aria-labelledby="question-title"><div class="drawer-header"><h2 id="question-title">Question</h2><button id="drawer-close" class="drawer-close" type="button" aria-label="Close question">×</button></div><div id="question-meta" class="drawer-meta"></div><p id="question-text" class="question-text"></p></aside><script>
const $=id=>document.getElementById(id),dot=$('request-dot'),paths={source:$('packet-source'),classify:$('packet-classify'),result:$('packet-result'),cpu:$('packet-cpu'),gpu:$('packet-gpu'),pod:$('packet-pod')},flows={source:[$('flow-source')],classify:[$('flow-classify'),$('flow-result')],cpu:[$('flow-cpu'),$('flow-cpu-pod')],gpu:[$('flow-gpu'),$('flow-gpu-pod')]};let last='',lastAnimated='',selectedId='',dismissedThroughId='',traceItems=[],animating=false,animationGeneration=0,openTraces=new Set();
const questionDrawer=$('question-drawer'),drawerBackdrop=$('drawer-backdrop');
function esc(x){return String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function isGpu(t){return t.route?t.route.toLowerCase().includes('gpu'):['COMPLEX','REASONING'].includes((t.classification||'').toUpperCase())}
function timelineHtml(t){const hops=t.hops||[];if(!hops.length||!t.duration_ms)return'';const segs=[];let cursor=0;hops.forEach((h,i)=>{const gap=Math.max(0,h.offset_ms-cursor);if(i===0){const c=h.classifier_ms!=null?Math.min(h.classifier_ms,gap):0;if(c)segs.push({k:'classify',l:'classifier',ms:c});segs.push({k:'praxis',l:c?'Praxis':'Praxis + classifier',ms:gap-c})}else segs.push({k:'gap',l:'failover gap',ms:gap});const bad=h.status==null||h.status>=500||h.status===429;segs.push({k:bad?'fail':'backend',l:h.route+(bad?' (failed)':''),ms:h.total_ms||0,note:(h.ttfb_ms!=null?'first byte '+h.ttfb_ms+' ms':(h.error||'no response'))+(h.replica?' · replica '+h.replica.replace(/^vllm-(cpu|gpu)-predictor-/,''):'')});cursor=h.offset_ms+(h.total_ms||0)});segs.push({k:'praxis',l:'Praxis return',ms:Math.max(0,t.duration_ms-cursor)});return`<div class="tl">${segs.map(x=>`<i class="k-${x.k}" style="flex:${Math.max(x.ms,0)}" title="${esc(x.l)} ${x.ms} ms"></i>`).join('')}</div>${segs.map(x=>`<span class="tl-line"><i class="sw k-${x.k}"></i>${esc(x.l)} <b>${x.ms} ms</b>${x.note?' · '+esc(x.note):''}</span>`).join('')}`}
function pathText(t){if(!(t.route||t.classification))return'Open WebUI → Praxis (routing metadata unavailable)';return`Open WebUI → Praxis → llm-d-sc → ${isGpu(t)?'vLLM GPU service → vllm-gpu-predictor pod':'vLLM CPU service → vllm-cpu-predictor pod'}`}
let current=null;
function servedReplica(t){const h=t&&t.hops&&t.hops[t.hops.length-1];return h&&h.status&&h.status<500&&h.replica||null}
function replicaArrow(t){const r=servedReplica(t);return r?document.querySelector('.flow[data-pod="'+CSS.escape(r)+'"]'):null}
function highlightReplica(t){document.querySelectorAll('.flow[data-pod]').forEach(a=>a.classList.remove('selected'));const a=replicaArrow(t);if(a)a.classList.add('selected')}
function finalPath(t){const key=isGpu(t)?'gpu':'cpu',a=replicaArrow(t);if(!a)return paths[key];paths.pod.setAttribute('d',`M490 292 V350 H${POOLS[key].sx} V544 H${a.dataset.cx} V598`);return paths.pod}
function highlight(t){current=t;Object.values(flows).flat().forEach(path=>path.classList.remove('selected'));highlightReplica(t);if(!t)return;flows.source.forEach(path=>path.classList.add('selected'));if(t.route||t.classification){flows.classify.forEach(path=>path.classList.add('selected'));flows[isGpu(t)?'gpu':'cpu'].forEach(path=>path.classList.add('selected'))}}
async function travel(path,color,duration){const length=path.getTotalLength(),started=performance.now();dot.style.display='block';dot.setAttribute('fill',color);dot.style.color=color;return new Promise(resolve=>{function frame(now){const progress=Math.min((now-started)/duration,1),point=path.getPointAtLength(length*progress);dot.setAttribute('cx',point.x);dot.setAttribute('cy',point.y);if(progress<1)requestAnimationFrame(frame);else resolve()}requestAnimationFrame(frame)})}
async function animate(t){if(!t||animating||matchMedia('(prefers-reduced-motion: reduce)').matches)return;const generation=animationGeneration;animating=true;await travel(paths.source,'#64b5ff',450);if(generation!==animationGeneration){animating=false;return}if(t.route||t.classification){await travel(paths.classify,'#55e6d0',450);if(generation!==animationGeneration){animating=false;return}await travel(paths.result,'#55e6d0',450);if(generation!==animationGeneration){animating=false;return}await travel(finalPath(t),isGpu(t)?'#ffc76a':'#ad9cff',900)}await sleep(300);dot.style.display='none';animating=false}
function render(){const items=traceItems;$('count').textContent=`(${items.length} saved)`;$('traces').innerHTML=items.length?items.map(t=>`<tr data-trace-id="${esc(t.id)}" tabindex="0" class="${t.id===selectedId?'selected-row':''}"><td>${new Date(t.started_at*1000).toLocaleTimeString()}</td><td><span class="tag">${esc(t.method)}</span> ${esc(t.path)}${t.question?`<button class="question-button" type="button" data-question-id="${esc(t.id)}">View question</button>`:'<span class="no-question">question unavailable</span>'}<details${openTraces.has(t.id)?' data-r="1" open':''}><summary>trace ${esc(t.request_id.slice(0,8))}</summary><div class="trace-detail"><span><b>Path:</b> ${esc(pathText(t))}</span><span><b>Request ID:</b> ${esc(t.request_id)}</span><span><b>Classification:</b> ${esc(t.classification||'label not emitted')}</span><span><b>Destination:</b> ${esc(t.route||'not emitted')}</span><span><b>Result:</b> ${esc(t.status||'in flight')} · ${t.duration_ms?esc(t.duration_ms+' ms'):'in progress'}${t.fallback?' · fallback':''}</span>${timelineHtml(t)}${t.error?`<span><b>Error:</b> ${esc(t.error)}</span>`:''}</div></details></td><td>${esc(t.classification||'not emitted')}</td><td class="${t.route?(isGpu(t)?'gpu':'cpu'):'unknown'}">${t.route?esc(t.route):(t.status?'unavailable':'routing…')}${t.fallback?'<small class="fellback">fell back</small>':t.route&&!t.classification&&t.status&&!t.error?'<small class="defaulted">no label &middot; default</small>':''}</td><td>${esc(t.status||'in flight')}</td><td>${t.duration_ms?esc(t.duration_ms+' ms'):'—'}</td><td>${t.response?`<button class="question-button" type="button" data-response-id="${esc(t.id)}">${t.response_tokens?(t.response_tokens_est?'~':'')+esc(t.response_tokens)+' tokens':'View'}</button>`:'<span class="no-question">—</span>'}</td></tr>`).join(''):'<tr><td colspan="7">Waiting for a request…</td></tr>'}
function syncSelection(){document.querySelectorAll('tr[data-trace-id]').forEach(row=>row.classList.toggle('selected-row',row.dataset.traceId===selectedId))}
function clearSelection(){selectedId='';dismissedThroughId=(traceItems.find(t=>t.status&&(t.route||t.classification))||{}).id||'';animationGeneration++;dot.style.display='none';highlight(null);syncSelection();unpinState()}
function selectTrace(id){if(selectedId===id){clearSelection();return}const trace=traceItems.find(item=>item.id===id);if(!trace)return;selectedId=id;dismissedThroughId='';syncSelection();highlight(trace);pinState(trace);animate(trace)}
function openQuestion(id,field='question',title='Question'){const trace=traceItems.find(item=>item.id===id);if(!trace||!trace[field])return;selectedId=id;dismissedThroughId='';syncSelection();highlight(trace);pinState(trace);$('question-title').textContent=title;$('question-meta').textContent=`${new Date(trace.started_at*1000).toLocaleString()} · trace ${trace.request_id.slice(0,8)}`;$('question-text').textContent=trace[field];questionDrawer.classList.add('open');drawerBackdrop.classList.add('open');questionDrawer.setAttribute('aria-hidden','false');$('drawer-close').focus()}
function closeQuestion(){questionDrawer.classList.remove('open');drawerBackdrop.classList.remove('open');questionDrawer.setAttribute('aria-hidden','true')}
async function load(){try{const data=await fetch('/_trace/api/traces',{cache:'no-store'}).then(r=>r.json()),items=data.traces;traceItems=items;const key=items.map(x=>x.id+':'+x.status+':'+(x.classification||'')+':'+(x.response?1:0)).join();if(key!==last){last=key;render()}if(selectedId){const selected=items.find(t=>t.id===selectedId);if(selected)highlight(selected);else clearSelection()}else{const routed=items.find(t=>t.status&&(t.route||t.classification));if(routed&&routed.id!==dismissedThroughId){highlight(routed);if(routed.id!==lastAnimated){lastAnimated=routed.id;animate(routed)}}}}catch(e){$('traces').innerHTML='<tr><td colspan="7">Trace API unavailable.</td></tr>'}}
$('traces').addEventListener('click',event=>{const question=event.target.closest('[data-question-id]');if(question){openQuestion(question.dataset.questionId);return}const answer=event.target.closest('[data-response-id]');if(answer){openQuestion(answer.dataset.responseId,'response','Response');return}if(event.target.closest('details'))return;const row=event.target.closest('tr[data-trace-id]');if(row)selectTrace(row.dataset.traceId)});
$('traces').addEventListener('toggle',event=>{const d=event.target;if(d.tagName!=='DETAILS')return;const row=d.closest('tr[data-trace-id]');if(!row)return;if(d.open)openTraces.add(row.dataset.traceId);else openTraces.delete(row.dataset.traceId);if(d.dataset.r){delete d.dataset.r;return}if(d.open&&row.dataset.traceId!==selectedId)selectTrace(row.dataset.traceId)},true);
$('traces').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){const row=event.target.closest('tr[data-trace-id]');if(row){event.preventDefault();selectTrace(row.dataset.traceId)}}});
document.addEventListener('click',event=>{if(!selectedId)return;if(event.target.closest('tr[data-trace-id]'))return;if(questionDrawer.classList.contains('open')&&event.target.closest('#question-drawer'))return;clearSelection()});
$('drawer-close').addEventListener('click',closeQuestion);drawerBackdrop.addEventListener('click',closeQuestion);document.addEventListener('keydown',event=>{if(event.key==='Escape'&&questionDrawer.classList.contains('open'))closeQuestion()});
$('clear-recents').addEventListener('click',async()=>{const button=$('clear-recents');button.disabled=true;try{const response=await fetch('/_trace/api/traces',{method:'DELETE'});if(!response.ok)throw new Error();traceItems=[];selectedId='';dismissedThroughId='';last='';lastAnimated='';animationGeneration++;dot.style.display='none';highlight(null);closeQuestion();render()}finally{button.disabled=false}});
let traffic={state:'unknown'};
function renderTraffic(){const s=traffic.state,names={stopped:'Stopped',initializing:'Initializing',ready:'Ready',terminating:'Terminating',unavailable:'Unavailable'},el=$('traffic-status'),b=$('traffic-toggle'),running=s==='initializing'||s==='ready';el.className='traffic-status '+s;el.title=traffic.error||'';el.innerHTML='<i></i>Traffic: '+esc(names[s]||'…');b.textContent=running?'Stop Traffic':'Simulate Traffic';b.disabled=!(s==='stopped'||running)}
async function loadTraffic(){try{traffic=await fetch('/_trace/api/traffic',{cache:'no-store'}).then(r=>r.json())}catch(e){traffic={state:'unavailable',error:'trace API unavailable'}}renderTraffic()}
$('traffic-toggle').addEventListener('click',async()=>{const b=$('traffic-toggle'),running=traffic.state==='initializing'||traffic.state==='ready';b.disabled=true;try{traffic=await fetch('/_trace/api/traffic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:running?'stop':'start'})}).then(r=>r.json())}catch(e){traffic={state:'unavailable',error:'trace API unavailable'}}renderTraffic()});
const POOLS={cpu:{route:'vllm-cpu',sx:245,x:80,cls:'box-violet',name:'vllm-cpu-predictor'},gpu:{route:'vllm-gpu',sx:735,x:545,cls:'box-amber',name:'vllm-gpu-predictor'}};
let liveState=null,pinned=false,pinSeq=0;
function podList(cfg,state){const pool=state&&state.pools?state.pools[cfg.route]:null;
if(!pool)return{pods:[{kind:'unknown',title:cfg.name,sub:'state unknown'}],count:''};
const probe=pool.probe;
if(!state.k8s){const kind=probe===false?'down':probe===true?'up':'unknown';return{pods:[{kind,title:cfg.name,sub:probe===false?'health check failing':probe===true?'healthy':'status unknown'}],count:''}}
const list=pool.pods||[],desired=pool.desired==null?list.length:pool.desired,n=Math.max(list.length,desired),pods=[];let ready=0;
if(n===0)pods.push({kind:'zero',title:cfg.name,sub:'scaled to 0 replicas'});
for(let i=0;i<n;i++){const p=list[i];
if(!p){pods.push({kind:'pending',title:cfg.name,sub:'starting…'});continue}
const suffix=p.name.startsWith(cfg.name+'-')?p.name.slice(cfg.name.length+1):p.name,up=p.ready&&!p.terminating&&probe!==false;if(up)ready++;
const sub=p.terminating?'terminating':!p.ready?'not ready':probe===false?'health check failing':'healthy';
pods.push({name:p.name,kind:up?'up':(p.terminating?'pending':'down'),title:cfg.name,sub:suffix+' · '+sub})}
return{pods,count:ready+'/'+desired+' healthy'}}
function renderPool(key,state){const cfg=POOLS[key],{pods,count}=podList(cfg,state),W=355,gap=10,shown=pods.slice(0,3),more=pods.length-shown.length,w=(W-gap*(shown.length-1))/shown.length;
$('pool-'+key+'-count').textContent=count;
const multi=shown.filter(p=>p.name&&p.kind==='up').length>1;$('flow-'+key+'-pod').style.display=multi?'none':'';
$('arrows-'+key).innerHTML=multi?shown.map((p,i)=>{if(!(p.name&&p.kind==='up'))return'';const cx=cfg.x+i*(w+gap)+w/2,ax=Math.min(Math.max(cx,cfg.x+85),cfg.x+270);return`<path class="flow ${key}-path" data-pod="${esc(p.name)}" data-cx="${ax}" d="M${cfg.sx} 492 V544 H${ax} V598"/>`}).join(''):'';
$('pods-'+key).innerHTML=shown.map((p,i)=>{const x=cfg.x+i*(w+gap),up=p.kind==='up',dot=up?'#55e6a0':p.kind==='down'?'#ff4d5e':p.kind==='pending'?'#ffc76a':'#6b7c90',dashed=['zero','pending','unknown'].includes(p.kind)?' dashed':'';
return`<g><rect x="${x}" y="598" width="${w}" height="50" rx="12" class="${up?cfg.cls:'box-grey'}${dashed}"/><circle cx="${x+16}" cy="623" r="5" fill="${dot}"/><text x="${x+30}" y="619" class="pod-title${up?'':' dim'}">${w>=170?esc(p.title):'pod'}</text><text x="${x+30}" y="637" class="pod-sub">${esc(p.sub)}</text></g>`}).join('')+(more>0?`<text x="${cfg.x+W}" y="662" text-anchor="end" class="small">+${more} more</text>`:'')}
function renderState(state,note){renderPool('cpu',state);renderPool('gpu',state);$('state-note').textContent=note||'';highlightReplica(current)}
async function loadState(){try{const d=await fetch('/_trace/api/state',{cache:'no-store'}).then(r=>r.json());liveState=d.state;if(!pinned)renderState(d.state,'')}catch(e){}}
async function pinState(t){const seq=++pinSeq;try{const d=await fetch('/_trace/api/state?at='+t.started_at,{cache:'no-store'}).then(r=>r.json());if(seq!==pinSeq)return;pinned=true;const at=ts=>new Date(ts*1000).toLocaleTimeString();renderState(d.state,d.state?'cluster state when sent · '+at(t.started_at):'no state recorded for '+at(t.started_at)+' · history starts '+(d.history_starts?at(d.history_starts):'—'))}catch(e){}}
function unpinState(){pinSeq++;pinned=false;renderState(liveState,'')}
load();setInterval(load,2000);loadTraffic();setInterval(loadTraffic,2000);loadState();setInterval(loadState,2000)
</script></body></html>'''


if __name__ == "__main__":
    with db():
        pass
    threading.Thread(target=state_poller, daemon=True).start()
    for port, upstream, route in (
        (8081, CPU_UPSTREAM, "vllm-cpu"),
        (8082, GPU_UPSTREAM, "vllm-gpu"),
    ):
        threading.Thread(
            target=RelayServer(("0.0.0.0", port), upstream, route).serve_forever,
            daemon=True,
        ).start()
    RelayServer(("0.0.0.0", 8080), UPSTREAM).serve_forever()
