#!/usr/bin/env python3
"""A dependency-free OpenAI relay with a small persisted request trace view."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATABASE = Path(os.environ.get("TRACE_DB_PATH", "/data/traces.db"))
UPSTREAM = os.environ.get("PRAXIS_UPSTREAM", "http://praxis:8080").rstrip("/")
CPU_UPSTREAM = os.environ.get("CPU_UPSTREAM", "http://vllm-cpu-predictor:8080").rstrip("/")
GPU_UPSTREAM = os.environ.get("GPU_UPSTREAM", "http://vllm-gpu-predictor:8080").rstrip("/")
CONTEXT_PROBE_MARKER = os.environ.get("TRACE_CONTEXT_PROBE_MARKER", "")
MAX_TRACES = int(os.environ.get("TRACE_MAX_TRACES", "250"))
DB_LOCK = threading.Lock()
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def db() -> sqlite3.Connection:
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.execute("""CREATE TABLE IF NOT EXISTS traces (
        id TEXT PRIMARY KEY, started_at REAL NOT NULL, completed_at REAL,
        method TEXT NOT NULL, path TEXT NOT NULL, request_id TEXT NOT NULL,
        status INTEGER, classification TEXT, route TEXT, fallback INTEGER NOT NULL,
        duration_ms INTEGER, response_headers TEXT NOT NULL DEFAULT '{}', error TEXT,
        question TEXT
    )""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(traces)")}
    if "question" not in columns:
        connection.execute("ALTER TABLE traces ADD COLUMN question TEXT")
    return connection


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


def complete_trace(trace_id: str, status: int, headers: dict[str, str], started: float, error: str | None = None) -> None:
    classification, route, fallback = routing_signal(headers)
    with DB_LOCK, db() as connection:
        connection.execute("""UPDATE traces SET completed_at=?, status=?, classification=?, route=?, fallback=?,
            duration_ms=?, response_headers=?, error=? WHERE id=?""", (
            time.time(), status, classification, route, fallback,
            round((time.monotonic() - started) * 1000), json.dumps(headers), error, trace_id,
        ))
        connection.execute("DELETE FROM traces WHERE id NOT IN (SELECT id FROM traces ORDER BY started_at DESC LIMIT ?)", (MAX_TRACES,))


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


def traces() -> list[dict]:
    with DB_LOCK, db() as connection:
        rows = connection.execute("""SELECT id, started_at, completed_at, method, path, request_id,
            status, classification, route, fallback, duration_ms, response_headers, error, question
            FROM traces ORDER BY started_at DESC LIMIT ?""", (MAX_TRACES,)).fetchall()
    keys = ("id", "started_at", "completed_at", "method", "path", "request_id", "status", "classification", "route", "fallback", "duration_ms", "response_headers", "error", "question")
    return [{**dict(zip(keys, row)), "response_headers": json.loads(row[11])} for row in rows]


def clear_traces() -> None:
    with DB_LOCK, db() as connection:
        connection.execute("DELETE FROM traces")


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
        else:
            self.proxy()

    def do_POST(self) -> None: self.proxy()
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
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in HOP_BY_HOP and not key.lower().startswith("x-praxis-")}
        headers["X-Request-Id"] = request_id
        upstream_request = Request(upstream + self.path, data=body if body else None, headers=headers, method=self.command)
        response = None
        try:
            response = urlopen(upstream_request, timeout=600)
        except HTTPError as error:
            response = error
        except (URLError, TimeoutError, OSError) as error:
            if route is None:
                complete_trace(trace_id, 502, {}, started, str(error))
            self.json(502, {"error": {"message": "Upstream unavailable", "type": "upstream_error"}})
            return
        response_headers = dict(response.headers.items())
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
        try:
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
            if route is None:
                complete_trace(trace_id, response.status, response_headers, started)
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            if route is None:
                complete_trace(trace_id, response.status, response_headers, started, str(error))


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
:root{color-scheme:dark;--bg:#07111f;--panel:#0d1b2d;--line:#2b4865;--text:#e9f2ff;--muted:#9db1c8;--blue:#64b5ff;--cyan:#55e6d0;--violet:#ad9cff;--amber:#ffc76a;--active:#ff4d5e}*{box-sizing:border-box}body{margin:0;min-height:100vh;padding:24px;color:var(--text);background:radial-gradient(circle at 15% 5%,#17385a 0,transparent 30%),radial-gradient(circle at 90% 95%,#173c3b 0,transparent 28%),var(--bg);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}main{width:min(1040px,100%);margin:auto}header,.section-heading,.drawer-header{display:flex;align-items:center;justify-content:space-between;gap:16px}header{margin-bottom:22px}h1{margin:0;font-size:25px}h2{margin:0}.section-heading{margin-bottom:12px}header p{margin:5px 0 0;color:var(--muted)}button{border:1px solid #365775;border-radius:10px;padding:7px 12px;color:var(--text);background:#172b41;font:inherit;cursor:pointer}button:hover{border-color:var(--blue);background:#1c3855}button:disabled{cursor:wait;opacity:.6}.live{color:var(--cyan);font-weight:700}.live:before{content:'●';margin-right:7px}.graphic,.traces{overflow:hidden;border:1px solid #23415d;border-radius:24px;background:linear-gradient(145deg,rgba(13,27,45,.97),rgba(7,17,31,.97));box-shadow:0 24px 80px #00000059}.canvas{padding:26px 30px 18px;overflow-x:auto}svg{display:block;width:100%;height:auto}.label{fill:var(--muted);font-size:12px;font-weight:700;letter-spacing:.07em}.small{fill:var(--muted);font-size:12px}.title{fill:var(--text);font-size:18px;font-weight:700}.box{fill:var(--panel);stroke:var(--line);stroke-width:1.5}.box-blue{fill:#102d49;stroke:var(--blue)}.box-cyan{fill:#103b3c;stroke:var(--cyan)}.box-violet{fill:#28234b;stroke:var(--violet)}.box-amber{fill:#45331a;stroke:var(--amber)}.pill{fill:#172b41;stroke:#365775;stroke-width:1}.pill-text{fill:#c5d7ea;font-size:11px;font-weight:600}.flow{fill:none;stroke:var(--blue);stroke-width:2.5;marker-end:url(#arrow-blue)}.flow-cyan{fill:none;stroke:var(--cyan);stroke-width:2.5;stroke-dasharray:7 6;marker-end:url(#arrow-cyan)}.flow.selected,.flow-cyan.selected{color:var(--active);stroke:var(--active);stroke-width:5;marker-end:url(#arrow-red);filter:drop-shadow(0 0 4px var(--active)) drop-shadow(0 0 9px var(--active))}.cluster{fill:#10243a80;stroke:#31516f;stroke-dasharray:6 6;stroke-width:1.5}.cluster-label{fill:var(--text);font-size:15px;font-weight:700}.packet-path{fill:none;stroke:none}.packet{display:none;filter:drop-shadow(0 0 6px currentColor);pointer-events:none}.legend{display:flex;justify-content:center;gap:28px;padding:16px 24px 20px;border-top:1px solid #1d354c;color:var(--muted);font-size:12px}.legend span{display:inline-flex;align-items:center;gap:8px}.swatch{width:24px;height:3px;border-radius:4px;background:var(--blue)}.swatch.cyan{background:var(--cyan)}.traces{margin-top:24px;padding:20px 24px}.traces small{color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:11px 8px;border-bottom:1px solid #203b55;vertical-align:top}th{color:var(--muted);font-size:11px;text-transform:uppercase}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected-row{background:#132b42}.tag{display:inline-block;padding:2px 7px;border-radius:999px;background:#18324d;font-size:12px;font-weight:700}.cpu{color:var(--violet)}.gpu{color:var(--amber)}.unknown{color:var(--muted)}.fellback{display:block;margin-top:3px;color:var(--active);font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}.defaulted{display:block;margin-top:3px;color:var(--cyan);font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}details summary{cursor:pointer;color:var(--blue)}.trace-detail{display:grid;gap:4px;margin-top:8px;padding:9px 10px;border-left:2px solid var(--blue);background:#0a1726;color:var(--muted);font-size:12px}.trace-detail b{color:var(--text)}.question-button{margin-left:7px;padding:2px 7px;border-radius:7px;color:var(--blue);font-size:12px}.no-question{margin-left:7px;color:var(--muted);font-size:11px}.drawer-backdrop{position:fixed;inset:0;z-index:20;background:#02071199;opacity:0;pointer-events:none;transition:opacity .2s}.drawer-backdrop.open{opacity:1;pointer-events:auto}.drawer{position:fixed;top:0;right:0;z-index:21;width:min(480px,92vw);height:100vh;padding:26px;background:#0b1929;border-left:1px solid #31516f;box-shadow:-24px 0 70px #0009;transform:translateX(102%);transition:transform .24s ease;overflow:auto}.drawer.open{transform:translateX(0)}.drawer-header{padding-bottom:18px;border-bottom:1px solid #203b55}.drawer-header h2{font-size:21px}.drawer-close{width:36px;height:36px;padding:0;font-size:22px}.drawer-meta{margin:16px 0;color:var(--muted);font-size:12px}.question-text{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--text);font:16px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}@media(prefers-reduced-motion:reduce){.packet{display:none!important}.drawer,.drawer-backdrop{transition:none}}@media(max-width:700px){body{padding:10px}.canvas{padding:14px 8px 12px}svg{min-width:920px}.legend{justify-content:flex-start;flex-wrap:wrap}.traces{padding:16px 12px;overflow:auto}th:nth-child(3),td:nth-child(3){display:none}.drawer{width:100%;padding:20px}}
</style>
</head>
<body><main>
<header><div><h1>Praxis request traces</h1><p>Live semantic routing with the latest user question saved alongside each recent trace.</p></div><span class="live">LIVE</span></header>
<section class="graphic"><div class="canvas"><svg viewBox="0 0 980 690" role="img" aria-labelledby="diagram-title diagram-desc">
<title id="diagram-title">Live Praxis request routing</title><desc id="diagram-desc">Open WebUI sends a request to Praxis. Praxis calls llm-d-sc for classification and routes the request to either the CPU or GPU vLLM service.</desc>
<defs><marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L10,3 L0,6 z" fill="#64b5ff"/></marker><marker id="arrow-cyan" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L10,3 L0,6 z" fill="#55e6d0"/></marker><marker id="arrow-red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L10,3 L0,6 z" fill="#ff4d5e"/></marker></defs>
<path id="packet-source" class="packet-path" d="M490 124 V180"/><path id="packet-classify" class="packet-path" d="M623 225 H720"/><path id="packet-result" class="packet-path" d="M720 256 H623"/><path id="packet-cpu" class="packet-path" d="M490 292 V350 H245 V598"/><path id="packet-gpu" class="packet-path" d="M490 292 V350 H735 V598"/><circle id="request-dot" class="packet" r="7" fill="#64b5ff"/>

<text x="0" y="30" class="label">REQUEST</text><rect x="290" y="48" width="400" height="76" rx="18" class="box-blue"/><text x="330" y="84" class="title">Open WebUI / client</text><text x="330" y="105" class="small">OpenAI-compatible POST /v1/chat/completions</text><path id="flow-source" d="M490 124 V180" class="flow"/>

<text x="0" y="160" class="label">SEMANTIC ROUTING</text><rect x="357" y="180" width="266" height="112" rx="18" class="box-blue"/><text x="385" y="218" class="title">Praxis / AI Gateway</text><text x="385" y="242" class="small">Classifies, selects, and proxies</text><rect x="720" y="180" width="210" height="112" rx="18" class="box-cyan"/><text x="750" y="220" class="title">llm-d-sc</text><text x="750" y="243" class="small">Returns complexity label</text><rect x="750" y="257" width="70" height="21" rx="10" class="pill"/><text x="785" y="271" text-anchor="middle" class="pill-text">SIMPLE</text><rect x="828" y="257" width="70" height="21" rx="10" class="pill"/><text x="863" y="271" text-anchor="middle" class="pill-text">COMPLEX</text><path id="flow-classify" d="M623 225 H720" class="flow-cyan"/><path id="flow-result" d="M720 256 H623" class="flow-cyan"/>

<path id="flow-cpu" d="M490 292 V350 H245 V382" class="flow cpu-path"/><path id="flow-gpu" d="M490 292 V350 H735 V382" class="flow gpu-path"/><text x="0" y="370" class="label">INFERENCE ROUTING</text><rect x="90" y="386" width="310" height="106" rx="18" class="box-violet"/><text x="116" y="425" class="title">vLLM CPU service</text><text x="116" y="450" class="small">SIMPLE / MEDIUM requests</text><rect x="580" y="386" width="310" height="106" rx="18" class="box-amber"/><text x="606" y="425" class="title">vLLM GPU service</text><text x="606" y="450" class="small">COMPLEX / REASONING requests</text><path id="flow-cpu-pod" d="M245 492 V550" class="flow cpu-path"/><path id="flow-gpu-pod" d="M735 492 V550" class="flow gpu-path"/>

<text x="0" y="536" class="label">OPENSHIFT WORKLOADS</text><rect x="55" y="550" width="405" height="120" rx="20" class="cluster"/><text x="82" y="582" class="cluster-label">CPU pool</text><rect x="104" y="598" width="282" height="50" rx="12" class="box-violet"/><text x="245" y="628" text-anchor="middle" class="small">vllm-cpu-predictor pod</text><rect x="520" y="550" width="405" height="120" rx="20" class="cluster"/><text x="547" y="582" class="cluster-label">GPU pool</text><rect x="594" y="598" width="282" height="50" rx="12" class="box-amber"/><text x="735" y="628" text-anchor="middle" class="small">vllm-gpu-predictor pod</text>
</svg></div><footer class="legend"><span><i class="swatch"></i> request / routing path</span><span><i class="swatch cyan"></i> classifier request + result</span></footer></section>

<section class="traces"><div class="section-heading"><h2>Recent requests <small id="count"></small></h2><button id="clear-recents" type="button">Clear Recents</button></div><table><thead><tr><th>Time</th><th>Request</th><th>Classification</th><th>Destination</th><th>Status</th><th>Duration</th></tr></thead><tbody id="traces"><tr><td colspan="6">Waiting for a request…</td></tr></tbody></table></section>
</main><div id="drawer-backdrop" class="drawer-backdrop"></div><aside id="question-drawer" class="drawer" aria-hidden="true" aria-labelledby="question-title"><div class="drawer-header"><h2 id="question-title">Question</h2><button id="drawer-close" class="drawer-close" type="button" aria-label="Close question">×</button></div><div id="question-meta" class="drawer-meta"></div><p id="question-text" class="question-text"></p></aside><script>
const $=id=>document.getElementById(id),dot=$('request-dot'),paths={source:$('packet-source'),classify:$('packet-classify'),result:$('packet-result'),cpu:$('packet-cpu'),gpu:$('packet-gpu')},flows={source:[$('flow-source')],classify:[$('flow-classify'),$('flow-result')],cpu:[$('flow-cpu'),$('flow-cpu-pod')],gpu:[$('flow-gpu'),$('flow-gpu-pod')]};let last='',lastAnimated='',selectedId='',dismissedThroughId='',traceItems=[],animating=false,animationGeneration=0;
const questionDrawer=$('question-drawer'),drawerBackdrop=$('drawer-backdrop');
function esc(x){return String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function isGpu(t){return t.route?t.route.toLowerCase().includes('gpu'):['COMPLEX','REASONING'].includes((t.classification||'').toUpperCase())}
function pathText(t){if(!(t.route||t.classification))return'Open WebUI → Praxis (routing metadata unavailable)';return`Open WebUI → Praxis → llm-d-sc → ${isGpu(t)?'vLLM GPU service → vllm-gpu-predictor pod':'vLLM CPU service → vllm-cpu-predictor pod'}`}
function highlight(t){Object.values(flows).flat().forEach(path=>path.classList.remove('selected'));if(!t)return;flows.source.forEach(path=>path.classList.add('selected'));if(t.route||t.classification){flows.classify.forEach(path=>path.classList.add('selected'));flows[isGpu(t)?'gpu':'cpu'].forEach(path=>path.classList.add('selected'))}}
async function travel(path,color,duration){const length=path.getTotalLength(),started=performance.now();dot.style.display='block';dot.setAttribute('fill',color);dot.style.color=color;return new Promise(resolve=>{function frame(now){const progress=Math.min((now-started)/duration,1),point=path.getPointAtLength(length*progress);dot.setAttribute('cx',point.x);dot.setAttribute('cy',point.y);if(progress<1)requestAnimationFrame(frame);else resolve()}requestAnimationFrame(frame)})}
async function animate(t){if(!t||animating||matchMedia('(prefers-reduced-motion: reduce)').matches)return;const generation=animationGeneration;animating=true;await travel(paths.source,'#64b5ff',450);if(generation!==animationGeneration){animating=false;return}if(t.route||t.classification){await travel(paths.classify,'#55e6d0',450);if(generation!==animationGeneration){animating=false;return}await travel(paths.result,'#55e6d0',450);if(generation!==animationGeneration){animating=false;return}await travel(paths[isGpu(t)?'gpu':'cpu'],isGpu(t)?'#ffc76a':'#ad9cff',900)}await sleep(300);dot.style.display='none';animating=false}
function render(){const items=traceItems;$('count').textContent=`(${items.length} saved)`;$('traces').innerHTML=items.length?items.map(t=>`<tr data-trace-id="${esc(t.id)}" tabindex="0" class="${t.id===selectedId?'selected-row':''}"><td>${new Date(t.started_at*1000).toLocaleTimeString()}</td><td><span class="tag">${esc(t.method)}</span> ${esc(t.path)}${t.question?`<button class="question-button" type="button" data-question-id="${esc(t.id)}">View question</button>`:'<span class="no-question">question unavailable</span>'}<details><summary>trace ${esc(t.request_id.slice(0,8))}</summary><div class="trace-detail"><span><b>Path:</b> ${esc(pathText(t))}</span><span><b>Request ID:</b> ${esc(t.request_id)}</span><span><b>Classification:</b> ${esc(t.classification||'label not emitted')}</span><span><b>Destination:</b> ${esc(t.route||'not emitted')}</span><span><b>Result:</b> ${esc(t.status||'in flight')} · ${t.duration_ms?esc(t.duration_ms+' ms'):'in progress'}${t.fallback?' · fallback':''}</span>${t.error?`<span><b>Error:</b> ${esc(t.error)}</span>`:''}</div></details></td><td>${esc(t.classification||'not emitted')}</td><td class="${isGpu(t)?'gpu':(t.route||t.classification)?'cpu':'unknown'}">${esc(t.route||'unavailable')}${t.fallback&&!isGpu(t)?'<small class="fellback">fell back</small>':t.route&&!t.classification&&t.status&&!t.error?'<small class="defaulted">no label &middot; default</small>':''}</td><td>${esc(t.status||'in flight')}</td><td>${t.duration_ms?esc(t.duration_ms+' ms'):'—'}</td></tr>`).join(''):'<tr><td colspan="6">Waiting for a request…</td></tr>'}
function syncSelection(){document.querySelectorAll('tr[data-trace-id]').forEach(row=>row.classList.toggle('selected-row',row.dataset.traceId===selectedId))}
function clearSelection(){selectedId='';dismissedThroughId=(traceItems.find(t=>t.status&&(t.route||t.classification))||{}).id||'';animationGeneration++;dot.style.display='none';highlight(null);syncSelection()}
function selectTrace(id){if(selectedId===id){clearSelection();return}const trace=traceItems.find(item=>item.id===id);if(!trace)return;selectedId=id;dismissedThroughId='';syncSelection();highlight(trace);animate(trace)}
function openQuestion(id){const trace=traceItems.find(item=>item.id===id);if(!trace||!trace.question)return;selectedId=id;dismissedThroughId='';syncSelection();highlight(trace);$('question-meta').textContent=`${new Date(trace.started_at*1000).toLocaleString()} · trace ${trace.request_id.slice(0,8)}`;$('question-text').textContent=trace.question;questionDrawer.classList.add('open');drawerBackdrop.classList.add('open');questionDrawer.setAttribute('aria-hidden','false');$('drawer-close').focus()}
function closeQuestion(){questionDrawer.classList.remove('open');drawerBackdrop.classList.remove('open');questionDrawer.setAttribute('aria-hidden','true')}
async function load(){try{const data=await fetch('/_trace/api/traces',{cache:'no-store'}).then(r=>r.json()),items=data.traces;traceItems=items;const key=items.map(x=>x.id+':'+x.status).join();if(key!==last){last=key;render()}if(selectedId){const selected=items.find(t=>t.id===selectedId);if(selected)highlight(selected);else clearSelection()}else{const routed=items.find(t=>t.status&&(t.route||t.classification));if(routed&&routed.id!==dismissedThroughId){highlight(routed);if(routed.id!==lastAnimated){lastAnimated=routed.id;animate(routed)}}}}catch(e){$('traces').innerHTML='<tr><td colspan="6">Trace API unavailable.</td></tr>'}}
$('traces').addEventListener('click',event=>{const question=event.target.closest('[data-question-id]');if(question){openQuestion(question.dataset.questionId);return}if(event.target.closest('details'))return;const row=event.target.closest('tr[data-trace-id]');if(row)selectTrace(row.dataset.traceId)});
$('traces').addEventListener('toggle',event=>{if(event.target.tagName==='DETAILS'&&event.target.open){const row=event.target.closest('tr[data-trace-id]');if(row&&row.dataset.traceId!==selectedId)selectTrace(row.dataset.traceId)}},true);
$('traces').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){const row=event.target.closest('tr[data-trace-id]');if(row){event.preventDefault();selectTrace(row.dataset.traceId)}}});
document.querySelector('.graphic').addEventListener('click',()=>{if(selectedId)clearSelection()});
$('drawer-close').addEventListener('click',closeQuestion);drawerBackdrop.addEventListener('click',closeQuestion);document.addEventListener('keydown',event=>{if(event.key==='Escape'&&questionDrawer.classList.contains('open'))closeQuestion()});
$('clear-recents').addEventListener('click',async()=>{const button=$('clear-recents');button.disabled=true;try{const response=await fetch('/_trace/api/traces',{method:'DELETE'});if(!response.ok)throw new Error();traceItems=[];selectedId='';dismissedThroughId='';last='';lastAnimated='';animationGeneration++;dot.style.display='none';highlight(null);closeQuestion();render()}finally{button.disabled=false}});
load();setInterval(load,2000)
</script></body></html>'''


if __name__ == "__main__":
    with db():
        pass
    for port, upstream, route in (
        (8081, CPU_UPSTREAM, "vllm-cpu"),
        (8082, GPU_UPSTREAM, "vllm-gpu"),
    ):
        threading.Thread(
            target=RelayServer(("0.0.0.0", port), upstream, route).serve_forever,
            daemon=True,
        ).start()
    RelayServer(("0.0.0.0", 8080), UPSTREAM).serve_forever()
