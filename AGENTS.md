# Agent guide

## What this repository is

This is a deployment/demo repository, not the source tree for Praxis, vLLM, or
`llm-d-sc`. It deploys those externally built images to OpenShift as one request
path, and adds one small app of its own (`trace-ui/`):

`Open WebUI / client -> trace-ui :8080 -> Praxis :8080 -> llm-d-sc :50051 ->
trace-ui :8081 (CPU) or :8082 (GPU) -> KServe predictor Service -> vLLM`

The semantic classifier labels each prompt. `SIMPLE` and `MEDIUM` use the CPU
model; `COMPLEX` and `REASONING` use the GPU model. If classification is
unavailable or overloaded, Praxis falls back to CPU.

A request whose chosen backend fails (transport error or 429/5xx) is replayed
once against `vllm-any`, a health-checked pool of both backends, so CPU traffic
can fail over to the GPU and the reverse.

## Repository map

- `README.md`: OpenShift prerequisites, deploy steps, and a screenshot.
- `openshift/`: Kustomize base plus the RHOAI overlay. Read its nested
  `AGENTS.md` before changing manifests.
- `trace-ui/`: `server.py` (stdlib-only Python) and its `Dockerfile`. It serves
  the trace/architecture page and relays traffic in front of Praxis and each
  vLLM backend. It is the only application code here.
- `architecture.html`, `llm-d-sc-slides.html`: standalone architecture
  visualization and slides. Documentation; not kept in sync automatically.
- `imgs/`: README screenshots; treat as documentation assets.

There is no package manager or automated test suite. Most changes should be
validated as rendered manifests; `trace-ui/server.py` also needs the syntax
checks below.

## Common commands

Fast validation that does not deploy anything:

```bash
kustomize build openshift/overlays/rhoai >/tmp/praxis-rendered.yaml
python3 -I -m py_compile trace-ui/server.py
git diff --check
```

The page JavaScript lives inside the `PAGE` string in `server.py` and is not
covered by `py_compile`; extract the `<script>` block and run `node --check` on
it after editing it, and check diagram or layout changes visually.

Deploying is in the README (`oc apply -k openshift/overlays/rhoai`).

Trace UI image (only when asked; these mutate the cluster):

```bash
oc -n praxis-sr-demo patch bc praxis-trace-ui --type=merge -p '{"spec":{"output":{"to":{"kind":"ImageStreamTag","name":"praxis-trace-ui:vNN"}}}}'
oc -n praxis-sr-demo start-build praxis-trace-ui --from-dir=trace-ui --follow
```

The `praxis-trace-ui` BuildConfig is a binary Docker build that lives in the
cluster, not in this repo, and its output tag is repointed by hand for each
build. The image is then mirrored to Quay with `oc image mirror`, given its
release version tag, and that tag is set in
`openshift/base/trace-ui-deployment.yaml`.

## Change rules and invariants

- Praxis reaches vLLM only through the trace-ui relays. Do not point it at the
  predictor Services directly without changing trace-ui, which records health
  probes and per-hop timing from that traffic.
- The failover pool `vllm-any` only skips unhealthy endpoints because it has its
  own entry in the top-level `clusters:` list with a `health_check`; health state
  is looked up by cluster name. Add a backend to both that entry and the
  `failover` step's `load_balancer`. Rules cannot tell which cluster failed, so
  failover relies on health checks and a just-failed backend can briefly be
  picked again.
- trace-ui joins a request to its backend hops with a `_trace=<id>` query
  parameter it adds on the way to Praxis and strips at the backend relays. Praxis
  keeps the request URI but drops headers when it moves to the next step, so do
  not replace this with a header.
- trace-ui's SQLite database persists across image versions, so schema changes
  need a migration in `db()`.
- trace-ui's ServiceAccount is deliberately narrow: read pods and Deployments,
  and scale only `praxis-traffic-gen`. Treat any widening as a security decision
  and call it out.
- The traffic generator Deployment `praxis-traffic-gen` stays at `replicas: 0`
  in git; the trace UI's Simulate Traffic button scales it.
- Preserve the classifier contract: gRPC on `50051`, classifier name
  `complexity`, and a model mounted at `/models`.
- Preserve the public API contract: OpenAI-compatible Chat Completions at
  `/v1/chat/completions` through Praxis port `8080`.
- Never commit tokens, registry credentials, downloaded models, `.env` files, or
  generated Python caches.
- All images are pinned to version tags; do not reintroduce `:latest`. A new tag
  needs the image built and mirrored to Quay first. Model names, runtime names,
  resource sizes, and the OpenShift route hostname are demo/environment choices.
  Call out intentional changes to them in documentation or the handoff.
- Keep edits narrow. Do not reformat all YAML around a functional change.

## Praxis reference

Config docs live upstream at `github.com/praxis-proxy/praxis`, under
`docs/filters/http/traffic_management/` (for example
`iterative_request_router.md` for `steps` and `on_result`).

## Before handing off

Run the relevant fast validation above, then summarize what was and was not
tested. Full deployment is environment-dependent and should not be claimed
unless the OpenShift resources were actually exercised. Do not run `oc apply`,
builds, or other cluster-changing commands unless asked.
