# OpenShift manifest guide

This directory deploys the same logical stack as the compose files under
`podman-compose/` using Kustomize, OpenShift Routes, KServe `InferenceService`
resources, and a ModelCar init container, plus Open WebUI and a trace UI that
the compose stack does not have.

## Layout and ownership

- `base/`: reusable Deployments, Services, Routes, and storage:
  - Praxis (`praxis-*.yaml`) and the classifier (`classifier-*.yaml`).
  - Open WebUI (`openwebui-*.yaml`).
  - trace-ui (`trace-ui-*.yaml`), including its PVC and RBAC.
  - `traffic-deployment.yaml`, the traffic generator the trace UI scales.
- `base/praxis.yaml`: Praxis config embedded into a generated ConfigMap by
  `base/kustomization.yaml`.
- `overlays/rhoai/`: deployable environment overlay, including Namespace,
  the CPU/GPU `ServingRuntime` and `InferenceService` backends, and cluster
  specifics.
- The deploy steps and screenshots are in the root `README.md`; there is no
  separate `openshift/README.md`.

Make reusable changes in `base/`; keep cluster-specific values in an overlay.
The current overlay targets namespace `praxis-sr-demo`. Although the base names
`praxis-demo`, the overlay namespace wins in the rendered output.

## Runtime relationships to preserve

- Open WebUI sends to `http://trace-ui:8080/v1`, the trace-ui ingress relay,
  which forwards to Praxis.
- Praxis loads `/etc/praxis/config.yaml` from the generated `praxis-config`
  ConfigMap and exposes only HTTP `8080` through its Service and Route.
- Praxis admin ports stay Pod-local and supply the readiness/liveness
  endpoints; see the comment in `base/praxis-deployment.yaml` before changing
  which one the probes use.
- `llm-d-sc` receives its model through the ModelCar init container and serves
  gRPC on `50051`.
- Praxis routes to trace-ui, not to the predictors: `trace-ui:8081` is the CPU
  relay and `:8082` the GPU relay. The relays forward to
  `CPU_UPSTREAM` / `GPU_UPSTREAM`, which point at the KServe predictor Services
  `vllm-cpu-predictor` and `vllm-gpu-predictor` on port `80` (targetPort `8080`).
- trace-ui's ServiceAccount (`trace-ui-rbac.yaml`) can read pods and
  Deployments, and scale only `praxis-traffic-gen` (pinned with
  `resourceNames`). Any wider write access is a security decision.
- `praxis-traffic-gen` stays at `replicas: 0`; the trace UI's Simulate Traffic
  button scales it to 0 or 1. Do not set replicas in a patch.
- Both `InferenceService` resources depend on cluster-provided runtime names.
  Do not assume those names are portable to another cluster.

## Validation

Render before applying:

```bash
kustomize build openshift/overlays/rhoai >/tmp/praxis-rendered.yaml
oc apply --dry-run=client -f /tmp/praxis-rendered.yaml
```

For an actual deployment, follow the OpenShift section of the root `README.md`
(`oc apply -k openshift/overlays/rhoai`). Do not run `oc apply` or mutate a
cluster merely to validate a manifest unless the user explicitly asked for
deployment. A changed trace-ui image tag needs a built and mirrored image first;
see the root `AGENTS.md`.

When changing image names, route hostnames, namespaces, runtime names, model
URIs, or resource requests, treat them as environment-specific choices and
update the README when operator action is required. The Praxis image is pinned
(`v0.7.0`); the trace-ui image tag is bumped by hand after each build.
