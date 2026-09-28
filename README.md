# Praxis & llm-d-sc Demo

### Deploy

```bash
# download the classifier model (requires the Hugging Face CLI)
hf download cnuland/llm-d-sc-complexity --local-dir ./models/llm-d-sc-complexity

# point LLM_D_SC_MODEL_DIR in .env at the downloaded directory
cp .env.example .env
vi .env

# ensure you have access to the registry
podman login registry.redhat.io

# start the stack!
podman compose -f podman-compose/podman-compose.yaml up -d
```

And then test the stack once it's all up:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64
  }'
```

Praxis classifies the prompt with `llm-d-sc` and routes it to the CPU or GPU vLLM backend.

### Results

![Compose logs](imgs/compose-logs.png)
*Compose logs*

![Curl responses](imgs/curl-responses.png)
*Curl responses*

## OpenShift stack

The RHOAI overlay deploys Open WebUI, the trace UI, Praxis, `llm-d-sc`, and
CPU and GPU KServe vLLM predictors. Praxis classifies each request and routes
it to the appropriate predictor.

### Minimum prerequisites

- `oc` access and `kustomize` as an authenticated OpenShift user with permission to create a namespace.
- RHOAI/KServe installed
- At least one schedulable NVIDIA GPU: the GPU predictor requests one
  `nvidia.com/gpu`.

### Deploy

1. Apply the manifests to the cluster:

   ```bash
   oc apply -k openshift/overlays/rhoai
   oc -n praxis-sr-demo get pods --watch
   ```

2. Model downloads and first GPU scheduling can take several minutes. Press
   `Ctrl-C` once the pods are ready, then retrieve the UI URLs:

   ```bash
   WEBUI_HOST=$(oc -n praxis-sr-demo get route openwebui -o jsonpath='{.spec.host}')
   TRACE_UI_HOST=$(oc -n praxis-sr-demo get route trace-ui -o jsonpath='{.spec.host}')
   echo "Open WebUI: https://${WEBUI_HOST}"
   echo "Trace UI: https://${TRACE_UI_HOST}"
   ```

### Results

![OpenShift request trace diagram](imgs/trace-diagram.png)

![OpenShift recent request traces](imgs/trace-logs.png)
