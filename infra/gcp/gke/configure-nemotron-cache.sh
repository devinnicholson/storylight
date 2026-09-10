#!/usr/bin/env bash
set -euo pipefail

CONTEXT="gke_${GOOGLE_CLOUD_PROJECT:-your-gcp-project}_${STORYLIGHT_GKE_REGION:-us-central1}_${STORYLIGHT_GKE_CLUSTER:-storylight-anticipatory}"
DEPLOYMENT=storylight-nemotron
NAMESPACE=storylight

if [[ "${STORYLIGHT_NIM_CACHE_APPLY:-}" != "I_UNDERSTAND_THIS_CONFIGURES_THE_STOPPED_NIM" ]]; then
  echo "Dry guard: configure persistent TensorRT engines only while NIM is scaled to zero."
  echo "Set STORYLIGHT_NIM_CACHE_APPLY=I_UNDERSTAND_THIS_CONFIGURES_THE_STOPPED_NIM to apply."
  exit 2
fi

kubectl --context="$CONTEXT" -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o json |
  python3 -c '
import json, sys
deployment = json.load(sys.stdin)
spec = deployment["spec"]
pod = spec["template"]["spec"]
nim = next(c for c in pod["containers"] if c["name"] == "nemotron-nim")
env = {item["name"]: item.get("value") for item in nim["env"]}
expected_image = "nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1@sha256:f4f0ef214fc448af0b9e7a9de84f8d163b21b698d2ad9e4a5d6d8afa8f03b065"
checks = [
    spec["replicas"] == 0,
    deployment.get("status", {}).get("replicas", 0) == 0,
    nim["image"] == expected_image,
    env.get("NIM_MAX_MODEL_LEN") == "2048",
    env.get("NIM_MAX_BATCH_SIZE") == "1",
    env.get("NIM_LOW_MEMORY_MODE") == "1",
    env.get("NIM_CACHE_PATH") == "/opt/nim/.cache",
    nim["resources"]["limits"].get("nvidia.com/gpu") == "1",
    pod["nodeSelector"].get("cloud.google.com/gke-accelerator") == "nvidia-l4",
    any(v.get("persistentVolumeClaim", {}).get("claimName") == "storylight-nim-cache" for v in pod["volumes"]),
    any(c["name"] == "gpu-watchdog" for c in pod["containers"]),
]
if not all(checks):
    sys.exit("Refusing cache configuration: stop NIM and verify the pinned L4/2048/batch-one profile first.")
'

kubectl --context="$CONTEXT" -n "$NAMESPACE" set env "deployment/$DEPLOYMENT" \
  --containers=nemotron-nim \
  NIM_MODEL_PROFILE=308eb4483d24f2e7f53a75d669cacd83539f3415167733accfea1a5efe6986aa \
  NIM_CUSTOM_MODEL_NAME=storylight-nvl8b-f4f0ef21-l4-bf16-ctx2048-b1-v1 \
  NIM_SERVED_MODEL_NAME=nvidia/llama-3.1-nemotron-nano-vl-8b-v1 \
  NIM_ENABLE_KV_CACHE_REUSE-

echo "Engine persistence configured; NIM remains stopped. First startup builds, subsequent startups can reuse the cached engines."
