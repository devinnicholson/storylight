#!/usr/bin/env bash
set -euo pipefail
project=your-gcp-project
run=storylight-compiler-restart-20260909
zone=us-east1-b
receipt_dir=${1:?Pass a new local receipt directory}
mkdir "$receipt_dir"
gcloud compute firewall-rules create "${run}-deny" --project="$project" \
  --network=default --direction=INGRESS --priority=900 --action=DENY \
  --rules=all --source-ranges=0.0.0.0/0 --target-tags="$run" --format=json \
  > "$receipt_dir/firewall-deny.json"
gcloud compute firewall-rules create "${run}-iap" --project="$project" \
  --network=default --direction=INGRESS --priority=800 --action=ALLOW \
  --rules=tcp:22 --source-ranges=35.235.240.0/20 --target-tags="$run" --format=json \
  > "$receipt_dir/firewall-iap.json"
gcloud compute instances create "$run" --project="$project" --zone="$zone" \
  --machine-type=g2-standard-8 --image-project=deeplearning-platform-release \
  --image=pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20260831 \
  --boot-disk-size=100GB --boot-disk-auto-delete --boot-disk-type=pd-balanced \
  --network-interface=network=default,stack-type=IPV4_ONLY \
  --tags="$run" --metadata=enable-oslogin=TRUE \
  --no-service-account --no-scopes --no-restart-on-failure \
  --maintenance-policy=TERMINATE --max-run-duration=2h \
  --instance-termination-action=DELETE --format=json > "$receipt_dir/instance-created.json"
gcloud compute instances describe "$run" --project="$project" --zone="$zone" \
  --format=json > "$receipt_dir/instance-readback.json"
