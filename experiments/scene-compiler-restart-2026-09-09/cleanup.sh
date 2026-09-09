#!/usr/bin/env bash
set -euo pipefail
project=your-gcp-project
run=bookforge-compiler-restart-20260909
zone=us-east1-b
instance=$(gcloud compute instances list --project="$project" --filter="name=$run" --format='value(name)')
if [[ -n "$instance" ]]; then
  [[ "$instance" == "$run" ]]
  if ! gcloud compute instances delete "$run" --project="$project" --zone="$zone" --quiet; then
    remaining=$(gcloud compute instances list --project="$project" --filter="name=$run" --format='value(name)')
    [[ -z "$remaining" ]]
  fi
fi
for rule in "${run}-iap" "${run}-deny"; do
  existing=$(gcloud compute firewall-rules list --project="$project" --filter="name=$rule" --format='value(name)')
  if [[ -n "$existing" ]]; then
    [[ "$existing" == "$rule" ]]
    gcloud compute firewall-rules delete "$rule" --project="$project" --quiet
  fi
done
for resource in instances disks; do
  remaining=$(gcloud compute "$resource" list --project="$project" --filter="name=$run" --format='value(name)')
  [[ -z "$remaining" ]]
done
for rule in "${run}-iap" "${run}-deny"; do
  remaining=$(gcloud compute firewall-rules list --project="$project" --filter="name=$rule" --format='value(name)')
  [[ -z "$remaining" ]]
done
printf '%s\n' 'Verified experiment VM, boot disk and firewall rules absent.'
