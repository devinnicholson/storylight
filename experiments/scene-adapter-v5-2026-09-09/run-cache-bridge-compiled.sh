#!/usr/bin/env bash
set -euo pipefail
cd /opt/bookforge-scene-adapter-v5
[[ ! -e cache-bridge-compiled-01 && ! -L cache-bridge-compiled-01 ]] || exit 2
sha256sum --check --status <<'PINS'
b308c7e236ef81b30915fe416639a6b8032e7a5faa2b94a422754082d62a4939  /tmp/bookforge-v5-cache-bridge-compiled.py
66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c  training-01/completed.json
e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31  /tmp/bookforge-v5-cache-bridge-reuse.py
PINS
exec /usr/bin/timeout --signal=TERM --kill-after=30s 960s \
  .venv/bin/python /tmp/bookforge-v5-cache-bridge-compiled.py \
  --compiler-cache-root /tmp/bookforge-v5-bridge-compiler-cache-01 \
  --bridge-helper /tmp/bookforge-v5-cache-bridge-reuse.py \
  --v5-trainer payload/train.py --v5-runner comparison/run.py \
  --v2-trainer payload/v2-train.py --v3-runner comparison/v3-run.py \
  --merge-helper comparison/serving-profile.py \
  --model-dir model --model-manifest model-manifest.json \
  --model-manifest-sha256 703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d \
  --training-run training-01 \
  --completed-sha256 66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c \
  --data-dir payload/data --grammar comparison/grammar.ebnf \
  --output cache-bridge-compiled-01 --max-runtime-seconds 900
