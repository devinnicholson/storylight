#!/usr/bin/env bash
set -euo pipefail
cd /opt/storylight-scene-adapter-v5
arm=${1:?Pass cold-0, reuse-0 or reuse-1}
prior_args=()
case "$arm" in
  cold-0) export PYTHONHASHSEED=0; [[ $# == 1 ]] ;;
  reuse-0) export PYTHONHASHSEED=0; prior=seeded-cold-0 ;;
  reuse-1) export PYTHONHASHSEED=1; prior=seeded-reuse-0 ;;
  *) exit 2 ;;
esac
if [[ "$arm" != cold-0 ]]; then
  [[ $# == 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || exit 2
  prior_args=(--prior-run "$prior" --prior-completed-sha256 "$2")
fi
[[ ! -e "seeded-$arm" && ! -L "seeded-$arm" ]] || exit 2
sha256sum --check --status <<'PINS'
4a3c44494e45821817b56fe5bb846d520e30a0bd0e28797160316710aaa37e00  /tmp/storylight-compiler-restart-run.py
b308c7e236ef81b30915fe416639a6b8032e7a5faa2b94a422754082d62a4939  support/cache-bridge-compiled.py
e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31  support/cache-bridge-reuse.py
66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c  training-01/completed.json
PINS
exec /usr/bin/timeout --signal=TERM --kill-after=30s 960s \
  .venv/bin/python /tmp/storylight-compiler-restart-run.py \
  --arm "$arm" "${prior_args[@]}" \
  --compiled-profile support/cache-bridge-compiled.py \
  --compiler-cache-root /tmp/storylight-v5-seeded-compiler-cache-01 \
  --bridge-helper support/cache-bridge-reuse.py \
  --v5-trainer payload/train.py --v5-runner comparison/run.py \
  --v2-trainer payload/v2-train.py --v3-runner comparison/v3-run.py \
  --merge-helper comparison/serving-profile.py \
  --model-dir model --model-manifest model-manifest.json \
  --model-manifest-sha256 703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d \
  --training-run training-01 \
  --completed-sha256 66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c \
  --data-dir payload/data --grammar comparison/grammar.ebnf \
  --output "seeded-$arm" --max-runtime-seconds 900
