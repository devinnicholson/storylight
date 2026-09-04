"""Start the isolated candidate workbench; no GPU allocation until explicit user action."""

import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--planner-url", default="http://127.0.0.1:18435")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    settings = {
        "MODEL_BACKEND": "openai",
        "MODEL_BASE_URL": args.planner_url,
        "MODEL_NAME": "llm",
        "MODEL_API_KEY": "",
        "ASR_BACKEND": "disabled",
        "ASSET_BACKEND": "disabled",
        "ANTICIPATORY_BACKEND": "disabled",
        "LIVE_SCENE_CRITIC_BACKEND": "disabled",
        "LIVE_SCENE_BACKEND": "modal_klein",
        "LIVE_SCENE_PLANNER": "model",
        "LIVE_SCENE_PLANNER_BACKEND": "tensorrt_slots",
        "LIVE_SCENE_PLANNER_BASE_URL": args.planner_url,
        "LIVE_SCENE_PLANNER_MODEL_REVISION": (
            "sha256:95b69991b68c57a2d2d4bfa4116feb9ec57295588551d109353a42a9c16c4fdf"
        ),
        "LIVE_SCENE_PLANNER_AUTO_WARMUP": "false",
        "LIVE_SCENE_ENABLE_MOTION": "false",
        "DATA_DIR": ".bookforge/overnight-candidate/data",
        "CACHE_DIR": ".bookforge/overnight-candidate/cache",
        "LIVE_SCENE_OUTPUT_DIR": ".bookforge/overnight-candidate/scenes",
        "LIVE_SCENE_MODAL_PLAN_FILE": "experiments/renderer-fidelity/overnight-modal-plan.json",
        "LIVE_SCENE_MODAL_LEDGER_PATH": ".bookforge/overnight-candidate/modal-ledger.json",
        "LIVE_SCENE_MODAL_SESSION_GPU_CAP_USD": "3",
        "ASSET_MODAL_COMMAND": str(root / ".venv/bin/modal"),
    }
    os.environ.update({f"BOOKFORGE_{key}": value for key, value in settings.items()})
    uvicorn.run("bookforge.api:app", host="127.0.0.1", port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
