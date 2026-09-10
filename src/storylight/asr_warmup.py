from __future__ import annotations

import asyncio
import json

from storylight.asr import build_asr_backend
from storylight.asr_jetson import WhisperTrtBackend
from storylight.config import Settings


async def warm(settings: Settings) -> dict[str, object]:
    backend = build_asr_backend(settings)
    if not isinstance(backend, WhisperTrtBackend):
        raise RuntimeError("STORYLIGHT_ASR_BACKEND must be whisper_trt")
    await backend.warmup()
    return {
        "ready": True,
        "backend": backend.name,
        "model": backend.model_name,
        "engine_path": backend.engine_path,
    }


def main() -> int:
    result = asyncio.run(warm(Settings()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
