"""Loopback-only review of original synthetic full/concise renderer pairs."""

import html
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".bookforge/overnight-20260904"
CORPUS = json.loads((ROOT / "experiments/renderer-fidelity/overnight-corpus.json").read_text())
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def gallery(split: str = "validation", start: int = 0):
    if split not in {"development", "validation"}:
        raise HTTPException(404)
    if start < 0 or start >= len(CORPUS[split]):
        raise HTTPException(404)
    result = [
        "<html><head><title>Bookforge fidelity comparison</title><style>",
        "body{background:#101720;color:#e8edf4;font:14px system-ui;margin:20px}",
        "article{margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}",
        "img{width:100%}p{margin:6px 0}h2{font-size:16px}a{color:#a8d9ff}",
        "</style></head><body><h1>Same Gemma output · same Klein model · same seed</h1>",
        "<p>Left: full contract. Right: concise candidate. "
        "These are newly generated images, not video.</p>",
    ]
    result.append(
        '<p><a href="/?split=development">Development</a> · '
        '<a href="/?split=validation">Validation</a></p>'
    )
    if start > 0:
        result.append(f'<a href="/?split={split}&start={max(0, start - 3)}">Previous three</a> ')
    if start + 3 < len(CORPUS[split]):
        result.append(f'<a href="/?split={split}&start={start + 3}">Next three</a>')
    for case in CORPUS[split][start : start + 3]:
        result.append(
            f"<article><h2>{case['id']} · {html.escape(case['text'])}</h2><div class=pair>"
        )
        for kind in ("full", "concise"):
            name = f"{case['id']}-{kind}-r0.jpg"
            if (DATA / f"{split}-render" / name).is_file():
                result.append(
                    f'<div><p>{kind}</p><img alt="{case["id"]} {kind} contract" '
                    f'src="/image/{split}/{name}"></div>'
                )
            else:
                result.append("<div>Planner failed before rendering. Counted as a failure.</div>")
        result.append("</div></article>")
    return "".join(result) + "</body></html>"


@app.get("/image/{split}/{name}")
def image(split: str, name: str):
    if split not in {"development", "validation"}:
        raise HTTPException(404)
    allowed = {f"{c['id']}-{kind}-r0.jpg" for c in CORPUS[split] for kind in ("full", "concise")}
    if name not in allowed:
        raise HTTPException(404)
    return FileResponse(DATA / f"{split}-render" / name)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18088)
