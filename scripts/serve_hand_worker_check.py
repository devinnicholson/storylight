"""Local-only browser acceptance server; no camera access or cloud providers."""

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--port", type=int, default=18086)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
app = FastAPI()
app.mount("/workbench-assets", StaticFiles(directory=root / "src/storylight/static"))


@app.get("/")
def page():
    return FileResponse(root / "tests/browser/hand-worker.html")


@app.get("/fixture.jpg")
def fixture():
    return FileResponse(args.fixture.resolve())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port)
