"""HTTP API for the web interface (the same functions as the MCP tools)."""
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from s2d import server

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Sketch / text to diagram")


class Text(BaseModel):
    description: str


@app.post("/api/text")
def text(t: Text):
    return server.text_to_diagram(t.description)


@app.post("/api/sketch")
async def sketch(file: UploadFile = File(...)):
    return server.sketch_to_diagram(base64.b64encode(await file.read()).decode())


dist = ROOT / "web" / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(dist / "index.html")
