from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.inference.runtime import GenerationRuntime

app = FastAPI(title="Text-to-3D Generation API", version="1.0.0")


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    seed: int | None = None
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


@lru_cache(maxsize=1)
def get_runtime() -> GenerationRuntime:
    checkpoint = os.getenv("MODEL_CHECKPOINT", "artifacts/generator_checkpoint.pt")
    return GenerationRuntime(checkpoint)


@app.get("/health")
def health():
    checkpoint = Path(os.getenv("MODEL_CHECKPOINT", "artifacts/generator_checkpoint.pt"))
    return {"status": "ok", "model_ready": checkpoint.exists()}


@app.post("/generate")
def generate(request: GenerationRequest):
    try:
        runtime = get_runtime()
        stem = re.sub(r"[^a-z0-9]+", "-", request.prompt.lower()).strip("-")[:40] or "shape"
        filename = f"{stem}-{uuid4().hex[:8]}.obj"
        output = Path("artifacts/generated") / filename
        runtime.generate_mesh(
            request.prompt,
            output_path=output,
            seed=request.seed,
            threshold=request.threshold,
        )
        return {
            "prompt": request.prompt,
            "seed": request.seed,
            "mesh_path": str(output),
            "download_url": f"/meshes/{filename}",
        }
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/meshes/{filename}")
def mesh_file(filename: str):
    safe_name = Path(filename).name
    path = Path("artifacts/generated") / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(path, media_type="model/obj", filename=safe_name)
