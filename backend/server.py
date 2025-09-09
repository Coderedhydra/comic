import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse


DATA_OUTPUTS = os.environ.get("DATA_OUTPUTS", "/workspace/data/outputs")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.isdir(DATA_OUTPUTS):
    os.makedirs(DATA_OUTPUTS, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=DATA_OUTPUTS), name="outputs")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/project")
def read_project():
    proj_path = os.path.join(DATA_OUTPUTS, "project.json")
    if not os.path.isfile(proj_path):
        raise HTTPException(status_code=404, detail="project.json not found")
    with open(proj_path, "r", encoding="utf-8") as f:
        import json
        data = json.load(f)
    # include a base_url hint for clients
    data.setdefault("base_url", "/outputs")
    return JSONResponse(content=data)

