import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ingest_mouser, parts, replace

app = FastAPI(title="PartSync-Llama API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_mouser.router, prefix="/ingest/mouser", tags=["ingest:mouser"])
app.include_router(parts.router, prefix="/parts", tags=["parts"])
app.include_router(replace.router, prefix="/replace", tags=["replacement"])

@app.get("/")
def root():
    return {"status": "ok", "message": "PartSync-Llama API"}
