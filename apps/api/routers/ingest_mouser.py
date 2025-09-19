import os, httpx, asyncio, pathlib
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "aYG0l9s7dbB7j3gbS")
TOKEN = os.getenv("APIFY_TOKEN")

router = APIRouter()

@router.post("/run")
async def run_actor(payload: Dict[str, Any]):
    if not TOKEN:
        raise HTTPException(400, "Missing APIFY_TOKEN")
    url = f"{APIFY_BASE}/acts/{ACTOR_ID}/runs?token={TOKEN}"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json().get("data", {})

@router.get("/status/{run_id}")
async def run_status(run_id: str):
    if not TOKEN:
        raise HTTPException(400, "Missing APIFY_TOKEN")
    url = f"{APIFY_BASE}/actor-runs/{run_id}?token={TOKEN}"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json().get("data", {})

@router.post("/collect")
async def collect(body: Dict[str, Any]):
    if not TOKEN:
        raise HTTPException(400, "Missing APIFY_TOKEN")
    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(400, "Missing runId")

    dataset_url = f"{APIFY_BASE}/actor-runs/{run_id}/dataset/items?token={TOKEN}&format=json"
    async with httpx.AsyncClient(timeout=600) as client:
        items = (await client.get(dataset_url)).json()

    pdf_links = {i.get("datasheetUrl") for i in items if i.get("datasheetUrl")}
    samples_dir = pathlib.Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    async with httpx.AsyncClient(timeout=600) as client:
        for url in pdf_links:
            try:
                fname = url.split("/")[-1].split("?")[0] or "file.pdf"
                path = samples_dir / fname
                resp = await client.get(url)
                resp.raise_for_status()
                path.write_bytes(resp.content)
                downloaded += 1
            except Exception as e:
                print("skip", url, e)

    # Kick off parse+extract (sync for simplicity)
    from packages.extraction.ingest import load_docs_from_dir
    from packages.extraction.extract import extract_records_from_docs
    docs = load_docs_from_dir("data/samples")
    records = extract_records_from_docs(docs)

    # Upsert into DB (placeholder)
    from packages.storage.db import upsert_records
    upserted = upsert_records(records)

    return {"downloaded": downloaded, "parsed": len(records), "upserted": upserted}
