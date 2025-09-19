import os, httpx, asyncio, pathlib, logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from packages.crawlers.mouser import MouserCrawler
from packages.crawlers.mouser_api import MouserAPI
from packages.extraction.llamaindex_extractor import LlamaIndexExtractor
from packages.storage.db import upsert_records

logger = logging.getLogger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "aYG0l9s7dbB7j3gbS")
TOKEN = os.getenv("APIFY_TOKEN")

router = APIRouter()

class CrawlRequest(BaseModel):
    query: Optional[str] = None
    mpns: Optional[List[str]] = None
    manufacturers: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    max_pages: int = 5
    enrich_with_mouser: bool = True

class CrawlResponse(BaseModel):
    run_id: str
    status: str
    message: str

@router.post("/run", response_model=CrawlResponse)
async def run_actor(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start Apify crawler for Mouser datasheet collection"""
    if not TOKEN:
        raise HTTPException(400, "Missing APIFY_TOKEN")
    
    try:
        crawler = MouserCrawler()
        
        # Start the crawl
        result = await crawler.crawl_and_download(
            query=request.query or "",
            max_pages=request.max_pages,
            output_dir="data/samples"
        )
        
        # Schedule background processing
        background_tasks.add_task(
            process_crawled_data, 
            result["downloaded_files"], 
            request.enrich_with_mouser
        )
        
        return CrawlResponse(
            run_id=result["run_id"],
            status="completed",
            message=f"Downloaded {len(result['downloaded_files'])} datasheets"
        )
        
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        raise HTTPException(500, f"Crawl failed: {str(e)}")

@router.get("/status/{run_id}")
async def run_status(run_id: str):
    """Get status of a crawl run"""
    if not TOKEN:
        raise HTTPException(400, "Missing APIFY_TOKEN")
    
    try:
        crawler = MouserCrawler()
        status = await crawler.get_run_status(run_id)
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(500, f"Status check failed: {str(e)}")

@router.post("/collect")
async def collect(body: Dict[str, Any]):
    """Collect results from completed crawl and process"""
    if not TOKEN:
        raise HTTPException(400, "Missing APIFY_TOKEN")
    
    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(400, "Missing runId")

    try:
        crawler = MouserCrawler()
        items = await crawler.collect_results(run_id)
        downloaded_files = await crawler.download_datasheets(items, "data/samples")
        
        # Process with LlamaIndex
        extractor = LlamaIndexExtractor()
        parts = await extractor.process_datasheets(downloaded_files)
        
        # Enrich with Mouser API if requested
        if body.get("enrich_with_mouser", True):
            mouser_api = MouserAPI()
            enriched_parts = await mouser_api.batch_enrich_parts(parts)
            parts = enriched_parts
        
        # Store in database
        upserted = upsert_records(parts)
        
        return {
            "downloaded": len(downloaded_files),
            "parsed": len(parts),
            "upserted": upserted,
            "run_id": run_id
        }
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise HTTPException(500, f"Collection failed: {str(e)}")

@router.post("/search-mouser")
async def search_mouser_parts(query: str, limit: int = 50):
    """Search parts directly on Mouser API"""
    try:
        mouser_api = MouserAPI()
        results = await mouser_api.search_parts(query, limit)
        return results
    except Exception as e:
        logger.error(f"Mouser search failed: {e}")
        raise HTTPException(500, f"Mouser search failed: {str(e)}")

@router.post("/enrich-part")
async def enrich_part(part_data: Dict[str, Any]):
    """Enrich a single part with Mouser API data"""
    try:
        mouser_api = MouserAPI()
        enriched = await mouser_api.enrich_part_record(part_data)
        return enriched
    except Exception as e:
        logger.error(f"Part enrichment failed: {e}")
        raise HTTPException(500, f"Part enrichment failed: {str(e)}")

async def process_crawled_data(file_paths: List[str], enrich_with_mouser: bool = True):
    """Background task to process crawled datasheets"""
    try:
        logger.info(f"Processing {len(file_paths)} datasheets")
        
        # Extract parts using LlamaIndex
        extractor = LlamaIndexExtractor()
        parts = await extractor.process_datasheets(file_paths)
        
        # Enrich with Mouser API if requested
        if enrich_with_mouser:
            mouser_api = MouserAPI()
            parts = await mouser_api.batch_enrich_parts(parts)
        
        # Store in database
        upserted = upsert_records(parts)
        
        logger.info(f"Processed {len(parts)} parts, upserted {upserted}")
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
