import os
import httpx
import asyncio
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "aYG0l9s7dbB7j3gbS")
TOKEN = os.getenv("APIFY_TOKEN")

class MouserCrawler:
    def __init__(self):
        if not TOKEN:
            raise ValueError("APIFY_TOKEN environment variable is required")
        self.token = TOKEN
        self.actor_id = ACTOR_ID
    
    async def run_actor(self, 
                       query: Optional[str] = None,
                       mpns: Optional[List[str]] = None,
                       manufacturers: Optional[List[str]] = None,
                       categories: Optional[List[str]] = None,
                       max_pages: int = 5) -> Dict[str, Any]:
        """Trigger Apify actor run for Mouser crawling"""
        url = f"{APIFY_BASE}/acts/{self.actor_id}/runs?token={self.token}"
        
        payload = {
            "maxPages": max_pages,
            "search": query or "",
            "mpns": mpns or [],
            "manufacturers": manufacturers or [],
            "categories": categories or []
        }
        
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json().get("data", {})
            except httpx.HTTPError as e:
                logger.error(f"Failed to trigger Apify actor: {e}")
                raise
    
    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a running Apify actor"""
        url = f"{APIFY_BASE}/actor-runs/{run_id}?token={self.token}"
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json().get("data", {})
            except httpx.HTTPError as e:
                logger.error(f"Failed to get run status: {e}")
                raise
    
    async def collect_results(self, run_id: str) -> List[Dict[str, Any]]:
        """Collect results from completed Apify run"""
        url = f"{APIFY_BASE}/actor-runs/{run_id}/dataset/items?token={self.token}&format=json"
        
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Failed to collect results: {e}")
                raise
    
    async def download_datasheets(self, 
                                items: List[Dict[str, Any]], 
                                output_dir: str = "data/samples") -> List[str]:
        """Download datasheet PDFs from collected items"""
        os.makedirs(output_dir, exist_ok=True)
        downloaded_files = []
        
        # Extract unique datasheet URLs
        datasheet_urls = set()
        for item in items:
            if datasheet_url := item.get("datasheetUrl"):
                datasheet_urls.add(datasheet_url)
        
        logger.info(f"Found {len(datasheet_urls)} unique datasheet URLs")
        
        async with httpx.AsyncClient(timeout=300) as client:
            for url in datasheet_urls:
                try:
                    # Extract filename from URL
                    parsed = urlparse(url)
                    filename = os.path.basename(parsed.path)
                    if not filename.endswith('.pdf'):
                        filename += '.pdf'
                    
                    filepath = os.path.join(output_dir, filename)
                    
                    # Skip if already exists
                    if os.path.exists(filepath):
                        downloaded_files.append(filepath)
                        continue
                    
                    # Download PDF
                    response = await client.get(url)
                    response.raise_for_status()
                    
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    
                    downloaded_files.append(filepath)
                    logger.info(f"Downloaded: {filename}")
                    
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
                    continue
        
        return downloaded_files
    
    async def crawl_and_download(self, 
                               query: str,
                               max_pages: int = 5,
                               output_dir: str = "data/samples") -> Dict[str, Any]:
        """Complete workflow: crawl Mouser, download datasheets"""
        # Step 1: Trigger Apify run
        run_data = await self.run_actor(query=query, max_pages=max_pages)
        run_id = run_data.get("id")
        
        if not run_id:
            raise ValueError("Failed to get run ID from Apify")
        
        logger.info(f"Started Apify run: {run_id}")
        
        # Step 2: Wait for completion (polling)
        while True:
            status = await self.get_run_status(run_id)
            status_name = status.get("status")
            
            if status_name == "SUCCEEDED":
                break
            elif status_name in ["FAILED", "ABORTED", "TIMED-OUT"]:
                raise RuntimeError(f"Apify run failed with status: {status_name}")
            
            logger.info(f"Run status: {status_name}, waiting...")
            await asyncio.sleep(10)
        
        # Step 3: Collect results
        items = await self.collect_results(run_id)
        logger.info(f"Collected {len(items)} items from Apify")
        
        # Step 4: Download datasheets
        downloaded_files = await self.download_datasheets(items, output_dir)
        logger.info(f"Downloaded {len(downloaded_files)} datasheet PDFs")
        
        return {
            "run_id": run_id,
            "items": items,
            "downloaded_files": downloaded_files,
            "total_items": len(items),
            "total_datasheets": len(downloaded_files)
        }

# Convenience function for direct usage
async def crawl_mouser(query: str, max_pages: int = 5) -> Dict[str, Any]:
    """Convenience function to crawl Mouser and download datasheets"""
    crawler = MouserCrawler()
    return await crawler.crawl_and_download(query, max_pages)
