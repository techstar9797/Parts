from fastapi import APIRouter, Query
from typing import Optional, List, Dict, Any
from packages.storage.db import search_parts, get_part

router = APIRouter()

@router.get("")
def list_parts(q: Optional[str] = None, category: Optional[str] = None, manufacturer: Optional[str] = None, limit: int = 20, offset: int = 0):
    return search_parts(q=q, category=category, manufacturer=manufacturer, limit=limit, offset=offset)

@router.get("/{mpn}")
def fetch_part(mpn: str):
    return get_part(mpn)
