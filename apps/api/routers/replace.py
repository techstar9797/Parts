from fastapi import APIRouter
from typing import Dict, Any
from packages.storage.db import get_part, iter_all_parts
from packages.matching.engine import find_replacements

router = APIRouter()

@router.post("")
def find(body: Dict[str, Any]):
    mpn = body.get("mpn")
    constraints = body.get("constraints", {})
    weights = body.get("weights", {})
    if not mpn:
        return {"error": "mpn is required"}
    target = get_part(mpn)
    candidates = list(iter_all_parts(exclude_mpn=mpn))
    results = find_replacements(target, candidates, weights=weights, constraints=constraints)
    return {"target": target, "candidates": results}
