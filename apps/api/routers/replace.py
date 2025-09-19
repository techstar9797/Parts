from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from packages.storage.db import get_part, iter_all_parts
from packages.matching.engine import FFFReplacementEngine, find_replacements
from packages.extraction.schemas_partsync import PartRecordPS, FFFWeights, HardGates

router = APIRouter()

class ReplacementRequest(BaseModel):
    mpn: str
    manufacturer: Optional[str] = None
    weights: Optional[FFFWeights] = None
    gates: Optional[HardGates] = None
    k: int = 10

class BOMReplacementRequest(BaseModel):
    bom_lines: List[Dict[str, Any]]
    weights: Optional[FFFWeights] = None
    gates: Optional[HardGates] = None
    k: int = 5

@router.post("/find-replacement")
async def find_replacement(request: ReplacementRequest):
    """Find replacement candidates for a single part using enhanced FFF engine"""
    try:
        # Get target part
        target_data = get_part(request.mpn)
        if not target_data:
            raise HTTPException(404, f"Part {request.mpn} not found")
        
        # Convert to PartRecordPS
        target = PartRecordPS(
            mpn=target_data.get("mpn", ""),
            manufacturer=target_data.get("manufacturer", ""),
            category=target_data.get("category", ""),
            package={"name": target_data.get("package", {}).get("name", ""), "pins": target_data.get("package", {}).get("pins")},
            v_range={"min": target_data.get("v_range", {}).get("min"), "max": target_data.get("v_range", {}).get("max"), "unit": "V"},
            temp_range_c={"min": target_data.get("temp_range_c", {}).get("min"), "max": target_data.get("temp_range_c", {}).get("max"), "unit": "°C"},
            attrs=target_data.get("attrs", {}),
            rohs=target_data.get("rohs"),
            lifecycle={"status": target_data.get("lifecycle", {}).get("status", "ACTIVE")} if target_data.get("lifecycle") else None,
            confidence=target_data.get("confidence")
        )
        
        # Get candidates
        candidates_data = list(iter_all_parts(exclude_mpn=request.mpn))
        candidates = []
        
        for candidate_data in candidates_data:
            candidate = PartRecordPS(
                mpn=candidate_data.get("mpn", ""),
                manufacturer=candidate_data.get("manufacturer", ""),
                category=candidate_data.get("category", ""),
                package={"name": candidate_data.get("package", {}).get("name", ""), "pins": candidate_data.get("package", {}).get("pins")},
                v_range={"min": candidate_data.get("v_range", {}).get("min"), "max": candidate_data.get("v_range", {}).get("max"), "unit": "V"},
                temp_range_c={"min": candidate_data.get("temp_range_c", {}).get("min"), "max": candidate_data.get("temp_range_c", {}).get("max"), "unit": "°C"},
                attrs=candidate_data.get("attrs", {}),
                rohs=candidate_data.get("rohs"),
                lifecycle={"status": candidate_data.get("lifecycle", {}).get("status", "ACTIVE")} if candidate_data.get("lifecycle") else None,
                confidence=candidate_data.get("confidence")
            )
            candidates.append(candidate)
        
        # Use enhanced FFF engine
        engine = FFFReplacementEngine(weights=request.weights, gates=request.gates)
        replacements = engine.find_replacements(target, candidates, k=request.k)
        
        return {
            "target": target.dict(),
            "replacements": [r.dict() for r in replacements],
            "total_candidates": len(candidates)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Replacement search failed: {str(e)}")

@router.post("/bom-replace")
async def bom_replace(request: BOMReplacementRequest):
    """Find replacements for all parts in a BOM"""
    try:
        results = []
        engine = FFFReplacementEngine(weights=request.weights, gates=request.gates)
        
        for line in request.bom_lines:
            mpn = line.get("mpn")
            if not mpn:
                results.append({
                    "line": line,
                    "error": "Missing MPN",
                    "replacements": []
                })
                continue
            
            # Get target part
            target_data = get_part(mpn)
            if not target_data:
                results.append({
                    "line": line,
                    "error": f"Part {mpn} not found",
                    "replacements": []
                })
                continue
            
            # Convert to PartRecordPS
            target = PartRecordPS(
                mpn=target_data.get("mpn", ""),
                manufacturer=target_data.get("manufacturer", ""),
                category=target_data.get("category", ""),
                package={"name": target_data.get("package", {}).get("name", ""), "pins": target_data.get("package", {}).get("pins")},
                v_range={"min": target_data.get("v_range", {}).get("min"), "max": target_data.get("v_range", {}).get("max"), "unit": "V"},
                temp_range_c={"min": target_data.get("temp_range_c", {}).get("min"), "max": target_data.get("temp_range_c", {}).get("max"), "unit": "°C"},
                attrs=target_data.get("attrs", {}),
                rohs=target_data.get("rohs"),
                lifecycle={"status": target_data.get("lifecycle", {}).get("status", "ACTIVE")} if target_data.get("lifecycle") else None,
                confidence=target_data.get("confidence")
            )
            
            # Get candidates
            candidates_data = list(iter_all_parts(exclude_mpn=mpn))
            candidates = []
            
            for candidate_data in candidates_data:
                candidate = PartRecordPS(
                    mpn=candidate_data.get("mpn", ""),
                    manufacturer=candidate_data.get("manufacturer", ""),
                    category=candidate_data.get("category", ""),
                    package={"name": candidate_data.get("package", {}).get("name", ""), "pins": candidate_data.get("package", {}).get("pins")},
                    v_range={"min": candidate_data.get("v_range", {}).get("min"), "max": candidate_data.get("v_range", {}).get("max"), "unit": "V"},
                    temp_range_c={"min": candidate_data.get("temp_range_c", {}).get("min"), "max": candidate_data.get("temp_range_c", {}).get("max"), "unit": "°C"},
                    attrs=candidate_data.get("attrs", {}),
                    rohs=candidate_data.get("rohs"),
                    lifecycle={"status": candidate_data.get("lifecycle", {}).get("status", "ACTIVE")} if candidate_data.get("lifecycle") else None,
                    confidence=candidate_data.get("confidence")
                )
                candidates.append(candidate)
            
            # Find replacements
            replacements = engine.find_replacements(target, candidates, k=request.k)
            
            results.append({
                "line": line,
                "target": target.dict(),
                "replacements": [r.dict() for r in replacements],
                "total_candidates": len(candidates)
            })
        
        return {
            "bom_results": results,
            "total_lines": len(request.bom_lines)
        }
        
    except Exception as e:
        raise HTTPException(500, f"BOM replacement failed: {str(e)}")

@router.get("/weights")
async def get_default_weights():
    """Get default FFF weights and gates configuration"""
    return {
        "weights": FFFWeights().dict(),
        "gates": HardGates().dict()
    }

@router.post("/weights")
async def update_weights(weights: FFFWeights, gates: HardGates):
    """Update FFF weights and gates configuration"""
    # In a real implementation, this would be stored in a database
    return {
        "message": "Weights and gates updated",
        "weights": weights.dict(),
        "gates": gates.dict()
    }

# Backward compatibility
@router.post("")
def find(body: Dict[str, Any]):
    """Backward compatible endpoint"""
    mpn = body.get("mpn")
    constraints = body.get("constraints", {})
    weights = body.get("weights", {})
    if not mpn:
        return {"error": "mpn is required"}
    target = get_part(mpn)
    candidates = list(iter_all_parts(exclude_mpn=mpn))
    results = find_replacements(target, candidates, weights=weights, constraints=constraints)
    return {"target": target, "candidates": results}
