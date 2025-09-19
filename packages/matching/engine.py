import os
import logging
from typing import Dict, Any, List, Optional
from ..extraction.schemas_partsync import PartRecordPS, ReplacementCandidate, FFFWeights, HardGates

logger = logging.getLogger(__name__)

class FFFReplacementEngine:
    def __init__(self, weights: Optional[FFFWeights] = None, gates: Optional[HardGates] = None):
        self.weights = weights or FFFWeights()
        self.gates = gates or HardGates()
    
    def _form_score(self, target: PartRecordPS, candidate: PartRecordPS) -> float:
        """Calculate form factor compatibility score"""
        score = 0.0
        factors = 0
        
        # Package name match (exact)
        if target.package.name and candidate.package.name:
            factors += 1
            if target.package.name == candidate.package.name:
                score += 1.0
            else:
                # Partial credit for similar packages
                if self._is_similar_package(target.package.name, candidate.package.name):
                    score += 0.7
        
        # Pin count match
        if target.package.pins and candidate.package.pins:
            factors += 1
            if target.package.pins == candidate.package.pins:
                score += 1.0
            else:
                # Partial credit for compatible pin counts
                if abs(target.package.pins - candidate.package.pins) <= 2:
                    score += 0.5
        
        # Pitch compatibility
        if target.package.pitch_mm and candidate.package.pitch_mm:
            factors += 1
            if target.package.pitch_mm == candidate.package.pitch_mm:
                score += 1.0
            else:
                # Partial credit for similar pitch
                pitch_diff = abs(target.package.pitch_mm - candidate.package.pitch_mm)
                if pitch_diff <= 0.1:  # 0.1mm tolerance
                    score += 0.8
                elif pitch_diff <= 0.2:
                    score += 0.5
        
        return score / factors if factors > 0 else 0.0
    
    def _is_similar_package(self, pkg1: str, pkg2: str) -> bool:
        """Check if packages are similar (e.g., SOIC-8 vs SOIC-8N)"""
        # Remove common suffixes and compare base package
        base1 = pkg1.split('-')[0] if '-' in pkg1 else pkg1
        base2 = pkg2.split('-')[0] if '-' in pkg2 else pkg2
        return base1 == base2
    
    def _fit_score(self, target: PartRecordPS, candidate: PartRecordPS) -> float:
        """Calculate electrical fit compatibility score"""
        score = 0.0
        factors = 0
        
        # Voltage range overlap
        v_overlap = self._range_overlap(target.v_range, candidate.v_range)
        if v_overlap is not None:
            factors += 1
            score += v_overlap
        
        # Temperature range overlap
        temp_overlap = self._range_overlap(target.temp_range_c, candidate.temp_range_c)
        if temp_overlap is not None:
            factors += 1
            score += temp_overlap
        
        # Power dissipation compatibility
        if target.power_dissipation_mw and candidate.power_dissipation_mw:
            factors += 1
            if candidate.power_dissipation_mw >= target.power_dissipation_mw:
                score += 1.0
            else:
                # Partial credit if close
                ratio = candidate.power_dissipation_mw / target.power_dissipation_mw
                score += max(0.0, ratio)
        
        # IO level compatibility
        if target.io_level and candidate.io_level:
            factors += 1
            if target.io_level == candidate.io_level:
                score += 1.0
            else:
                # Check for compatibility
                if self._is_compatible_io_level(target.io_level, candidate.io_level):
                    score += 0.8
        
        return score / factors if factors > 0 else 0.0
    
    def _range_overlap(self, range1, range2) -> Optional[float]:
        """Calculate overlap between two ranges (0.0 to 1.0)"""
        if not range1 or not range2:
            return None
        
        min1, max1 = range1.min, range1.max
        min2, max2 = range2.min, range2.max
        
        if min1 is None or max1 is None or min2 is None or max2 is None:
            return None
        
        # Calculate intersection
        intersection_min = max(min1, min2)
        intersection_max = min(max1, max2)
        
        if intersection_min > intersection_max:
            return 0.0  # No overlap
        
        intersection = intersection_max - intersection_min
        union = max(max1, max2) - min(min1, min2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_compatible_io_level(self, level1: str, level2: str) -> bool:
        """Check if IO levels are compatible"""
        # Simple compatibility check - can be enhanced
        compatible_groups = [
            {"3.3V", "3.3V CMOS", "3.3V TTL"},
            {"5V", "5V TTL", "5V CMOS"},
            {"1.8V", "1.8V CMOS"}
        ]
        
        for group in compatible_groups:
            if level1 in group and level2 in group:
                return True
        return False
    
    def _func_score(self, target: PartRecordPS, candidate: PartRecordPS) -> float:
        """Calculate functional compatibility score based on category-specific attributes"""
        if target.category != candidate.category:
            return 0.0  # Different categories are not functionally compatible
        
        target_attrs = target.attrs or {}
        candidate_attrs = candidate.attrs or {}
        
        if not target_attrs:
            return 0.5  # Neutral score if no target attributes
        
        score = 0.0
        factors = 0
        
        for key, target_value in target_attrs.items():
            if target_value is None:
                continue
                
            if key not in candidate_attrs or candidate_attrs[key] is None:
                continue
            
            factors += 1
            candidate_value = candidate_attrs[key]
            
            # Numeric comparison
            if isinstance(target_value, (int, float)) and isinstance(candidate_value, (int, float)):
                if candidate_value >= target_value:
                    score += 1.0
                else:
                    # Partial credit based on ratio
                    ratio = candidate_value / target_value
                    score += max(0.0, ratio)
            # String comparison
            elif isinstance(target_value, str) and isinstance(candidate_value, str):
                if target_value.lower() == candidate_value.lower():
                    score += 1.0
                else:
                    # Partial credit for similar strings
                    if target_value.lower() in candidate_value.lower() or candidate_value.lower() in target_value.lower():
                        score += 0.5
        
        return score / factors if factors > 0 else 0.0
    
    def _check_hard_gates(self, target: PartRecordPS, candidate: PartRecordPS) -> tuple[bool, List[str]]:
        """Check hard gates and return (passed, reasons)"""
        reasons = []
        
        # RoHS compliance gate
        if self.gates.rohs_match and target.rohs is True and candidate.rohs is False:
            return False, ["RoHS compliance mismatch"]
        
        # Lifecycle gate
        if target.lifecycle and candidate.lifecycle:
            lifecycle_order = ["ACTIVE", "PREVIEW", "NRND", "EOL", "OBSOLETE"]
            target_idx = lifecycle_order.index(target.lifecycle.status) if target.lifecycle.status in lifecycle_order else 0
            candidate_idx = lifecycle_order.index(candidate.lifecycle.status) if candidate.lifecycle.status in lifecycle_order else 0
            max_idx = lifecycle_order.index(self.gates.lifecycle_max) if self.gates.lifecycle_max in lifecycle_order else 2
            
            if candidate_idx > max_idx:
                return False, [f"Lifecycle status {candidate.lifecycle.status} exceeds maximum {self.gates.lifecycle_max}"]
        
        # Confidence gate
        if candidate.confidence and candidate.confidence < self.gates.min_confidence:
            return False, [f"Confidence {candidate.confidence:.2f} below minimum {self.gates.min_confidence}"]
        
        return True, reasons
    
    def find_replacements(self, 
                         target: PartRecordPS, 
                         candidates: List[PartRecordPS], 
                         k: int = 10) -> List[ReplacementCandidate]:
        """Find replacement candidates with FFF scoring"""
        results = []
        
        for candidate in candidates:
            # Skip self
            if candidate.mpn == target.mpn and candidate.manufacturer == target.manufacturer:
                continue
            
            # Check hard gates
            passed, gate_reasons = self._check_hard_gates(target, candidate)
            if not passed:
                continue
            
            # Calculate FFF scores
            form_score = self._form_score(target, candidate)
            fit_score = self._fit_score(target, candidate)
            func_score = self._func_score(target, candidate)
            
            # Calculate weighted total score
            total_score = (self.weights.form * form_score + 
                          self.weights.fit * fit_score + 
                          self.weights.func * func_score)
            
            # Generate reasons
            reasons = []
            if form_score > 0.8:
                reasons.append("Excellent form factor match")
            elif form_score > 0.5:
                reasons.append("Good form factor match")
            
            if fit_score > 0.8:
                reasons.append("Excellent electrical fit")
            elif fit_score > 0.5:
                reasons.append("Good electrical fit")
            
            if func_score > 0.8:
                reasons.append("Excellent functional match")
            elif func_score > 0.5:
                reasons.append("Good functional match")
            
            if not reasons:
                reasons.append("Basic compatibility")
            
            # Add gate reasons
            reasons.extend(gate_reasons)
            
            # Create replacement candidate
            replacement = ReplacementCandidate(
                mpn=candidate.mpn,
                manufacturer=candidate.manufacturer,
                score=round(total_score, 4),
                form_score=round(form_score, 4),
                fit_score=round(fit_score, 4),
                func_score=round(func_score, 4),
                reasons=reasons,
                provenance=candidate.provenance,
                part=candidate
            )
            
            results.append(replacement)
        
        # Sort by total score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

# Backward compatibility
def find_replacements(target: Dict[str, Any], candidates: List[Dict[str, Any]], weights=None, constraints=None):
    """Backward compatible function for old API"""
    engine = FFFReplacementEngine()
    
    # Convert dict to PartRecordPS (simplified)
    target_part = PartRecordPS(
        mpn=target.get("mpn", ""),
        manufacturer=target.get("manufacturer", ""),
        category=target.get("category", ""),
        package={"name": target.get("package", {}).get("name", ""), "pins": target.get("package", {}).get("pins")},
        v_range={"min": target.get("v_range", {}).get("min"), "max": target.get("v_range", {}).get("max"), "unit": "V"},
        temp_range_c={"min": target.get("temp_range_c", {}).get("min"), "max": target.get("temp_range_c", {}).get("max"), "unit": "°C"},
        attrs=target.get("attrs", {}),
        rohs=target.get("rohs"),
        lifecycle={"status": target.get("lifecycle", {}).get("status", "ACTIVE")} if target.get("lifecycle") else None
    )
    
    candidate_parts = []
    for c in candidates:
        candidate_parts.append(PartRecordPS(
            mpn=c.get("mpn", ""),
            manufacturer=c.get("manufacturer", ""),
            category=c.get("category", ""),
            package={"name": c.get("package", {}).get("name", ""), "pins": c.get("package", {}).get("pins")},
            v_range={"min": c.get("v_range", {}).get("min"), "max": c.get("v_range", {}).get("max"), "unit": "V"},
            temp_range_c={"min": c.get("temp_range_c", {}).get("min"), "max": c.get("temp_range_c", {}).get("max"), "unit": "°C"},
            attrs=c.get("attrs", {}),
            rohs=c.get("rohs"),
            lifecycle={"status": c.get("lifecycle", {}).get("status", "ACTIVE")} if c.get("lifecycle") else None
        ))
    
    replacements = engine.find_replacements(target_part, candidate_parts)
    
    # Convert back to old format
    results = []
    for r in replacements:
        results.append({
            "mpn": r.mpn,
            "manufacturer": r.manufacturer,
            "score": r.score,
            "reasons": r.reasons
        })
    
    return results
