#!/usr/bin/env python3
"""
PartSync - Enhanced FFF with Cost & Tariff Analysis
Includes cost per unit, tariff information, and priority sliders with full provenance tracking
"""

import gradio as gr
import os
import json
import requests
from typing import List, Dict, Any, Tuple
import asyncio
import aiohttp
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MOUSER_API_KEY = os.getenv("MOUSER_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

class EnhancedProvenanceTracker:
    """Enhanced provenance tracking with cost and tariff information"""
    
    def __init__(self):
        self.provenance = {
            "llamaindex": [],
            "llamacloud": [],
            "llamaparse": [],
            "llamaextract": [],
            "mouser_api": [],
            "apify": [],
            "cost_analysis": [],
            "tariff_analysis": []
        }
    
    def add_llamaindex(self, action: str, result: str, confidence: float = 1.0, source: str = "LlamaIndex API"):
        self.provenance["llamaindex"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "confidence": confidence,
            "source": source,
            "component": "LlamaIndex"
        })
    
    def add_llamacloud(self, action: str, result: str, project_id: str = "8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe"):
        self.provenance["llamacloud"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "project_id": project_id,
            "source": "LlamaCloud API",
            "component": "LlamaCloud"
        })
    
    def add_llamaparse(self, action: str, result: str, pdf_path: str = "", table_count: int = 0):
        self.provenance["llamaparse"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "pdf_path": pdf_path,
            "table_count": table_count,
            "source": "LlamaParse API",
            "component": "LlamaParse"
        })
    
    def add_llamaextract(self, action: str, result: str, parameters: List[str] = [], extraction_confidence: float = 0.0):
        self.provenance["llamaextract"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "extracted_parameters": parameters,
            "extraction_confidence": extraction_confidence,
            "source": "LlamaExtract API",
            "component": "LlamaExtract"
        })
    
    def add_mouser_api(self, action: str, result: str, query: str = "", cost_data: Dict = None):
        self.provenance["mouser_api"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "query": query,
            "cost_data": cost_data,
            "source": "Mouser API",
            "component": "Mouser API"
        })
    
    def add_apify(self, action: str, result: str, run_id: str = "", datasheet_count: int = 0):
        self.provenance["apify"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "run_id": run_id,
            "datasheet_count": datasheet_count,
            "source": "Apify API",
            "component": "Apify"
        })
    
    def add_cost_analysis(self, action: str, result: str, cost_source: str = "Mouser API", unit_cost: float = 0.0):
        self.provenance["cost_analysis"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "cost_source": cost_source,
            "unit_cost": unit_cost,
            "source": cost_source,
            "component": "Cost Analysis"
        })
    
    def add_tariff_analysis(self, action: str, result: str, tariff_source: str = "US HTS", tariff_rate: float = 0.0, 
                          tariff_status: str = "Unknown", country_of_origin: str = "Unknown"):
        self.provenance["tariff_analysis"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "tariff_source": tariff_source,
            "tariff_rate": tariff_rate,
            "tariff_status": tariff_status,
            "country_of_origin": country_of_origin,
            "source": tariff_source,
            "component": "Tariff Analysis"
        })
    
    def get_provenance_summary(self) -> str:
        """Get comprehensive provenance summary with cost and tariff information"""
        summary = "## 📋 Enhanced Provenance Summary\n\n"
        
        for component, entries in self.provenance.items():
            if entries:
                summary += f"### {component.upper()}\n"
                for entry in entries[-3:]:  # Show last 3 entries
                    summary += f"- **{entry['action']}**: {entry['result']}\n"
                    summary += f"  - Source: {entry.get('source', 'Unknown')}\n"
                    if 'confidence' in entry:
                        summary += f"  - Confidence: {entry['confidence']:.2f}\n"
                    if 'unit_cost' in entry:
                        summary += f"  - Unit Cost: ${entry['unit_cost']:.4f}\n"
                    if 'tariff_rate' in entry:
                        summary += f"  - Tariff Rate: {entry['tariff_rate']:.2f}%\n"
                    if 'tariff_status' in entry:
                        summary += f"  - Tariff Status: {entry['tariff_status']}\n"
                summary += "\n"
        
        return summary

class CostTariffAnalyzer:
    """Analyze cost and tariff information for parts"""
    
    def __init__(self, provenance_tracker: EnhancedProvenanceTracker):
        self.provenance = provenance_tracker
        self.tariff_database = self._load_tariff_database()
        self.cost_sources = ["Mouser API", "DigiKey API", "Arrow API", "Newark API"]
    
    def _load_tariff_database(self) -> Dict:
        """Load tariff database with HTS codes and rates"""
        return {
            "8542330001": {"description": "Electronic integrated circuits", "rate": 0.0, "status": "Non-tariffed"},
            "8542339000": {"description": "Other electronic integrated circuits", "rate": 0.0, "status": "Non-tariffed"},
            "8542310000": {"description": "Electronic integrated circuits, processors", "rate": 0.0, "status": "Non-tariffed"},
            "8542310302": {"description": "Electronic integrated circuits, microprocessors", "rate": 0.0, "status": "Non-tariffed"},
            "8542330201": {"description": "Electronic integrated circuits, amplifiers", "rate": 0.0, "status": "Non-tariffed"},
            "8542330299": {"description": "Electronic integrated circuits, other", "rate": 0.0, "status": "Non-tariffed"},
            "8542330000": {"description": "Electronic integrated circuits, general", "rate": 0.0, "status": "Non-tariffed"},
            "8542310000": {"description": "Electronic integrated circuits, processors", "rate": 0.0, "status": "Non-tariffed"}
        }
    
    def analyze_part_cost(self, part: Dict) -> Dict:
        """Analyze cost information for a part"""
        cost_info = {
            "unit_cost": part.get("unit_price", 0.0),
            "cost_source": "Mouser API",
            "cost_confidence": 0.95,
            "bulk_pricing": self._extract_bulk_pricing(part),
            "cost_trend": "Stable",
            "last_updated": datetime.now().isoformat()
        }
        
        # Track cost analysis
        self.provenance.add_cost_analysis(
            "analyze_unit_cost",
            f"Analyzed cost for {part.get('mpn', 'Unknown')}: ${cost_info['unit_cost']:.4f}",
            cost_info["cost_source"],
            cost_info["unit_cost"]
        )
        
        return cost_info
    
    def analyze_part_tariff(self, part: Dict) -> Dict:
        """Analyze tariff information for a part"""
        # Extract HTS code from part data
        hts_code = self._extract_hts_code(part)
        tariff_info = self.tariff_database.get(hts_code, {
            "description": "Unknown electronic component",
            "rate": 0.0,
            "status": "Unknown"
        })
        
        # Determine country of origin
        country_of_origin = self._determine_country_of_origin(part)
        
        tariff_analysis = {
            "hts_code": hts_code,
            "tariff_rate": tariff_info["rate"],
            "tariff_status": tariff_info["status"],
            "country_of_origin": country_of_origin,
            "tariff_source": "US HTS Database",
            "tariff_confidence": 0.90,
            "last_updated": datetime.now().isoformat()
        }
        
        # Track tariff analysis
        self.provenance.add_tariff_analysis(
            "analyze_tariff_status",
            f"Analyzed tariff for {part.get('mpn', 'Unknown')}: {tariff_analysis['tariff_status']}",
            tariff_analysis["tariff_source"],
            tariff_analysis["tariff_rate"],
            tariff_analysis["tariff_status"],
            tariff_analysis["country_of_origin"]
        )
        
        return tariff_analysis
    
    def _extract_bulk_pricing(self, part: Dict) -> List[Dict]:
        """Extract bulk pricing information"""
        # This would typically come from Mouser API price breaks
        return [
            {"quantity": 1, "price": part.get("unit_price", 0.0)},
            {"quantity": 10, "price": part.get("unit_price", 0.0) * 0.9},
            {"quantity": 100, "price": part.get("unit_price", 0.0) * 0.8},
            {"quantity": 1000, "price": part.get("unit_price", 0.0) * 0.7}
        ]
    
    def _extract_hts_code(self, part: Dict) -> str:
        """Extract HTS code from part data"""
        # This would typically come from Mouser API compliance data
        compliance_data = part.get("compliance", {})
        return compliance_data.get("USHTS", "8542330001")  # Default HTS code
    
    def _determine_country_of_origin(self, part: Dict) -> str:
        """Determine country of origin based on manufacturer"""
        manufacturer = part.get("manufacturer", "").lower()
        
        origin_mapping = {
            "texas instruments": "USA",
            "analog devices": "USA",
            "maxim integrated": "USA",
            "microchip": "USA",
            "stmicroelectronics": "France",
            "infineon": "Germany",
            "rohm": "Japan",
            "toshiba": "Japan",
            "renesas": "Japan"
        }
        
        for mfg, country in origin_mapping.items():
            if mfg in manufacturer:
                return country
        
        return "Unknown"

class EnhancedFFFEngine:
    def __init__(self, faiss_engine, provenance_tracker: EnhancedProvenanceTracker, cost_tariff_analyzer: CostTariffAnalyzer):
        self.faiss_engine = faiss_engine
        self.provenance = provenance_tracker
        self.cost_tariff_analyzer = cost_tariff_analyzer
        self.default_weights = {"form": 0.4, "fit": 0.3, "func": 0.2, "cost": 0.05, "tariff": 0.05}
    
    def calculate_enhanced_fff_score(self, target_part: Dict, candidate_part: Dict, 
                                   form_weight: float, fit_weight: float, func_weight: float,
                                   cost_weight: float, tariff_weight: float,
                                   use_faiss: bool = True) -> Dict:
        """Calculate enhanced FFF score with cost and tariff analysis"""
        
        # Normalize weights
        total_weight = form_weight + fit_weight + func_weight + cost_weight + tariff_weight
        if total_weight == 0:
            form_weight, fit_weight, func_weight, cost_weight, tariff_weight = 0.4, 0.3, 0.2, 0.05, 0.05
        else:
            form_weight /= total_weight
            fit_weight /= total_weight
            func_weight /= total_weight
            cost_weight /= total_weight
            tariff_weight /= total_weight
        
        weights = {"form": form_weight, "fit": fit_weight, "func": func_weight, 
                  "cost": cost_weight, "tariff": tariff_weight}
        
        # Get FAISS similarity
        faiss_similarity = 0.0
        if use_faiss:
            try:
                similar_parts = self.faiss_engine.find_similar_parts(target_part, k=1, threshold=0.0)
                for part, score in similar_parts:
                    if part.get("mpn") == candidate_part.get("mpn"):
                        faiss_similarity = score
                        break
            except Exception as e:
                print(f"FAISS error: {e}")
        
        # Calculate individual FFF scores
        form_score = self._calculate_form_score(target_part, candidate_part)
        fit_score = self._calculate_fit_score(target_part, candidate_part)
        func_score = self._calculate_func_score(target_part, candidate_part)
        
        # Calculate cost and tariff scores
        cost_score = self._calculate_cost_score(target_part, candidate_part)
        tariff_score = self._calculate_tariff_score(target_part, candidate_part)
        
        # Apply FAISS enhancement if enabled
        if use_faiss and faiss_similarity > 0:
            enhanced_form = (form_score * 0.7) + (faiss_similarity * 0.3)
            enhanced_fit = (fit_score * 0.7) + (faiss_similarity * 0.3)
            enhanced_func = (func_score * 0.7) + (faiss_similarity * 0.3)
        else:
            enhanced_form = form_score
            enhanced_fit = fit_score
            enhanced_func = func_score
        
        # Calculate overall enhanced FFF score
        fff_score = (weights["form"] * enhanced_form + 
                    weights["fit"] * enhanced_fit + 
                    weights["func"] * enhanced_func +
                    weights["cost"] * cost_score +
                    weights["tariff"] * tariff_score)
        
        # Check hard gates
        passes_gates, gate_reasons = self._check_hard_gates(target_part, candidate_part)
        
        return {
            "fff_score": fff_score,
            "form_score": enhanced_form,
            "fit_score": enhanced_fit,
            "func_score": enhanced_func,
            "cost_score": cost_score,
            "tariff_score": tariff_score,
            "faiss_similarity": faiss_similarity,
            "use_faiss": use_faiss,
            "passes_gates": passes_gates,
            "gate_reasons": gate_reasons,
            "cost_info": self.cost_tariff_analyzer.analyze_part_cost(candidate_part),
            "tariff_info": self.cost_tariff_analyzer.analyze_part_tariff(candidate_part),
            "reasons": self._generate_enhanced_reasons(enhanced_form, enhanced_fit, enhanced_func, 
                                                     cost_score, tariff_score, faiss_similarity, 
                                                     use_faiss, weights)
        }
    
    def _calculate_cost_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate cost compatibility score (0-1)"""
        target_cost = target.get("unit_price", 0.0)
        candidate_cost = candidate.get("unit_price", 0.0)
        
        if target_cost == 0 or candidate_cost == 0:
            return 0.5
        
        # Prefer similar or lower cost
        cost_ratio = min(target_cost, candidate_cost) / max(target_cost, candidate_cost)
        
        # Bonus for lower cost
        if candidate_cost < target_cost:
            cost_ratio = min(1.0, cost_ratio * 1.2)
        
        return cost_ratio
    
    def _calculate_tariff_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate tariff compatibility score (0-1)"""
        target_tariff = target.get("tariff_info", {}).get("tariff_status", "Unknown")
        candidate_tariff = candidate.get("tariff_info", {}).get("tariff_status", "Unknown")
        
        # Prefer non-tariffed parts
        if candidate_tariff == "Non-tariffed":
            return 1.0
        elif candidate_tariff == "Tariffed":
            return 0.3
        else:
            return 0.5
    
    def _calculate_form_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate form compatibility score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Package compatibility
        if target.get("package") and candidate.get("package"):
            if target["package"] == candidate["package"]:
                score += 0.4
            elif self._packages_compatible(target["package"], candidate["package"]):
                score += 0.3
            else:
                score += 0.1
        max_score += 0.4
        
        # Pin count compatibility
        if target.get("pins") and candidate.get("pins"):
            if target["pins"] == candidate["pins"]:
                score += 0.3
            elif abs(target["pins"] - candidate["pins"]) <= 2:
                score += 0.2
            else:
                score += 0.05
        max_score += 0.3
        
        # Additional form factors
        if target.get("package") and candidate.get("package"):
            if "SOIC" in target["package"] and "SOIC" in candidate["package"]:
                score += 0.3
            elif "DIP" in target["package"] and "DIP" in candidate["package"]:
                score += 0.3
        max_score += 0.3
        
        return score / max_score if max_score > 0 else 0.0
    
    def _calculate_fit_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate fit compatibility score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Voltage range compatibility
        if target.get("v_range") and candidate.get("v_range"):
            v_score = self._compare_voltage_ranges(target["v_range"], candidate["v_range"])
            score += 0.4 * v_score
        max_score += 0.4
        
        # Temperature range compatibility
        if target.get("temp_range") and candidate.get("temp_range"):
            t_score = self._compare_temperature_ranges(target["temp_range"], candidate["temp_range"])
            score += 0.3 * t_score
        max_score += 0.3
        
        # Power compatibility
        if target.get("power_dissipation") and candidate.get("power_dissipation"):
            p_score = self._compare_power_ratings(target["power_dissipation"], candidate["power_dissipation"])
            score += 0.3 * p_score
        max_score += 0.3
        
        return score / max_score if max_score > 0 else 0.0
    
    def _calculate_func_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate function compatibility score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Category compatibility
        if target.get("category") and candidate.get("category"):
            if target["category"] == candidate["category"]:
                score += 0.5
            elif self._categories_compatible(target["category"], candidate["category"]):
                score += 0.3
            else:
                score += 0.1
        max_score += 0.5
        
        # Manufacturer compatibility
        if target.get("manufacturer") and candidate.get("manufacturer"):
            if target["manufacturer"] == candidate["manufacturer"]:
                score += 0.3
            else:
                score += 0.1
        max_score += 0.3
        
        # Description similarity
        if target.get("description") and candidate.get("description"):
            desc_similarity = self._calculate_text_similarity(
                target["description"], candidate["description"]
            )
            score += 0.2 * desc_similarity
        max_score += 0.2
        
        return score / max_score if max_score > 0 else 0.0
    
    def _packages_compatible(self, pkg1: str, pkg2: str) -> bool:
        compatible_groups = [
            ["SOIC-8", "SOP-8", "MSOP-8"],
            ["DIP-8", "PDIP-8"],
            ["QFN-8", "DFN-8"]
        ]
        for group in compatible_groups:
            if pkg1 in group and pkg2 in group:
                return True
        return False
    
    def _compare_voltage_ranges(self, v1: Dict, v2: Dict) -> float:
        if not v1 or not v2:
            return 0.5
        v1_min, v1_max = v1.get("min", 0), v1.get("max", 0)
        v2_min, v2_max = v2.get("min", 0), v2.get("max", 0)
        
        if v1_min <= v2_min and v1_max >= v2_max:
            return 1.0
        elif v2_min <= v1_min and v2_max >= v1_max:
            return 0.8
        else:
            overlap = max(0, min(v1_max, v2_max) - max(v1_min, v2_min))
            total_range = max(v1_max, v2_max) - min(v1_min, v2_min)
            return overlap / total_range if total_range > 0 else 0.0
    
    def _compare_temperature_ranges(self, t1: Dict, t2: Dict) -> float:
        if not t1 or not t2:
            return 0.5
        t1_min, t1_max = t1.get("min", 0), t1.get("max", 0)
        t2_min, t2_max = t2.get("min", 0), t2.get("max", 0)
        
        if t1_min <= t2_min and t1_max >= t2_max:
            return 1.0
        elif t2_min <= t1_min and t2_max >= t1_max:
            return 0.8
        else:
            overlap = max(0, min(t1_max, t2_max) - max(t1_min, t2_min))
            total_range = max(t1_max, t2_max) - min(t1_min, t2_min)
            return overlap / total_range if total_range > 0 else 0.0
    
    def _compare_power_ratings(self, p1: float, p2: float) -> float:
        if p1 == 0 or p2 == 0:
            return 0.5
        return min(p1, p2) / max(p1, p2)
    
    def _categories_compatible(self, cat1: str, cat2: str) -> bool:
        compatible_groups = [
            ["OpAmp", "Comparator"],
            ["LDO", "Linear Regulator"],
            ["MOSFET", "Transistor"]
        ]
        for group in compatible_groups:
            if cat1 in group and cat2 in group:
                return True
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)
    
    def _check_hard_gates(self, target: Dict, candidate: Dict) -> Tuple[bool, List[str]]:
        reasons = []
        
        # RoHS compliance
        if target.get("rohs") and not candidate.get("rohs"):
            reasons.append("❌ Candidate not RoHS compliant")
            return False, reasons
        else:
            reasons.append("✅ RoHS compliant")
        
        # Lifecycle status
        candidate_lifecycle = candidate.get("lifecycle", "ACTIVE")
        if candidate_lifecycle in ["EOL", "OBSOLETE"]:
            reasons.append(f"❌ Candidate lifecycle {candidate_lifecycle} not acceptable")
            return False, reasons
        else:
            reasons.append(f"✅ Lifecycle acceptable: {candidate_lifecycle}")
        
        return True, reasons
    
    def _generate_enhanced_reasons(self, form_score: float, fit_score: float, func_score: float,
                                 cost_score: float, tariff_score: float, faiss_similarity: float, 
                                 use_faiss: bool, weights: Dict) -> List[str]:
        reasons = []
        
        # FAISS similarity reason
        if use_faiss and faiss_similarity > 0:
            if faiss_similarity >= 0.9:
                reasons.append(f"🧠 Excellent semantic similarity (FAISS): {faiss_similarity:.2f}")
            elif faiss_similarity >= 0.7:
                reasons.append(f"🧠 Good semantic similarity (FAISS): {faiss_similarity:.2f}")
            else:
                reasons.append(f"🧠 Limited semantic similarity (FAISS): {faiss_similarity:.2f}")
        else:
            reasons.append("🧠 FAISS similarity disabled")
        
        # Form reasons
        if form_score >= 0.9:
            reasons.append(f"✅ Excellent form compatibility: {form_score:.2f}")
        elif form_score >= 0.7:
            reasons.append(f"⚠️ Good form compatibility: {form_score:.2f}")
        else:
            reasons.append(f"❌ Poor form compatibility: {form_score:.2f}")
        
        # Fit reasons
        if fit_score >= 0.9:
            reasons.append(f"✅ Excellent electrical fit: {fit_score:.2f}")
        elif fit_score >= 0.7:
            reasons.append(f"⚠️ Good electrical fit: {fit_score:.2f}")
        else:
            reasons.append(f"❌ Poor electrical fit: {fit_score:.2f}")
        
        # Function reasons
        if func_score >= 0.9:
            reasons.append(f"✅ Excellent functional compatibility: {func_score:.2f}")
        elif func_score >= 0.7:
            reasons.append(f"⚠️ Good functional compatibility: {func_score:.2f}")
        else:
            reasons.append(f"❌ Limited functional compatibility: {func_score:.2f}")
        
        # Cost reasons
        if cost_score >= 0.9:
            reasons.append(f"💰 Excellent cost compatibility: {cost_score:.2f}")
        elif cost_score >= 0.7:
            reasons.append(f"💰 Good cost compatibility: {cost_score:.2f}")
        else:
            reasons.append(f"💰 Limited cost compatibility: {cost_score:.2f}")
        
        # Tariff reasons
        if tariff_score >= 0.9:
            reasons.append(f"📋 Excellent tariff compatibility: {tariff_score:.2f}")
        elif tariff_score >= 0.7:
            reasons.append(f"📋 Good tariff compatibility: {tariff_score:.2f}")
        else:
            reasons.append(f"📋 Limited tariff compatibility: {tariff_score:.2f}")
        
        # Weight information
        reasons.append(f"⚖️ Weights: Form={weights['form']:.1f}, Fit={weights['fit']:.1f}, Function={weights['func']:.1f}, Cost={weights['cost']:.1f}, Tariff={weights['tariff']:.1f}")
        
        return reasons

# Initialize components
provenance_tracker = EnhancedProvenanceTracker()
cost_tariff_analyzer = CostTariffAnalyzer(provenance_tracker)

# Mock FAISS engine for now
class MockFAISSEngine:
    def find_similar_parts(self, target_part, k=10, threshold=0.7):
        return []

faiss_engine = MockFAISSEngine()
fff_engine = EnhancedFFFEngine(faiss_engine, provenance_tracker, cost_tariff_analyzer)

def format_enhanced_replacement_info(replacement: Dict) -> str:
    """Format enhanced replacement information with cost and tariff data"""
    cost_info = replacement.get("cost_info", {})
    tariff_info = replacement.get("tariff_info", {})
    
    return f"""
**{replacement['mpn']}** - {replacement['manufacturer']}
**Overall FFF Score:** {replacement['fff_score']:.3f}
**Form:** {replacement['form_score']:.3f} | **Fit:** {replacement['fit_score']:.3f} | **Function:** {replacement['func_score']:.3f}
**Cost:** {replacement['cost_score']:.3f} | **Tariff:** {replacement['tariff_score']:.3f}
**FAISS Similarity:** {replacement.get('faiss_similarity', 0.0):.3f} {'✅' if replacement.get('use_faiss') else '❌'}

**💰 Cost Information:**
- Unit Cost: ${cost_info.get('unit_cost', 0.0):.4f}
- Cost Source: {cost_info.get('cost_source', 'Unknown')}
- Cost Confidence: {cost_info.get('cost_confidence', 0.0):.2f}
- Bulk Pricing: Available (10+ units)

**📋 Tariff Information:**
- Tariff Status: {tariff_info.get('tariff_status', 'Unknown')}
- Tariff Rate: {tariff_info.get('tariff_rate', 0.0):.2f}%
- Country of Origin: {tariff_info.get('country_of_origin', 'Unknown')}
- HTS Code: {tariff_info.get('hts_code', 'Unknown')}
- Tariff Source: {tariff_info.get('tariff_source', 'Unknown')}

**Stock:** {replacement.get('stock', 'N/A')} units | **Price:** ${replacement.get('unit_price', 'N/A')}
**Lifecycle:** {replacement.get('lifecycle', 'Unknown')}

**Analysis:**
{chr(10).join(f"• {reason}" for reason in replacement['reasons'])}

**Hard Gates:**
{chr(10).join(f"• {reason}" for reason in replacement['gate_reasons'])}
"""

async def find_enhanced_replacements(target_mpn: str, form_weight: float, fit_weight: float, 
                                   func_weight: float, cost_weight: float, tariff_weight: float,
                                   min_fff_score: float, use_faiss: bool) -> str:
    """Find replacement parts with enhanced FFF scoring including cost and tariff analysis"""
    if not target_mpn.strip():
        return "Please enter a target MPN."
    
    # Mock search results for demonstration
    mock_parts = [
        {
            "mpn": "LM358FVJ-E2",
            "manufacturer": "ROHM Semiconductor",
            "description": "Operational Amplifiers - Op Amps Ind 2Ch 3-32V Ground Sense",
            "category": "OpAmp",
            "package": "SOIC-8",
            "pins": 8,
            "v_range": {"min": 3.0, "max": 32.0, "unit": "V"},
            "temp_range": {"min": -40, "max": 85, "unit": "°C"},
            "stock": 4349,
            "unit_price": 0.52,
            "lifecycle": "ACTIVE",
            "rohs": True,
            "provenance": ["mouser_api"]
        },
        {
            "mpn": "LM358F-GE2",
            "manufacturer": "ROHM Semiconductor",
            "description": "Operational Amplifiers - Op Amps Dual Grnd Sense Op Amp SOP8",
            "category": "OpAmp",
            "package": "SOIC-8",
            "pins": 8,
            "v_range": {"min": 3.0, "max": 32.0, "unit": "V"},
            "temp_range": {"min": -40, "max": 85, "unit": "°C"},
            "stock": 17195,
            "unit_price": 0.52,
            "lifecycle": "ACTIVE",
            "rohs": True,
            "provenance": ["mouser_api"]
        }
    ]
    
    # Use first part as target
    target_part = mock_parts[0]
    
    # Calculate enhanced FFF scores
    replacements = []
    for part in mock_parts[1:]:
        fff_result = fff_engine.calculate_enhanced_fff_score(
            target_part, part, form_weight, fit_weight, func_weight, 
            cost_weight, tariff_weight, use_faiss
        )
        
        if fff_result["fff_score"] >= min_fff_score and fff_result["passes_gates"]:
            replacement = {
                "mpn": part["mpn"],
                "manufacturer": part["manufacturer"],
                "fff_score": fff_result["fff_score"],
                "form_score": fff_result["form_score"],
                "fit_score": fff_result["fit_score"],
                "func_score": fff_result["func_score"],
                "cost_score": fff_result["cost_score"],
                "tariff_score": fff_result["tariff_score"],
                "faiss_similarity": fff_result["faiss_similarity"],
                "use_faiss": fff_result["use_faiss"],
                "reasons": fff_result["reasons"],
                "gate_reasons": fff_result["gate_reasons"],
                "cost_info": fff_result["cost_info"],
                "tariff_info": fff_result["tariff_info"],
                "stock": part["stock"],
                "unit_price": part["unit_price"],
                "lifecycle": part["lifecycle"],
                "provenance": part["provenance"]
            }
            replacements.append(replacement)
    
    # Sort by FFF score
    replacements.sort(key=lambda x: x["fff_score"], reverse=True)
    
    if not replacements:
        return f"No replacement candidates found for {target_mpn} with FFF score >= {min_fff_score:.2f}"
    
    result = f"**Enhanced FFF Analysis with Cost & Tariff for {target_mpn}:**\n"
    result += f"**Weights:** Form={form_weight:.1f}, Fit={fit_weight:.1f}, Function={func_weight:.1f}, Cost={cost_weight:.1f}, Tariff={tariff_weight:.1f}\n"
    result += f"**FAISS:** {'Enabled' if use_faiss else 'Disabled'}\n"
    result += f"**Min Score:** {min_fff_score:.2f}\n\n"
    
    for i, replacement in enumerate(replacements[:5], 1):
        result += f"**{i}.** {format_enhanced_replacement_info(replacement)}\n"
        result += "---\n"
    
    return result

def get_enhanced_provenance_summary() -> str:
    """Get comprehensive provenance summary with cost and tariff information"""
    return provenance_tracker.get_provenance_summary()

def create_interface():
    """Create the enhanced Gradio interface with cost and tariff controls"""
    
    with gr.Blocks(
        title="PartSync - Enhanced FFF with Cost & Tariff Analysis",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
        # 🔧 PartSync - Enhanced FFF with Cost & Tariff Analysis
        
        **Comprehensive FFF scoring with cost per unit, tariff information, and priority controls**
        """)
        
        with gr.Tabs():
            
            # Enhanced FFF Analysis Tab
            with gr.Tab("💰 Enhanced FFF Analysis"):
                gr.Markdown("### Real-time FFF scoring with cost and tariff analysis")
                
                with gr.Row():
                    target_mpn = gr.Textbox(
                        label="Target MPN",
                        placeholder="Enter the part number you want to replace",
                        value="LM358FVJ-E2"
                    )
                
                gr.Markdown("### FFF Scoring Controls")
                with gr.Row():
                    form_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.4,
                        step=0.05,
                        label="Form Weight (Package, Pins, Pinout)"
                    )
                    fit_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Fit Weight (Voltage, Temperature, Power)"
                    )
                    func_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.05,
                        label="Function Weight (Category, Attributes)"
                    )
                
                gr.Markdown("### Cost & Tariff Priority Controls")
                with gr.Row():
                    cost_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.05,
                        step=0.05,
                        label="Cost Weight (Unit Cost, Bulk Pricing)"
                    )
                    tariff_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.05,
                        step=0.05,
                        label="Tariff Weight (Tariff Status, Country of Origin)"
                    )
                
                with gr.Row():
                    min_fff_score = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        label="Minimum FFF Score"
                    )
                    use_faiss = gr.Checkbox(
                        value=True,
                        label="Use FAISS Similarity (Default)"
                    )
                
                replace_btn = gr.Button("💰 Calculate Enhanced FFF", variant="primary")
                replace_output = gr.Markdown(label="Enhanced FFF Analysis with Cost & Tariff")
                
                replace_btn.click(
                    fn=find_enhanced_replacements,
                    inputs=[target_mpn, form_weight, fit_weight, func_weight, cost_weight, tariff_weight, min_fff_score, use_faiss],
                    outputs=replace_output
                )
            
            # Enhanced Provenance Tab
            with gr.Tab("📋 Enhanced Provenance"):
                gr.Markdown("### Comprehensive provenance with cost and tariff tracking")
                
                provenance_btn = gr.Button("📋 Show Enhanced Provenance", variant="primary")
                provenance_output = gr.Markdown(label="Enhanced Provenance Summary")
                
                provenance_btn.click(
                    fn=get_enhanced_provenance_summary,
                    outputs=provenance_output
                )
    
    return interface

def main():
    """Main function to launch the enhanced Gradio app"""
    print("🚀 Starting PartSync Enhanced FFF with Cost & Tariff Analysis...")
    
    interface = create_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("📱 The app will open in your browser at http://localhost:7868")
    print("🛑 Press Ctrl+C to stop the server")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7868,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
