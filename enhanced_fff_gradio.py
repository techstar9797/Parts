#!/usr/bin/env python3
"""
PartSync - Enhanced FFF Scoring with FAISS Similarity Analysis
Implements Form-Fit-Function analysis with FAISS vector similarity for better part matching
"""

import gradio as gr
import os
import json
import requests
from typing import List, Dict, Any, Tuple
import asyncio
import aiohttp
from datetime import datetime
import tempfile
import shutil
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

print(f"Environment loaded:")
print(f"OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
print(f"Mouser API Key: {'✅ Set' if MOUSER_API_KEY else '❌ Not set'}")
print(f"LlamaCloud API Key: {'✅ Set' if LLAMA_CLOUD_API_KEY else '❌ Not set'}")
print(f"Apify API Token: {'✅ Set' if APIFY_API_TOKEN else '❌ Not set'}")

class FAISSSimilarityEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize FAISS similarity engine with sentence transformer"""
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.part_metadata = []
        self.is_trained = False
        
        # Create FAISS index directory
        self.index_dir = Path("data/faiss_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index if available"""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    self.part_metadata = pickle.load(f)
                self.is_trained = True
                print(f"✅ Loaded existing FAISS index with {len(self.part_metadata)} parts")
            except Exception as e:
                print(f"⚠️ Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.part_metadata = []
        self.is_trained = False
        print("🆕 Created new FAISS index")
    
    def add_parts(self, parts: List[Dict]):
        """Add parts to the FAISS index"""
        if not parts:
            return
        
        # Create text representations for each part
        part_texts = []
        for part in parts:
            text = self._create_part_text(part)
            part_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(part_texts, convert_to_tensor=False)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for part in parts:
            self.part_metadata.append(part)
        
        self.is_trained = True
        print(f"✅ Added {len(parts)} parts to FAISS index (total: {len(self.part_metadata)})")
    
    def _create_part_text(self, part: Dict) -> str:
        """Create text representation of a part for embedding"""
        text_parts = [
            part.get("mpn", ""),
            part.get("manufacturer", ""),
            part.get("description", ""),
            part.get("category", ""),
            part.get("package", ""),
            f"pins: {part.get('pins', '')}",
            f"voltage: {part.get('v_range', {}).get('min', '')}-{part.get('v_range', {}).get('max', '')}V",
            f"temperature: {part.get('temp_range', {}).get('min', '')}-{part.get('temp_range', {}).get('max', '')}°C",
            f"power: {part.get('power_dissipation', '')}mW",
            f"tolerance: {part.get('tolerance_pct', '')}%",
            f"lifecycle: {part.get('lifecycle', '')}",
            f"rohs: {part.get('rohs', '')}"
        ]
        
        return " ".join([str(p) for p in text_parts if p])
    
    def find_similar_parts(self, target_part: Dict, k: int = 10, threshold: float = 0.7) -> List[Tuple[Dict, float]]:
        """Find similar parts using FAISS similarity search"""
        if not self.is_trained or len(self.part_metadata) == 0:
            return []
        
        # Create text representation for target part
        target_text = self._create_part_text(target_part)
        
        # Generate embedding
        target_embedding = self.model.encode([target_text], convert_to_tensor=False)
        faiss.normalize_L2(target_embedding)
        
        # Search for similar parts
        scores, indices = self.index.search(target_embedding.astype('float32'), k)
        
        # Filter by threshold and return results
        similar_parts = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.part_metadata):
                similar_parts.append((self.part_metadata[idx], float(score)))
        
        return similar_parts
    
    def save_index(self):
        """Save FAISS index and metadata"""
        if not self.is_trained:
            return
        
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.pkl"
        
        try:
            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.part_metadata, f)
            print(f"✅ Saved FAISS index with {len(self.part_metadata)} parts")
        except Exception as e:
            print(f"❌ Failed to save index: {e}")

class EnhancedFFFScoringEngine:
    def __init__(self, faiss_engine: FAISSSimilarityEngine):
        self.faiss_engine = faiss_engine
        self.default_weights = {"form": 0.5, "fit": 0.3, "func": 0.2}
        self.hard_gates = {
            "rohs_match": True,
            "lifecycle_max": "NRND",
            "min_fff_score": 0.85
        }
    
    def calculate_fff_score(self, target_part: Dict, candidate_part: Dict, weights: Dict) -> Dict:
        """Calculate enhanced FFF score with FAISS similarity"""
        
        # Get FAISS similarity score
        faiss_similarity = self._get_faiss_similarity(target_part, candidate_part)
        
        # Traditional FFF scores
        form_score = self._calculate_form_score(target_part, candidate_part)
        fit_score = self._calculate_fit_score(target_part, candidate_part)
        func_score = self._calculate_func_score(target_part, candidate_part)
        
        # Enhanced scoring with FAISS similarity
        enhanced_form = (form_score + faiss_similarity) / 2
        enhanced_fit = (fit_score + faiss_similarity) / 2
        enhanced_func = (func_score + faiss_similarity) / 2
        
        # Overall FFF Score with FAISS enhancement
        fff_score = (weights["form"] * enhanced_form + 
                    weights["fit"] * enhanced_fit + 
                    weights["func"] * enhanced_func)
        
        # Check hard gates
        passes_gates, gate_reasons = self._check_hard_gates(target_part, candidate_part)
        
        return {
            "fff_score": fff_score,
            "form_score": enhanced_form,
            "fit_score": enhanced_fit,
            "func_score": enhanced_func,
            "faiss_similarity": faiss_similarity,
            "passes_gates": passes_gates,
            "gate_reasons": gate_reasons,
            "reasons": self._generate_enhanced_reasons(target_part, candidate_part, 
                                                     enhanced_form, enhanced_fit, enhanced_func, faiss_similarity)
        }
    
    def _get_faiss_similarity(self, target_part: Dict, candidate_part: Dict) -> float:
        """Get FAISS similarity score between two parts"""
        try:
            similar_parts = self.faiss_engine.find_similar_parts(target_part, k=1, threshold=0.0)
            for part, score in similar_parts:
                if part.get("mpn") == candidate_part.get("mpn"):
                    return score
            return 0.0
        except Exception as e:
            print(f"FAISS similarity error: {e}")
            return 0.0
    
    def _calculate_form_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate form compatibility score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Package compatibility (40% of form score)
        if target.get("package") and candidate.get("package"):
            if target["package"] == candidate["package"]:
                score += 0.4
            elif self._packages_compatible(target["package"], candidate["package"]):
                score += 0.3
            else:
                score += 0.1
        max_score += 0.4
        
        # Pin count compatibility (30% of form score)
        if target.get("pins") and candidate.get("pins"):
            if target["pins"] == candidate["pins"]:
                score += 0.3
            elif abs(target["pins"] - candidate["pins"]) <= 2:
                score += 0.2
            else:
                score += 0.05
        max_score += 0.3
        
        # Pinout compatibility (30% of form score)
        if target.get("pinout") and candidate.get("pinout"):
            pinout_match = self._compare_pinouts(target["pinout"], candidate["pinout"])
            score += 0.3 * pinout_match
        max_score += 0.3
        
        return score / max_score if max_score > 0 else 0.0
    
    def _calculate_fit_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate fit compatibility score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Voltage range compatibility (40% of fit score)
        if target.get("v_range") and candidate.get("v_range"):
            v_score = self._compare_voltage_ranges(target["v_range"], candidate["v_range"])
            score += 0.4 * v_score
        max_score += 0.4
        
        # Temperature range compatibility (30% of fit score)
        if target.get("temp_range") and candidate.get("temp_range"):
            t_score = self._compare_temperature_ranges(target["temp_range"], candidate["temp_range"])
            score += 0.3 * t_score
        max_score += 0.3
        
        # Power dissipation compatibility (20% of fit score)
        if target.get("power_dissipation") and candidate.get("power_dissipation"):
            p_score = self._compare_power_ratings(target["power_dissipation"], candidate["power_dissipation"])
            score += 0.2 * p_score
        max_score += 0.2
        
        # Tolerance compatibility (10% of fit score)
        if target.get("tolerance") and candidate.get("tolerance"):
            tol_score = self._compare_tolerances(target["tolerance"], candidate["tolerance"])
            score += 0.1 * tol_score
        max_score += 0.1
        
        return score / max_score if max_score > 0 else 0.0
    
    def _calculate_func_score(self, target: Dict, candidate: Dict) -> float:
        """Calculate function compatibility score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Category compatibility (50% of func score)
        if target.get("category") and candidate.get("category"):
            if target["category"] == candidate["category"]:
                score += 0.5
            elif self._categories_compatible(target["category"], candidate["category"]):
                score += 0.3
            else:
                score += 0.1
        max_score += 0.5
        
        # Manufacturer compatibility (20% of func score)
        if target.get("manufacturer") and candidate.get("manufacturer"):
            if target["manufacturer"] == candidate["manufacturer"]:
                score += 0.2
            else:
                score += 0.1
        max_score += 0.2
        
        # Specific attributes (30% of func score)
        attr_score = self._compare_specific_attributes(target, candidate)
        score += 0.3 * attr_score
        max_score += 0.3
        
        return score / max_score if max_score > 0 else 0.0
    
    def _packages_compatible(self, pkg1: str, pkg2: str) -> bool:
        """Check if packages are compatible"""
        compatible_groups = [
            ["SOIC-8", "SOP-8", "MSOP-8"],
            ["DIP-8", "PDIP-8"],
            ["QFN-8", "DFN-8"],
            ["SOT-23", "SOT-223"]
        ]
        
        for group in compatible_groups:
            if pkg1 in group and pkg2 in group:
                return True
        return False
    
    def _compare_voltage_ranges(self, v1: Dict, v2: Dict) -> float:
        """Compare voltage ranges (0-1)"""
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
        """Compare temperature ranges (0-1)"""
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
        """Compare power ratings (0-1)"""
        if p1 == 0 or p2 == 0:
            return 0.5
        
        ratio = min(p1, p2) / max(p1, p2)
        return ratio
    
    def _compare_tolerances(self, t1: float, t2: float) -> float:
        """Compare tolerances (0-1)"""
        if t1 == 0 or t2 == 0:
            return 0.5
        
        ratio = min(t1, t2) / max(t1, t2)
        return ratio
    
    def _categories_compatible(self, cat1: str, cat2: str) -> bool:
        """Check if categories are compatible"""
        compatible_groups = [
            ["OpAmp", "Comparator"],
            ["LDO", "Linear Regulator"],
            ["MOSFET", "Transistor"],
            ["Resistor", "Resistor Array"]
        ]
        
        for group in compatible_groups:
            if cat1 in group and cat2 in group:
                return True
        return False
    
    def _compare_specific_attributes(self, target: Dict, candidate: Dict) -> float:
        """Compare specific attributes based on category"""
        category = target.get("category", "")
        
        if category == "OpAmp":
            return self._compare_opamp_attributes(target, candidate)
        elif category == "LDO":
            return self._compare_ldo_attributes(target, candidate)
        elif category == "Resistor":
            return self._compare_resistor_attributes(target, candidate)
        else:
            return 0.5
    
    def _compare_opamp_attributes(self, target: Dict, candidate: Dict) -> float:
        """Compare op-amp specific attributes"""
        score = 0.0
        count = 0
        
        attrs = ["gbw_hz", "slew_vus", "input_offset_mv", "psrr_db"]
        for attr in attrs:
            if attr in target and attr in candidate:
                t_val = target[attr]
                c_val = candidate[attr]
                if t_val and c_val:
                    ratio = min(t_val, c_val) / max(t_val, c_val)
                    score += ratio
                    count += 1
        
        return score / count if count > 0 else 0.5
    
    def _compare_ldo_attributes(self, target: Dict, candidate: Dict) -> float:
        """Compare LDO specific attributes"""
        score = 0.0
        count = 0
        
        attrs = ["vdrop_v_at_iout", "psrr_db_100k", "load_regulation_mv"]
        for attr in attrs:
            if attr in target and attr in candidate:
                t_val = target[attr]
                c_val = candidate[attr]
                if t_val and c_val:
                    ratio = min(t_val, c_val) / max(t_val, c_val)
                    score += ratio
                    count += 1
        
        return score / count if count > 0 else 0.5
    
    def _compare_resistor_attributes(self, target: Dict, candidate: Dict) -> float:
        """Compare resistor specific attributes"""
        score = 0.0
        count = 0
        
        attrs = ["resistance_ohms", "tolerance_pct", "power_rating_w"]
        for attr in attrs:
            if attr in target and attr in candidate:
                t_val = target[attr]
                c_val = candidate[attr]
                if t_val and c_val:
                    ratio = min(t_val, c_val) / max(t_val, c_val)
                    score += ratio
                    count += 1
        
        return score / count if count > 0 else 0.5
    
    def _compare_pinouts(self, pinout1: List, pinout2: List) -> float:
        """Compare pinout compatibility"""
        if not pinout1 or not pinout2:
            return 0.5
        
        # Simple pinout comparison - can be enhanced
        matches = 0
        for p1 in pinout1:
            for p2 in pinout2:
                if p1.get("pin") == p2.get("pin") and p1.get("function") == p2.get("function"):
                    matches += 1
                    break
        
        return matches / max(len(pinout1), len(pinout2))
    
    def _check_hard_gates(self, target: Dict, candidate: Dict) -> Tuple[bool, List[str]]:
        """Check hard gates for part compatibility"""
        reasons = []
        
        # RoHS compliance
        if self.hard_gates["rohs_match"]:
            if target.get("rohs") and not candidate.get("rohs"):
                reasons.append("❌ Candidate not RoHS compliant")
                return False, reasons
            else:
                reasons.append("✅ RoHS compliant")
        
        # Lifecycle status
        lifecycle_max = self.hard_gates["lifecycle_max"]
        candidate_lifecycle = candidate.get("lifecycle", "ACTIVE")
        if candidate_lifecycle in ["EOL", "OBSOLETE"] and lifecycle_max in ["ACTIVE", "NRND"]:
            reasons.append(f"❌ Candidate lifecycle {candidate_lifecycle} exceeds maximum {lifecycle_max}")
            return False, reasons
        else:
            reasons.append(f"✅ Lifecycle acceptable: {candidate_lifecycle}")
        
        return True, reasons
    
    def _generate_enhanced_reasons(self, target: Dict, candidate: Dict, 
                                 form_score: float, fit_score: float, func_score: float, 
                                 faiss_similarity: float) -> List[str]:
        """Generate enhanced reasons for compatibility scores"""
        reasons = []
        
        # FAISS similarity reason
        if faiss_similarity >= 0.9:
            reasons.append("🧠 Excellent semantic similarity (FAISS)")
        elif faiss_similarity >= 0.7:
            reasons.append("🧠 Good semantic similarity (FAISS)")
        else:
            reasons.append("🧠 Limited semantic similarity (FAISS)")
        
        # Form reasons
        if form_score >= 0.9:
            reasons.append("✅ Excellent form compatibility (package, pins, pinout)")
        elif form_score >= 0.7:
            reasons.append("⚠️ Good form compatibility with minor differences")
        else:
            reasons.append("❌ Poor form compatibility - significant physical differences")
        
        # Fit reasons
        if fit_score >= 0.9:
            reasons.append("✅ Excellent electrical fit (voltage, temperature, power)")
        elif fit_score >= 0.7:
            reasons.append("⚠️ Good electrical fit with some limitations")
        else:
            reasons.append("❌ Poor electrical fit - may not work in same applications")
        
        # Function reasons
        if func_score >= 0.9:
            reasons.append("✅ Excellent functional compatibility")
        elif func_score >= 0.7:
            reasons.append("⚠️ Good functional compatibility")
        else:
            reasons.append("❌ Limited functional compatibility")
        
        return reasons

class MouserAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mouser.com/api/v1"
    
    async def search_parts(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for parts using Mouser API"""
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/search/keyword"
            headers = {"Content-Type": "application/json"}
            data = {
                "SearchByKeywordRequest": {
                    "keyword": query,
                    "records": limit,
                    "searchOptions": "InStock"
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}?apiKey={self.api_key}",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    result = await response.json()
                    
                    if "Errors" in result and result["Errors"]:
                        print(f"Mouser API error: {result['Errors']}")
                        return []
                    
                    parts = []
                    search_results = result.get("SearchResults", {})
                    parts_data = search_results.get("Parts", [])
                    
                    for item in parts_data:
                        # Extract price information
                        price_breaks = item.get("PriceBreaks", [])
                        unit_price = 0.0
                        if price_breaks:
                            unit_price = float(price_breaks[0].get("Price", "0").replace("$", ""))
                        
                        # Extract stock information
                        availability = item.get("Availability", "0")
                        stock = 0
                        if "In Stock" in availability:
                            try:
                                stock = int(availability.split()[0])
                            except:
                                stock = 0
                        
                        part = {
                            "mpn": item.get("ManufacturerPartNumber", ""),
                            "mouser_part_number": item.get("MouserPartNumber", ""),
                            "manufacturer": item.get("Manufacturer", ""),
                            "description": item.get("Description", ""),
                            "category": self._categorize_part(item.get("Description", "")),
                            "package": self._extract_package_info(item),
                            "pins": self._extract_pin_count(item),
                            "v_range": self._extract_voltage_range(item),
                            "temp_range": self._extract_temperature_range(item),
                            "stock": stock,
                            "unit_price": unit_price,
                            "lifecycle": item.get("LifecycleStatus", "ACTIVE"),
                            "rohs": item.get("ROHSStatus", "").lower() == "rohs compliant",
                            "datasheet_url": item.get("DataSheetUrl", ""),
                            "product_url": item.get("ProductDetailUrl", ""),
                            "lead_time": item.get("LeadTime", ""),
                            "provenance": ["mouser_api"]
                        }
                        parts.append(part)
                    
                    return parts
                    
        except Exception as e:
            print(f"Mouser API search failed: {e}")
            return []
    
    def _categorize_part(self, description: str) -> str:
        """Categorize part based on description"""
        desc_lower = description.lower()
        if "operational amplifier" in desc_lower or "op amp" in desc_lower:
            return "OpAmp"
        elif "ldo" in desc_lower or "linear regulator" in desc_lower:
            return "LDO"
        elif "resistor" in desc_lower:
            return "Resistor"
        elif "capacitor" in desc_lower:
            return "Capacitor"
        elif "mosfet" in desc_lower or "transistor" in desc_lower:
            return "MOSFET"
        else:
            return "Unknown"
    
    def _extract_package_info(self, item: Dict) -> str:
        """Extract package information from Mouser data"""
        description = item.get("Description", "")
        if "SOP" in description or "SOIC" in description:
            return "SOIC-8"
        elif "DIP" in description:
            return "DIP-8"
        elif "QFN" in description:
            return "QFN-8"
        elif "SOT" in description:
            return "SOT-223"
        else:
            return "Unknown"
    
    def _extract_pin_count(self, item: Dict) -> int:
        """Extract pin count from description"""
        description = item.get("Description", "")
        if "8" in description:
            return 8
        elif "14" in description:
            return 14
        elif "16" in description:
            return 16
        else:
            return 0
    
    def _extract_voltage_range(self, item: Dict) -> Dict:
        """Extract voltage range from description"""
        description = item.get("Description", "")
        # Simple extraction - can be enhanced with regex
        if "3-32V" in description:
            return {"min": 3.0, "max": 32.0, "unit": "V"}
        elif "6-36V" in description:
            return {"min": 6.0, "max": 36.0, "unit": "V"}
        else:
            return {"min": 0.0, "max": 0.0, "unit": "V"}
    
    def _extract_temperature_range(self, item: Dict) -> Dict:
        """Extract temperature range from description"""
        description = item.get("Description", "")
        # Simple extraction - can be enhanced with regex
        if "-40°C to 85°C" in description or "-40C to 85C" in description:
            return {"min": -40, "max": 85, "unit": "°C"}
        elif "0°C to 70°C" in description or "0C to 70C" in description:
            return {"min": 0, "max": 70, "unit": "°C"}
        else:
            return {"min": 0, "max": 0, "unit": "°C"}

# Initialize components
faiss_engine = FAISSSimilarityEngine()
fff_engine = EnhancedFFFScoringEngine(faiss_engine)
mouser_client = MouserAPIClient(MOUSER_API_KEY)

def format_part_info(part: Dict) -> str:
    """Format part information for display"""
    return f"""
**{part['mpn']}** - {part['manufacturer']}
{part['description']}

**Mouser Part #:** {part.get('mouser_part_number', 'N/A')}
**Category:** {part['category']}
**Package:** {part['package']} ({part.get('pins', 'N/A')} pins)
**Voltage Range:** {part.get('v_range', {}).get('min', 'N/A')}-{part.get('v_range', {}).get('max', 'N/A')}V
**Temperature Range:** {part.get('temp_range', {}).get('min', 'N/A')}-{part.get('temp_range', {}).get('max', 'N/A')}°C
**Stock:** {part['stock']} units
**Price:** ${part['unit_price']:.3f}
**Lead Time:** {part.get('lead_time', 'N/A')}
**Lifecycle:** {part['lifecycle']}
**RoHS:** {'Yes' if part['rohs'] else 'No'}

**Links:**
- [Mouser Product Page]({part.get('product_url', '#')})
- [Datasheet]({part.get('datasheet_url', '#') if part.get('datasheet_url') else 'Not available'})

**Provenance:** {', '.join(part.get('provenance', []))}
"""

def format_replacement_info(replacement: Dict) -> str:
    """Format replacement information for display"""
    return f"""
**{replacement['mpn']}** - {replacement['manufacturer']}
**FFF Score:** {replacement['fff_score']:.2f} (Form: {replacement['form_score']:.2f}, Fit: {replacement['fit_score']:.2f}, Function: {replacement['func_score']:.2f})
**FAISS Similarity:** {replacement.get('faiss_similarity', 0.0):.2f}

**Stock:** {replacement.get('stock', 'N/A')} units
**Price:** ${replacement.get('unit_price', 'N/A')}
**Lifecycle:** {replacement.get('lifecycle', 'Unknown')}

**Compatibility Analysis:**
{chr(10).join(f"• {reason}" for reason in replacement['reasons'])}

**Hard Gate Results:**
{chr(10).join(f"• {reason}" for reason in replacement['gate_reasons'])}

**Provenance:** {', '.join(replacement.get('provenance', []))}
"""

async def search_parts_async(query: str, limit: int) -> str:
    """Search for parts using Mouser API and add to FAISS index"""
    if not query.strip():
        return "Please enter a search query."
    
    parts = await mouser_client.search_parts(query, limit)
    
    if not parts:
        return f"No parts found for query: {query}"
    
    # Add parts to FAISS index
    faiss_engine.add_parts(parts)
    faiss_engine.save_index()
    
    result = f"Found {len(parts)} parts for '{query}' and added to FAISS index:\n\n"
    for i, part in enumerate(parts, 1):
        result += f"**{i}.** {format_part_info(part)}\n"
        result += "---\n"
    
    return result

async def find_replacements_async(target_mpn: str, form_weight: float, fit_weight: float, func_weight: float, min_fff_score: float) -> str:
    """Find replacement parts using enhanced FFF scoring with FAISS similarity"""
    if not target_mpn.strip():
        return "Please enter a target MPN."
    
    # Normalize weights
    total_weight = form_weight + fit_weight + func_weight
    if total_weight == 0:
        form_weight, fit_weight, func_weight = 0.5, 0.3, 0.2
    else:
        form_weight /= total_weight
        fit_weight /= total_weight
        func_weight /= total_weight
    
    weights = {"form": form_weight, "fit": fit_weight, "func": func_weight}
    
    # Search for potential replacements
    search_parts = await mouser_client.search_parts(target_mpn, 20)
    
    if not search_parts:
        return f"No parts found for target: {target_mpn}"
    
    # Add to FAISS index
    faiss_engine.add_parts(search_parts)
    
    # Use first part as target
    target_part = search_parts[0]
    
    # Calculate enhanced FFF scores for all parts
    replacements = []
    for part in search_parts[1:]:  # Skip the target part itself
        fff_result = fff_engine.calculate_fff_score(target_part, part, weights)
        
        if fff_result["fff_score"] >= min_fff_score and fff_result["passes_gates"]:
            replacement = {
                "mpn": part["mpn"],
                "manufacturer": part["manufacturer"],
                "fff_score": fff_result["fff_score"],
                "form_score": fff_result["form_score"],
                "fit_score": fff_result["fit_score"],
                "func_score": fff_result["func_score"],
                "faiss_similarity": fff_result["faiss_similarity"],
                "reasons": fff_result["reasons"],
                "gate_reasons": fff_result["gate_reasons"],
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
    
    result = f"**Enhanced FFF Replacement Analysis for {target_mpn}:**\n"
    result += f"**Weights:** Form={form_weight:.1f}, Fit={fit_weight:.1f}, Function={func_weight:.1f}\n"
    result += f"**Minimum FFF Score:** {min_fff_score:.2f}\n"
    result += f"**FAISS Index:** {len(faiss_engine.part_metadata)} parts indexed\n\n"
    
    for i, replacement in enumerate(replacements[:10], 1):  # Show top 10
        result += f"**{i}.** {format_replacement_info(replacement)}\n"
        result += "---\n"
    
    return result

def get_system_status():
    """Get system status"""
    return f"""
**System Status:**

**OpenAI API:** {'✅ Configured' if OPENAI_API_KEY else '❌ Not configured'}
**Mouser API:** {'✅ Configured' if MOUSER_API_KEY else '❌ Not configured'}
**LlamaCloud API:** {'✅ Configured' if LLAMA_CLOUD_API_KEY else '❌ Not configured'}
**Apify API:** {'✅ Configured' if APIFY_API_TOKEN else '❌ Not configured'}

**FAISS Similarity Engine:**
- ✅ Sentence Transformer: all-MiniLM-L6-v2
- ✅ Vector Dimension: 384
- ✅ Indexed Parts: {len(faiss_engine.part_metadata)}
- ✅ Index Status: {'Trained' if faiss_engine.is_trained else 'Not trained'}

**Enhanced FFF Scoring:**
- ✅ Form analysis (package, pins, pinout)
- ✅ Fit analysis (voltage, temperature, power)
- ✅ Function analysis (category, attributes)
- ✅ FAISS semantic similarity
- ✅ Hard gates (RoHS, lifecycle)
- ✅ User-configurable weights

**LlamaCloud Dashboards:**
- [Parse Dashboard](https://cloud.llamaindex.ai/project/8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe/parse)
- [Extract Dashboard](https://cloud.llamaindex.ai/project/8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe/extract)
"""

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="PartSync - Enhanced FFF with FAISS",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
        # 🔧 PartSync - Enhanced FFF Scoring with FAISS Similarity
        
        **Form-Fit-Function Analysis with FAISS Vector Similarity for Superior Part Matching**
        
        Find compatible replacement parts using configurable FFF scoring enhanced with semantic similarity analysis.
        """)
        
        with gr.Tabs():
            
            # Part Search Tab
            with gr.Tab("🔍 Part Search + FAISS Indexing"):
                gr.Markdown("### Search for electronic parts and build FAISS similarity index")
                
                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter part number, description, or keywords (e.g., LM358, op amp, LDO)",
                        value="LM358"
                    )
                    search_limit = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Max Results"
                    )
                
                search_btn = gr.Button("🔍 Search & Index Parts", variant="primary")
                search_output = gr.Markdown(label="Search Results")
                
                search_btn.click(
                    fn=search_parts_async,
                    inputs=[search_query, search_limit],
                    outputs=search_output
                )
            
            # Enhanced FFF Replacement Tab
            with gr.Tab("🧠 Enhanced FFF Analysis"):
                gr.Markdown("### Find compatible replacement parts using FAISS-enhanced FFF scoring")
                
                with gr.Row():
                    target_mpn = gr.Textbox(
                        label="Target MPN",
                        placeholder="Enter the part number you want to replace (e.g., LM358FVJ-E2)",
                        value="LM358FVJ-E2"
                    )
                
                gr.Markdown("### FFF Scoring Weights")
                with gr.Row():
                    form_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Form Weight (Package, Pins, Pinout)"
                    )
                    fit_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Fit Weight (Voltage, Temperature, Power)"
                    )
                    func_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        label="Function Weight (Category, Attributes)"
                    )
                
                with gr.Row():
                    min_fff_score = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        label="Minimum FFF Score (0.85 = 85%)"
                    )
                
                replace_btn = gr.Button("🧠 Find Enhanced Replacements", variant="primary")
                replace_output = gr.Markdown(label="Enhanced Replacement Analysis")
                
                replace_btn.click(
                    fn=find_replacements_async,
                    inputs=[target_mpn, form_weight, fit_weight, func_weight, min_fff_score],
                    outputs=replace_output
                )
            
            # System Status Tab
            with gr.Tab("⚙️ System Status"):
                gr.Markdown("### API Status and FAISS Index Information")
                
                status_output = gr.Markdown(label="System Status")
                status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                
                status_btn.click(fn=get_system_status, outputs=status_output)
                
                # Show initial status
                interface.load(fn=get_system_status, outputs=status_output)
    
    return interface

def main():
    """Main function to launch the Gradio app"""
    print("🚀 Starting PartSync Enhanced FFF with FAISS...")
    
    interface = create_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("📱 The app will open in your browser at http://localhost:7865")
    print("🛑 Press Ctrl+C to stop the server")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
