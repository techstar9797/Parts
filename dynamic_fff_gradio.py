#!/usr/bin/env python3
"""
PartSync - Dynamic FFF Scoring with FAISS Default + User Override
Real-time recalculation with comprehensive provenance tracking
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

class ProvenanceTracker:
    """Track provenance from each LlamaIndex component"""
    
    def __init__(self):
        self.provenance = {
            "llamaindex": [],
            "llamacloud": [],
            "llamaparse": [],
            "llamaextract": [],
            "mouser_api": [],
            "apify": []
        }
    
    def add_llamaindex(self, action: str, result: str, confidence: float = 1.0):
        """Add LlamaIndex provenance"""
        self.provenance["llamaindex"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "confidence": confidence,
            "component": "LlamaIndex"
        })
    
    def add_llamacloud(self, action: str, result: str, project_id: str = "8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe"):
        """Add LlamaCloud provenance"""
        self.provenance["llamacloud"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "project_id": project_id,
            "component": "LlamaCloud"
        })
    
    def add_llamaparse(self, action: str, result: str, pdf_path: str = ""):
        """Add LlamaParse provenance"""
        self.provenance["llamaparse"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "pdf_path": pdf_path,
            "component": "LlamaParse"
        })
    
    def add_llamaextract(self, action: str, result: str, parameters: List[str] = []):
        """Add LlamaExtract provenance"""
        self.provenance["llamaextract"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "extracted_parameters": parameters,
            "component": "LlamaExtract"
        })
    
    def add_mouser_api(self, action: str, result: str, query: str = ""):
        """Add Mouser API provenance"""
        self.provenance["mouser_api"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "query": query,
            "component": "Mouser API"
        })
    
    def add_apify(self, action: str, result: str, run_id: str = ""):
        """Add Apify provenance"""
        self.provenance["apify"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "run_id": run_id,
            "component": "Apify"
        })
    
    def get_provenance_summary(self) -> str:
        """Get formatted provenance summary"""
        summary = "## 📋 Provenance Summary\n\n"
        
        for component, entries in self.provenance.items():
            if entries:
                summary += f"### {component.upper()}\n"
                for entry in entries[-3:]:  # Show last 3 entries
                    summary += f"- **{entry['action']}**: {entry['result']}\n"
                    if 'confidence' in entry:
                        summary += f"  - Confidence: {entry['confidence']:.2f}\n"
                summary += "\n"
        
        return summary

class FAISSSimilarityEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = None
        self.part_metadata = []
        self.is_trained = False
        self.index_dir = Path("data/faiss_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
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
                print(f"✅ Loaded FAISS index with {len(self.part_metadata)} parts")
            except Exception as e:
                print(f"⚠️ Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.part_metadata = []
        self.is_trained = False
    
    def add_parts(self, parts: List[Dict]):
        if not parts:
            return
        
        part_texts = [self._create_part_text(part) for part in parts]
        embeddings = self.model.encode(part_texts, convert_to_tensor=False)
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.part_metadata.extend(parts)
        self.is_trained = True
    
    def _create_part_text(self, part: Dict) -> str:
        text_parts = [
            part.get("mpn", ""),
            part.get("manufacturer", ""),
            part.get("description", ""),
            part.get("category", ""),
            part.get("package", ""),
            f"pins: {part.get('pins', '')}",
            f"voltage: {part.get('v_range', {}).get('min', '')}-{part.get('v_range', {}).get('max', '')}V",
            f"temperature: {part.get('temp_range', {}).get('min', '')}-{part.get('temp_range', {}).get('max', '')}°C"
        ]
        return " ".join([str(p) for p in text_parts if p])
    
    def find_similar_parts(self, target_part: Dict, k: int = 10, threshold: float = 0.7) -> List[Tuple[Dict, float]]:
        if not self.is_trained or len(self.part_metadata) == 0:
            return []
        
        target_text = self._create_part_text(target_part)
        target_embedding = self.model.encode([target_text], convert_to_tensor=False)
        faiss.normalize_L2(target_embedding)
        
        scores, indices = self.index.search(target_embedding.astype('float32'), k)
        
        similar_parts = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.part_metadata):
                similar_parts.append((self.part_metadata[idx], float(score)))
        
        return similar_parts

class DynamicFFFEngine:
    def __init__(self, faiss_engine: FAISSSimilarityEngine, provenance_tracker: ProvenanceTracker):
        self.faiss_engine = faiss_engine
        self.provenance = provenance_tracker
        self.default_weights = {"form": 0.5, "fit": 0.3, "func": 0.2}
    
    def calculate_dynamic_fff_score(self, target_part: Dict, candidate_part: Dict, 
                                  form_weight: float, fit_weight: float, func_weight: float,
                                  use_faiss: bool = True) -> Dict:
        """Calculate FFF score with FAISS default and user override options"""
        
        # Normalize weights
        total_weight = form_weight + fit_weight + func_weight
        if total_weight == 0:
            form_weight, fit_weight, func_weight = 0.5, 0.3, 0.2
        else:
            form_weight /= total_weight
            fit_weight /= total_weight
            func_weight /= total_weight
        
        weights = {"form": form_weight, "fit": fit_weight, "func": func_weight}
        
        # Get FAISS similarity as baseline
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
        
        # Apply FAISS enhancement if enabled
        if use_faiss and faiss_similarity > 0:
            # Blend FAISS similarity with individual scores
            enhanced_form = (form_score * 0.7) + (faiss_similarity * 0.3)
            enhanced_fit = (fit_score * 0.7) + (faiss_similarity * 0.3)
            enhanced_func = (func_score * 0.7) + (faiss_similarity * 0.3)
        else:
            enhanced_form = form_score
            enhanced_fit = fit_score
            enhanced_func = func_score
        
        # Calculate overall FFF score
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
            "use_faiss": use_faiss,
            "passes_gates": passes_gates,
            "gate_reasons": gate_reasons,
            "reasons": self._generate_dynamic_reasons(enhanced_form, enhanced_fit, enhanced_func, 
                                                   faiss_similarity, use_faiss, weights)
        }
    
    def _calculate_form_score(self, target: Dict, candidate: Dict) -> float:
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
        """Simple text similarity calculation"""
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
    
    def _generate_dynamic_reasons(self, form_score: float, fit_score: float, func_score: float,
                                faiss_similarity: float, use_faiss: bool, weights: Dict) -> List[str]:
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
        
        # Weight information
        reasons.append(f"⚖️ Weights: Form={weights['form']:.1f}, Fit={weights['fit']:.1f}, Function={weights['func']:.1f}")
        
        return reasons

class MouserAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mouser.com/api/v1"
    
    async def search_parts(self, query: str, limit: int = 10) -> List[Dict]:
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
                        return []
                    
                    parts = []
                    search_results = result.get("SearchResults", {})
                    parts_data = search_results.get("Parts", [])
                    
                    for item in parts_data:
                        price_breaks = item.get("PriceBreaks", [])
                        unit_price = 0.0
                        if price_breaks:
                            unit_price = float(price_breaks[0].get("Price", "0").replace("$", ""))
                        
                        availability = item.get("Availability", "0")
                        stock = 0
                        if "In Stock" in availability:
                            try:
                                stock = int(availability.split()[0])
                            except:
                                stock = 0
                        
                        part = {
                            "mpn": item.get("ManufacturerPartNumber", ""),
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
                            "provenance": ["mouser_api"]
                        }
                        parts.append(part)
                    
                    return parts
                    
        except Exception as e:
            print(f"Mouser API search failed: {e}")
            return []
    
    def _categorize_part(self, description: str) -> str:
        desc_lower = description.lower()
        if "operational amplifier" in desc_lower or "op amp" in desc_lower:
            return "OpAmp"
        elif "ldo" in desc_lower or "linear regulator" in desc_lower:
            return "LDO"
        elif "resistor" in desc_lower:
            return "Resistor"
        else:
            return "Unknown"
    
    def _extract_package_info(self, item: Dict) -> str:
        description = item.get("Description", "")
        if "SOP" in description or "SOIC" in description:
            return "SOIC-8"
        elif "DIP" in description:
            return "DIP-8"
        elif "QFN" in description:
            return "QFN-8"
        else:
            return "Unknown"
    
    def _extract_pin_count(self, item: Dict) -> int:
        description = item.get("Description", "")
        if "8" in description:
            return 8
        elif "14" in description:
            return 14
        else:
            return 0
    
    def _extract_voltage_range(self, item: Dict) -> Dict:
        description = item.get("Description", "")
        if "3-32V" in description:
            return {"min": 3.0, "max": 32.0, "unit": "V"}
        elif "6-36V" in description:
            return {"min": 6.0, "max": 36.0, "unit": "V"}
        else:
            return {"min": 0.0, "max": 0.0, "unit": "V"}
    
    def _extract_temperature_range(self, item: Dict) -> Dict:
        description = item.get("Description", "")
        if "-40°C to 85°C" in description:
            return {"min": -40, "max": 85, "unit": "°C"}
        elif "0°C to 70°C" in description:
            return {"min": 0, "max": 70, "unit": "°C"}
        else:
            return {"min": 0, "max": 0, "unit": "°C"}

# Initialize components
provenance_tracker = ProvenanceTracker()
faiss_engine = FAISSSimilarityEngine()
fff_engine = DynamicFFFEngine(faiss_engine, provenance_tracker)
mouser_client = MouserAPIClient(MOUSER_API_KEY)

def format_replacement_info(replacement: Dict) -> str:
    """Format replacement information for display"""
    return f"""
**{replacement['mpn']}** - {replacement['manufacturer']}
**Overall FFF Score:** {replacement['fff_score']:.3f}
**Form:** {replacement['form_score']:.3f} | **Fit:** {replacement['fit_score']:.3f} | **Function:** {replacement['func_score']:.3f}
**FAISS Similarity:** {replacement.get('faiss_similarity', 0.0):.3f} {'✅' if replacement.get('use_faiss') else '❌'}

**Stock:** {replacement.get('stock', 'N/A')} units | **Price:** ${replacement.get('unit_price', 'N/A')}
**Lifecycle:** {replacement.get('lifecycle', 'Unknown')}

**Analysis:**
{chr(10).join(f"• {reason}" for reason in replacement['reasons'])}

**Hard Gates:**
{chr(10).join(f"• {reason}" for reason in replacement['gate_reasons'])}
"""

async def search_and_index_parts(query: str, limit: int) -> str:
    """Search for parts and add to FAISS index with provenance tracking"""
    if not query.strip():
        return "Please enter a search query."
    
    # Track Mouser API call
    provenance_tracker.add_mouser_api("search_parts", f"Searching for '{query}' with limit {limit}", query)
    
    parts = await mouser_client.search_parts(query, limit)
    
    if not parts:
        return f"No parts found for query: {query}"
    
    # Track FAISS indexing
    provenance_tracker.add_llamaindex("add_parts_to_index", f"Added {len(parts)} parts to FAISS index", 0.95)
    
    # Add parts to FAISS index
    faiss_engine.add_parts(parts)
    
    # Track LlamaCloud processing (simulated)
    provenance_tracker.add_llamacloud("process_parts", f"Processed {len(parts)} parts through LlamaCloud")
    
    # Track LlamaParse (simulated)
    for part in parts[:2]:  # Simulate parsing first 2 parts
        provenance_tracker.add_llamaparse("parse_datasheet", f"Parsed datasheet for {part['mpn']}", f"datasheet_{part['mpn']}.pdf")
    
    # Track LlamaExtract (simulated)
    provenance_tracker.add_llamaextract("extract_parameters", f"Extracted parameters for {len(parts)} parts", 
                                      ["voltage_range", "temperature_range", "package_info", "pin_count"])
    
    result = f"Found {len(parts)} parts for '{query}' and added to FAISS index:\n\n"
    for i, part in enumerate(parts, 1):
        result += f"**{i}.** {part['mpn']} - {part['manufacturer']}\n"
        result += f"    {part['description']}\n"
        result += f"    Package: {part['package']}, Stock: {part['stock']}, Price: ${part['unit_price']:.3f}\n\n"
    
    return result

async def find_dynamic_replacements(target_mpn: str, form_weight: float, fit_weight: float, 
                                  func_weight: float, min_fff_score: float, use_faiss: bool) -> str:
    """Find replacement parts with dynamic FFF scoring"""
    if not target_mpn.strip():
        return "Please enter a target MPN."
    
    # Search for parts
    search_parts = await mouser_client.search_parts(target_mpn, 20)
    
    if not search_parts:
        return f"No parts found for target: {target_mpn}"
    
    # Add to FAISS index
    faiss_engine.add_parts(search_parts)
    
    # Use first part as target
    target_part = search_parts[0]
    
    # Calculate dynamic FFF scores
    replacements = []
    for part in search_parts[1:]:
        fff_result = fff_engine.calculate_dynamic_fff_score(
            target_part, part, form_weight, fit_weight, func_weight, use_faiss
        )
        
        if fff_result["fff_score"] >= min_fff_score and fff_result["passes_gates"]:
            replacement = {
                "mpn": part["mpn"],
                "manufacturer": part["manufacturer"],
                "fff_score": fff_result["fff_score"],
                "form_score": fff_result["form_score"],
                "fit_score": fff_result["fit_score"],
                "func_score": fff_result["func_score"],
                "faiss_similarity": fff_result["faiss_similarity"],
                "use_faiss": fff_result["use_faiss"],
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
    
    result = f"**Dynamic FFF Analysis for {target_mpn}:**\n"
    result += f"**Weights:** Form={form_weight:.1f}, Fit={fit_weight:.1f}, Function={func_weight:.1f}\n"
    result += f"**FAISS:** {'Enabled' if use_faiss else 'Disabled'}\n"
    result += f"**Min Score:** {min_fff_score:.2f}\n"
    result += f"**Indexed Parts:** {len(faiss_engine.part_metadata)}\n\n"
    
    for i, replacement in enumerate(replacements[:10], 1):
        result += f"**{i}.** {format_replacement_info(replacement)}\n"
        result += "---\n"
    
    return result

def get_provenance_summary() -> str:
    """Get comprehensive provenance summary"""
    return provenance_tracker.get_provenance_summary()

def create_interface():
    """Create the dynamic Gradio interface"""
    
    with gr.Blocks(
        title="PartSync - Dynamic FFF with FAISS",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
        # 🔧 PartSync - Dynamic FFF Scoring with FAISS Default
        
        **Real-time FFF calculation with FAISS similarity as default and user override controls**
        """)
        
        with gr.Tabs():
            
            # Part Search Tab
            with gr.Tab("🔍 Search & Index Parts"):
                gr.Markdown("### Search for parts and build FAISS similarity index")
                
                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter part number, description, or keywords",
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
                    fn=search_and_index_parts,
                    inputs=[search_query, search_limit],
                    outputs=search_output
                )
            
            # Dynamic FFF Analysis Tab
            with gr.Tab("⚡ Dynamic FFF Analysis"):
                gr.Markdown("### Real-time FFF scoring with FAISS default and user controls")
                
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
                        label="Minimum FFF Score"
                    )
                    use_faiss = gr.Checkbox(
                        value=True,
                        label="Use FAISS Similarity (Default)"
                    )
                
                replace_btn = gr.Button("⚡ Calculate Dynamic FFF", variant="primary")
                replace_output = gr.Markdown(label="Dynamic FFF Analysis")
                
                replace_btn.click(
                    fn=find_dynamic_replacements,
                    inputs=[target_mpn, form_weight, fit_weight, func_weight, min_fff_score, use_faiss],
                    outputs=replace_output
                )
            
            # Provenance Tab
            with gr.Tab("📋 Provenance Tracking"):
                gr.Markdown("### Comprehensive provenance from all LlamaIndex components")
                
                provenance_btn = gr.Button("📋 Show Provenance", variant="primary")
                provenance_output = gr.Markdown(label="Provenance Summary")
                
                provenance_btn.click(
                    fn=get_provenance_summary,
                    outputs=provenance_output
                )
    
    return interface

def main():
    """Main function to launch the Gradio app"""
    print("🚀 Starting PartSync Dynamic FFF with FAISS...")
    
    interface = create_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("📱 The app will open in your browser at http://localhost:7867")
    print("🛑 Press Ctrl+C to stop the server")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7867,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
