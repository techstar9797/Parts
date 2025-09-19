import os
import httpx
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

class MouserAPI:
    def __init__(self):
        self.api_key = os.getenv("MOUSER_API_KEY")
        if not self.api_key:
            raise ValueError("MOUSER_API_KEY environment variable is required")
        
        self.base_url = "https://api.mouser.com/api/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def search_parts(self, 
                          search_term: str,
                          records_per_page: int = 50,
                          page_number: int = 1) -> Dict[str, Any]:
        """Search for parts using Mouser API"""
        url = f"{self.base_url}/search/keyword"
        
        payload = {
            "SearchByKeywordRequest": {
                "keyword": search_term,
                "records": records_per_page,
                "startingRecord": (page_number - 1) * records_per_page,
                "searchOptions": "InStock",
                "searchWithYourSignUpLanguage": "English"
            }
        }
        
        params = {"apiKey": self.api_key}
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, json=payload, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Mouser API search failed: {e}")
                raise
    
    async def get_part_details(self, part_number: str) -> Dict[str, Any]:
        """Get detailed information for a specific part number"""
        url = f"{self.base_url}/search/partnumber"
        
        payload = {
            "SearchByPartRequest": {
                "mouserPartNumber": part_number,
                "partSearchOptions": "InStock"
            }
        }
        
        params = {"apiKey": self.api_key}
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, json=payload, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Mouser API part details failed for {part_number}: {e}")
                raise
    
    async def get_part_by_mpn(self, mpn: str, manufacturer: str) -> Optional[Dict[str, Any]]:
        """Get part information by MPN and manufacturer"""
        # First try searching by MPN
        search_results = await self.search_parts(f"{manufacturer} {mpn}")
        
        if search_results.get("SearchResults", {}).get("Parts"):
            parts = search_results["SearchResults"]["Parts"]
            # Find exact match
            for part in parts:
                if (part.get("ManufacturerPartNumber", "").upper() == mpn.upper() and 
                    part.get("Manufacturer", "").upper() == manufacturer.upper()):
                    return part
        
        return None
    
    def extract_part_data(self, mouser_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from Mouser API response"""
        if not mouser_data.get("SearchResults", {}).get("Parts"):
            return {}
        
        part = mouser_data["SearchResults"]["Parts"][0]  # Get first result
        
        # Extract basic information
        extracted = {
            "mpn": part.get("ManufacturerPartNumber", ""),
            "manufacturer": part.get("Manufacturer", ""),
            "description": part.get("Description", ""),
            "category": part.get("Category", ""),
            "mouser_part_number": part.get("MouserPartNumber", ""),
            "mouser_url": part.get("ProductDetailUrl", ""),
            "datasheet_url": part.get("DataSheetUrl", ""),
            "image_url": part.get("ImagePath", ""),
            "rohs": part.get("ROHSStatus", "").upper() == "ROHS COMPLIANT",
            "lifecycle": part.get("LifecycleStatus", ""),
            "stock": part.get("Availability", {}).get("OnDemand", 0) if part.get("Availability") else 0,
            "unit_price": part.get("PriceBreaks", [{}])[0].get("Price", 0) if part.get("PriceBreaks") else 0,
            "currency": part.get("PriceBreaks", [{}])[0].get("Currency", "USD") if part.get("PriceBreaks") else "USD"
        }
        
        # Extract package information
        if part.get("ProductAttributes"):
            attrs = part["ProductAttributes"]
            package_info = {}
            
            for attr in attrs:
                attr_name = attr.get("AttributeName", "").lower()
                attr_value = attr.get("AttributeValue", "")
                
                if "package" in attr_name or "case" in attr_name:
                    package_info["name"] = attr_value
                elif "pins" in attr_name or "pin count" in attr_name:
                    try:
                        package_info["pins"] = int(attr_value.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif "pitch" in attr_name:
                    try:
                        # Extract numeric value and convert to mm if needed
                        value_str = attr_value.replace("mm", "").replace("mil", "").strip()
                        pitch = float(value_str)
                        if "mil" in attr_value.lower():
                            pitch = pitch * 0.0254  # Convert mils to mm
                        package_info["pitch_mm"] = pitch
                    except (ValueError, IndexError):
                        pass
            
            extracted["package"] = package_info
        
        # Extract electrical specifications
        electrical_specs = {}
        if part.get("ProductAttributes"):
            for attr in part["ProductAttributes"]:
                attr_name = attr.get("AttributeName", "").lower()
                attr_value = attr.get("AttributeValue", "")
                
                # Voltage specifications
                if "voltage" in attr_name and "supply" in attr_name:
                    electrical_specs["supply_voltage"] = attr_value
                elif "operating voltage" in attr_name:
                    electrical_specs["operating_voltage"] = attr_value
                
                # Temperature specifications
                elif "operating temperature" in attr_name:
                    electrical_specs["operating_temp"] = attr_value
                
                # Other electrical specs
                elif any(keyword in attr_name for keyword in ["current", "power", "frequency", "speed"]):
                    electrical_specs[attr_name.replace(" ", "_")] = attr_value
        
        extracted["electrical_specs"] = electrical_specs
        
        return extracted
    
    async def enrich_part_record(self, part_record: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a part record with Mouser API data"""
        mpn = part_record.get("mpn", "")
        manufacturer = part_record.get("manufacturer", "")
        
        if not mpn or not manufacturer:
            return part_record
        
        try:
            mouser_data = await self.get_part_by_mpn(mpn, manufacturer)
            if mouser_data:
                extracted_data = self.extract_part_data({"SearchResults": {"Parts": [mouser_data]}})
                
                # Merge with existing part record
                enriched = {**part_record, **extracted_data}
                
                # Add provenance
                if "provenance" not in enriched:
                    enriched["provenance"] = []
                enriched["provenance"].append(f"mouser://{enriched.get('mouser_part_number', '')}")
                
                return enriched
            else:
                logger.warning(f"No Mouser data found for {manufacturer} {mpn}")
                return part_record
                
        except Exception as e:
            logger.error(f"Failed to enrich part {manufacturer} {mpn}: {e}")
            return part_record
    
    async def batch_enrich_parts(self, part_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich multiple part records with Mouser data"""
        enriched_parts = []
        
        for part_record in part_records:
            enriched = await self.enrich_part_record(part_record)
            enriched_parts.append(enriched)
        
        return enriched_parts

# Convenience functions
async def search_mouser_parts(search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search for parts on Mouser"""
    api = MouserAPI()
    results = await api.search_parts(search_term, limit)
    
    if results.get("SearchResults", {}).get("Parts"):
        return [api.extract_part_data({"SearchResults": {"Parts": [part]}}) 
                for part in results["SearchResults"]["Parts"]]
    return []

async def enrich_with_mouser_data(part_record: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a single part record with Mouser data"""
    api = MouserAPI()
    return await api.enrich_part_record(part_record)
