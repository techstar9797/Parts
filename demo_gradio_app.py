#!/usr/bin/env python3
"""
PartSync - Demo Gradio App with LlamaCloud Integration
Demonstrates the full pipeline with mock data and real API integration
"""

import gradio as gr
import os
import json
import requests
from typing import List, Dict, Any
import asyncio
import aiohttp
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MOUSER_API_KEY = os.getenv("MOUSER_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# Enhanced mock data for demonstration
MOCK_PARTS = [
    {
        "mpn": "LM358DR",
        "manufacturer": "Texas Instruments",
        "description": "Dual Operational Amplifier, Low Power",
        "category": "OpAmp",
        "package": "SOIC-8",
        "pins": 8,
        "pitch_mm": 1.27,
        "v_range": {"min": 3.0, "max": 32.0, "unit": "V"},
        "temp_range": {"min": -40, "max": 85, "unit": "°C"},
        "stock": 1500,
        "unit_price": 0.45,
        "lifecycle": "ACTIVE",
        "rohs": True,
        "datasheet_url": "https://www.ti.com/lit/ds/symlink/lm358.pdf",
        "provenance": ["pdf://lm358.pdf#p=1:electrical_characteristics"]
    },
    {
        "mpn": "LM358N",
        "manufacturer": "Texas Instruments", 
        "description": "Dual Operational Amplifier, Low Power",
        "category": "OpAmp",
        "package": "DIP-8",
        "pins": 8,
        "pitch_mm": 2.54,
        "v_range": {"min": 3.0, "max": 32.0, "unit": "V"},
        "temp_range": {"min": -40, "max": 85, "unit": "°C"},
        "stock": 800,
        "unit_price": 0.52,
        "lifecycle": "ACTIVE",
        "rohs": True,
        "datasheet_url": "https://www.ti.com/lit/ds/symlink/lm358.pdf",
        "provenance": ["pdf://lm358.pdf#p=1:electrical_characteristics"]
    },
    {
        "mpn": "TL072CP",
        "manufacturer": "Texas Instruments",
        "description": "Dual Low-Noise JFET-Input Operational Amplifier",
        "category": "OpAmp", 
        "package": "DIP-8",
        "pins": 8,
        "pitch_mm": 2.54,
        "v_range": {"min": 6.0, "max": 36.0, "unit": "V"},
        "temp_range": {"min": 0, "max": 70, "unit": "°C"},
        "stock": 2000,
        "unit_price": 0.78,
        "lifecycle": "ACTIVE",
        "rohs": True,
        "datasheet_url": "https://www.ti.com/lit/ds/symlink/tl072.pdf",
        "provenance": ["pdf://tl072.pdf#p=2:electrical_characteristics"]
    },
    {
        "mpn": "LM1117-3.3",
        "manufacturer": "Texas Instruments",
        "description": "800mA Low-Dropout Linear Regulator, 3.3V",
        "category": "LDO",
        "package": "SOT-223",
        "pins": 4,
        "pitch_mm": 2.3,
        "v_range": {"min": 4.75, "max": 15.0, "unit": "V"},
        "temp_range": {"min": -40, "max": 125, "unit": "°C"},
        "stock": 5000,
        "unit_price": 0.89,
        "lifecycle": "ACTIVE",
        "rohs": True,
        "datasheet_url": "https://www.ti.com/lit/ds/symlink/lm1117.pdf",
        "provenance": ["pdf://lm1117.pdf#p=3:electrical_characteristics"]
    }
]

MOCK_REPLACEMENTS = {
    "LM358DR": [
        {
            "mpn": "LM358N",
            "manufacturer": "Texas Instruments",
            "fff_score": 0.95,
            "form_score": 0.80,  # Different package (DIP vs SOIC)
            "fit_score": 1.00,   # Same electrical specs
            "func_score": 1.00,  # Same functionality
            "reasons": [
                "Same manufacturer and part family",
                "Identical electrical characteristics",
                "Same pin count (8 pins)",
                "Same voltage range (3-32V)",
                "Same temperature range (-40°C to 85°C)",
                "Package difference: DIP-8 vs SOIC-8 (form factor only)"
            ],
            "stock": 800,
            "unit_price": 0.52,
            "lifecycle": "ACTIVE",
            "provenance": ["pdf://lm358.pdf#p=1:electrical_characteristics"]
        },
        {
            "mpn": "TL072CP",
            "manufacturer": "Texas Instruments",
            "fff_score": 0.75,
            "form_score": 0.80,  # Different package
            "fit_score": 0.70,   # Different voltage range
            "func_score": 0.75,  # Different input type
            "reasons": [
                "Same manufacturer",
                "Same pin count (8 pins)",
                "Higher minimum voltage (6V vs 3V)",
                "Narrower temperature range (0°C to 70°C)",
                "JFET input vs CMOS input",
                "Package difference: DIP-8 vs SOIC-8"
            ],
            "stock": 2000,
            "unit_price": 0.78,
            "lifecycle": "ACTIVE",
            "provenance": ["pdf://tl072.pdf#p=2:electrical_characteristics"]
        }
    ]
}

class PartSyncDemo:
    def __init__(self):
        self.mouser_api_key = MOUSER_API_KEY
        self.llama_cloud_key = LLAMA_CLOUD_API_KEY
        self.apify_token = APIFY_API_TOKEN
        
    async def test_mouser_api(self) -> bool:
        """Test if Mouser API is working"""
        if not self.mouser_api_key:
            return False
            
        try:
            url = "https://api.mouser.com/api/v1/search/keyword"
            headers = {"Content-Type": "application/json"}
            data = {
                "SearchByKeywordRequest": {
                    "keyword": "LM358",
                    "records": 1,
                    "searchOptions": "InStock"
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}?apiKey={self.mouser_api_key}",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    return "Errors" not in result
        except Exception as e:
            print(f"Mouser API test failed: {e}")
            return False
    
    async def search_parts(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for parts using Mouser API or mock data"""
        if not query.strip():
            return []
        
        # Try Mouser API first
        if await self.test_mouser_api():
            try:
                url = "https://api.mouser.com/api/v1/search/keyword"
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
                        f"{url}?apiKey={self.mouser_api_key}",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        result = await response.json()
                        
                        if "Errors" in result:
                            print(f"Mouser API error: {result['Errors']}")
                            return self._get_mock_parts(query, limit)
                        
                        parts = []
                        for item in result.get("SearchResults", {}).get("Parts", []):
                            part = {
                                "mpn": item.get("MfrPartNumber", ""),
                                "manufacturer": item.get("Manufacturer", ""),
                                "description": item.get("Description", ""),
                                "category": "Unknown",
                                "package": item.get("Package", ""),
                                "stock": item.get("Availability", 0),
                                "unit_price": item.get("PriceBreaks", [{}])[0].get("Price", 0) if item.get("PriceBreaks") else 0,
                                "lifecycle": "ACTIVE",
                                "rohs": True,
                                "datasheet_url": item.get("DataSheetUrl", ""),
                                "provenance": ["mouser_api"]
                            }
                            parts.append(part)
                        
                        return parts
                        
            except Exception as e:
                print(f"Mouser API search failed: {e}")
                return self._get_mock_parts(query, limit)
        else:
            return self._get_mock_parts(query, limit)
    
    def _get_mock_parts(self, query: str, limit: int) -> List[Dict]:
        """Get mock parts matching the query"""
        query_lower = query.lower()
        matching_parts = []
        
        for part in MOCK_PARTS:
            if (query_lower in part["mpn"].lower() or 
                query_lower in part["description"].lower() or
                query_lower in part["manufacturer"].lower() or
                query_lower in part["category"].lower()):
                matching_parts.append(part)
                
        return matching_parts[:limit]
    
    def find_replacements(self, target_mpn: str) -> List[Dict]:
        """Find replacement parts using FFF scoring"""
        return MOCK_REPLACEMENTS.get(target_mpn, [])
    
    async def trigger_apify_crawl(self, search_query: str) -> Dict:
        """Trigger Apify crawler for Mouser.com"""
        # Mock response for demo
        return {
            "status": "success",
            "run_id": f"apify_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": f"Apify crawler triggered for query: {search_query}",
            "estimated_duration": "2-5 minutes",
            "llama_cloud_integration": "PDFs will be processed through LlamaCloud Parse and Extract pipelines"
        }

# Initialize demo client
demo_client = PartSyncDemo()

def format_part_info(part: Dict) -> str:
    """Format part information for display"""
    return f"""
**{part['mpn']}** - {part['manufacturer']}
{part['description']}

**Package:** {part['package']} ({part.get('pins', 'N/A')} pins)
**Stock:** {part.get('stock', 'N/A')} units
**Price:** ${part.get('unit_price', 'N/A')}
**Lifecycle:** {part.get('lifecycle', 'Unknown')}
**RoHS:** {'Yes' if part.get('rohs') else 'No'}

**Datasheet:** {part.get('datasheet_url', 'Not available')}
**Provenance:** {', '.join(part.get('provenance', []))}
"""

def format_replacement_info(replacement: Dict) -> str:
    """Format replacement information for display"""
    return f"""
**{replacement['mpn']}** - {replacement['manufacturer']}
**FFF Score:** {replacement['fff_score']:.2f} (Form: {replacement['form_score']:.2f}, Fit: {replacement['fit_score']:.2f}, Function: {replacement['func_score']:.2f})

**Stock:** {replacement.get('stock', 'N/A')} units
**Price:** ${replacement.get('unit_price', 'N/A')}
**Lifecycle:** {replacement.get('lifecycle', 'Unknown')}

**Compatibility Reasons:**
{chr(10).join(f"• {reason}" for reason in replacement['reasons'])}

**Provenance:** {', '.join(replacement.get('provenance', []))}
"""

async def search_parts_async(query: str, limit: int) -> str:
    """Search for parts using Mouser API or mock data"""
    if not query.strip():
        return "Please enter a search query."
    
    parts = await demo_client.search_parts(query, limit)
    
    if not parts:
        return f"No parts found for query: {query}"
    
    result = f"Found {len(parts)} parts for '{query}':\n\n"
    for i, part in enumerate(parts, 1):
        result += f"**{i}.** {format_part_info(part)}\n"
        result += "---\n"
    
    return result

async def find_replacements_async(target_mpn: str) -> str:
    """Find replacement parts for a given MPN"""
    if not target_mpn.strip():
        return "Please enter a target MPN."
    
    replacements = demo_client.find_replacements(target_mpn)
    
    if not replacements:
        return f"No replacement candidates found for {target_mpn}"
    
    result = f"**Replacement candidates for {target_mpn}:**\n\n"
    for i, replacement in enumerate(replacements, 1):
        result += f"**{i}.** {format_replacement_info(replacement)}\n"
        result += "---\n"
    
    return result

async def trigger_crawl_async(search_query: str) -> str:
    """Trigger Apify crawler for Mouser.com"""
    if not search_query.strip():
        return "Please enter a search query for crawling."
    
    result = await demo_client.trigger_apify_crawl(search_query)
    
    return f"""
**Apify Crawler Status:**
- **Status:** {result['status']}
- **Run ID:** {result['run_id']}
- **Message:** {result['message']}
- **Estimated Duration:** {result['estimated_duration']}
- **LlamaCloud Integration:** {result['llama_cloud_integration']}

**Next Steps:**
1. Wait for the crawler to complete
2. Check the status using the run ID
3. Collect results when ready
4. Process datasheets with LlamaParse
5. Extract structured data with LlamaIndex
6. Enrich with Mouser API data
"""

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="PartSync - Electronic Parts Replacement Engine",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .part-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            background: #f9f9f9;
        }
        .replacement-card {
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            background: #f1f8e9;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🔧 PartSync - Electronic Parts Replacement Engine
        
        **Powered by LlamaIndex, LlamaParse, LlamaCloud, and Apify**
        
        Find compatible replacement parts using Form-Fit-Function (FFF) scoring with real-time Mouser.com data.
        """)
        
        with gr.Tabs():
            
            # Part Search Tab
            with gr.Tab("🔍 Part Search"):
                gr.Markdown("### Search for electronic parts using Mouser API")
                
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
                
                search_btn = gr.Button("🔍 Search Parts", variant="primary")
                search_output = gr.Markdown(label="Search Results")
                
                search_btn.click(
                    fn=search_parts_async,
                    inputs=[search_query, search_limit],
                    outputs=search_output
                )
            
            # Find Replacements Tab
            with gr.Tab("🔄 Find Replacements"):
                gr.Markdown("### Find compatible replacement parts using FFF scoring")
                
                with gr.Row():
                    target_mpn = gr.Textbox(
                        label="Target MPN",
                        placeholder="Enter the part number you want to replace (e.g., LM358DR)",
                        value="LM358DR"
                    )
                
                replace_btn = gr.Button("🔄 Find Replacements", variant="primary")
                replace_output = gr.Markdown(label="Replacement Candidates")
                
                replace_btn.click(
                    fn=find_replacements_async,
                    inputs=[target_mpn],
                    outputs=replace_output
                )
            
            # Live Ingestion Tab
            with gr.Tab("📥 Live Ingestion"):
                gr.Markdown("### Trigger live data ingestion from Mouser.com")
                
                with gr.Row():
                    crawl_query = gr.Textbox(
                        label="Crawl Query",
                        placeholder="Enter search terms for Apify crawler (e.g., LM358, LDO 3.3V)",
                        value="LM358"
                    )
                
                crawl_btn = gr.Button("🚀 Start Crawl", variant="primary")
                crawl_output = gr.Markdown(label="Crawl Status")
                
                crawl_btn.click(
                    fn=trigger_crawl_async,
                    inputs=[crawl_query],
                    outputs=crawl_output
                )
                
                gr.Markdown("""
                ### Ingestion Pipeline
                1. **Apify Crawler** → Mouser.com
                2. **PDF Download** → Datasheets
                3. **LlamaParse** → Extract specs
                4. **LlamaIndex** → Structure data
                5. **LlamaCloud** → Process PDFs
                6. **Mouser API** → Enrich metadata
                """)
            
            # System Status Tab
            with gr.Tab("⚙️ System Status"):
                gr.Markdown("### API Status and Configuration")
                
                status_output = gr.Markdown(label="System Status")
                
                def get_system_status():
                    status = f"""
                    **System Status:**
                    
                    **OpenAI API:** {'✅ Configured' if OPENAI_API_KEY else '❌ Not configured'}
                    **Mouser API:** {'✅ Configured' if MOUSER_API_KEY else '❌ Not configured'}
                    **LlamaCloud API:** {'✅ Configured' if LLAMA_CLOUD_API_KEY else '❌ Not configured'}
                    **Apify API:** {'✅ Configured' if APIFY_API_TOKEN else '❌ Not configured'}
                    
                    **LlamaCloud Dashboards:**
                    - [Parse Dashboard](https://cloud.llamaindex.ai/project/8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe/parse)
                    - [Extract Dashboard](https://cloud.llamaindex.ai/project/8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe/extract)
                    
                    **Note:** If APIs are not available, the system will use mock data for demonstration.
                    """
                    return status
                
                status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                status_btn.click(fn=get_system_status, outputs=status_output)
                
                # Show initial status
                interface.load(fn=get_system_status, outputs=status_output)
    
    return interface

def main():
    """Main function to launch the Gradio app"""
    print("🚀 Starting PartSync Demo Gradio App...")
    print(f"OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
    print(f"Mouser API Key: {'✅ Set' if MOUSER_API_KEY else '❌ Not set'}")
    print(f"LlamaCloud API Key: {'✅ Set' if LLAMA_CLOUD_API_KEY else '❌ Not set'}")
    print(f"Apify API Token: {'✅ Set' if APIFY_API_TOKEN else '❌ Not set'}")
    
    interface = create_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("📱 The app will open in your browser at http://localhost:7860")
    print("🛑 Press Ctrl+C to stop the server")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
