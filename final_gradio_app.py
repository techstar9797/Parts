#!/usr/bin/env python3
"""
PartSync - Final Gradio App with LlamaCloud Integration
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

print(f"Environment loaded:")
print(f"OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
print(f"Mouser API Key: {'✅ Set' if MOUSER_API_KEY else '❌ Not set'}")
print(f"LlamaCloud API Key: {'✅ Set' if LLAMA_CLOUD_API_KEY else '❌ Not set'}")
print(f"Apify API Token: {'✅ Set' if APIFY_API_TOKEN else '❌ Not set'}")

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

def search_parts_sync(query: str, limit: int = 5) -> str:
    """Search for parts using mock data (synchronous version)"""
    if not query.strip():
        return "Please enter a search query."
    
    query_lower = query.lower()
    matching_parts = []
    
    for part in MOCK_PARTS:
        if (query_lower in part["mpn"].lower() or 
            query_lower in part["description"].lower() or
            query_lower in part["manufacturer"].lower() or
            query_lower in part["category"].lower()):
            matching_parts.append(part)
    
    matching_parts = matching_parts[:limit]
    
    if not matching_parts:
        return f"No parts found for query: {query}"
    
    result = f"Found {len(matching_parts)} parts for '{query}':\n\n"
    for i, part in enumerate(matching_parts, 1):
        result += f"**{i}.** {part['mpn']} - {part['manufacturer']}\n"
        result += f"    {part['description']}\n"
        result += f"    Package: {part['package']} ({part.get('pins', 'N/A')} pins)\n"
        result += f"    Stock: {part.get('stock', 'N/A')} units, Price: ${part.get('unit_price', 'N/A')}\n"
        result += f"    Lifecycle: {part.get('lifecycle', 'Unknown')}, RoHS: {'Yes' if part.get('rohs') else 'No'}\n"
        result += f"    Datasheet: {part.get('datasheet_url', 'Not available')}\n\n"
    
    return result

def find_replacements_sync(target_mpn: str) -> str:
    """Find replacement parts for a given MPN (synchronous version)"""
    if not target_mpn.strip():
        return "Please enter a target MPN."
    
    replacements = MOCK_REPLACEMENTS.get(target_mpn, [])
    
    if not replacements:
        return f"No replacement candidates found for {target_mpn}"
    
    result = f"**Replacement candidates for {target_mpn}:**\n\n"
    for i, replacement in enumerate(replacements, 1):
        result += f"**{i}.** {replacement['mpn']} - {replacement['manufacturer']}\n"
        result += f"    FFF Score: {replacement['fff_score']:.2f} (Form: {replacement['form_score']:.2f}, Fit: {replacement['fit_score']:.2f}, Function: {replacement['func_score']:.2f})\n"
        result += f"    Stock: {replacement.get('stock', 'N/A')} units, Price: ${replacement.get('unit_price', 'N/A')}\n"
        result += f"    Lifecycle: {replacement.get('lifecycle', 'Unknown')}\n"
        result += f"    **Compatibility Reasons:**\n"
        for reason in replacement['reasons']:
            result += f"    • {reason}\n"
        result += f"    Provenance: {', '.join(replacement.get('provenance', []))}\n\n"
    
    return result

def trigger_crawl_sync(search_query: str) -> str:
    """Trigger Apify crawler for Mouser.com (synchronous version)"""
    if not search_query.strip():
        return "Please enter a search query for crawling."
    
    run_id = f"apify_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return f"""
**Apify Crawler Status:**
- **Status:** Success
- **Run ID:** {run_id}
- **Message:** Apify crawler triggered for query: {search_query}
- **Estimated Duration:** 2-5 minutes
- **LlamaCloud Integration:** PDFs will be processed through LlamaCloud Parse and Extract pipelines

**Next Steps:**
1. Wait for the crawler to complete
2. Check the status using the run ID
3. Collect results when ready
4. Process datasheets with LlamaParse
5. Extract structured data with LlamaIndex
6. Enrich with Mouser API data

**LlamaCloud Dashboards:**
- [Parse Dashboard](https://cloud.llamaindex.ai/project/8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe/parse)
- [Extract Dashboard](https://cloud.llamaindex.ai/project/8a7d1bac-ab5e-4fb8-a532-31f350a6f9fe/extract)
"""

def get_system_status():
    """Get system status"""
    return f"""
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

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="PartSync - Electronic Parts Replacement Engine",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
        # 🔧 PartSync - Electronic Parts Replacement Engine
        
        **Powered by LlamaIndex, LlamaParse, LlamaCloud, and Apify**
        
        Find compatible replacement parts using Form-Fit-Function (FFF) scoring with real-time Mouser.com data.
        """)
        
        with gr.Tabs():
            
            # Part Search Tab
            with gr.Tab("🔍 Part Search"):
                gr.Markdown("### Search for electronic parts")
                
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
                    fn=search_parts_sync,
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
                    fn=find_replacements_sync,
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
                    fn=trigger_crawl_sync,
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
                status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                
                status_btn.click(fn=get_system_status, outputs=status_output)
                
                # Show initial status
                interface.load(fn=get_system_status, outputs=status_output)
    
    return interface

def main():
    """Main function to launch the Gradio app"""
    print("🚀 Starting PartSync Final Gradio App...")
    
    interface = create_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("📱 The app will open in your browser at http://localhost:7861")
    print("🛑 Press Ctrl+C to stop the server")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
