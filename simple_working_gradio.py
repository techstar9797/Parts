#!/usr/bin/env python3
"""
PartSync - Simple Working Gradio App
"""

import gradio as gr
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# API Configuration
MOUSER_API_KEY = os.getenv("MOUSER_API_KEY", "85143bc8-9d3b-495a-996f-1e180e42f402")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "llx-Ul6Kjdu37jzqTW3BRGWWH785oguwSSXEboNk10G2bbPAg8Yj")

# Mock data
MOCK_PARTS = [
    {
        "mpn": "LM358DR",
        "manufacturer": "Texas Instruments",
        "description": "Dual Operational Amplifier",
        "package": "SOIC-8",
        "stock": 1500,
        "unit_price": 0.45,
        "lifecycle": "ACTIVE"
    },
    {
        "mpn": "LM358N", 
        "manufacturer": "Texas Instruments",
        "description": "Dual Operational Amplifier",
        "package": "DIP-8",
        "stock": 800,
        "unit_price": 0.52,
        "lifecycle": "ACTIVE"
    }
]

async def search_parts(query: str) -> str:
    """Search for parts using Mouser API or mock data"""
    if not query.strip():
        return "Please enter a search query."
    
    # Try Mouser API first
    try:
        url = "https://api.mouser.com/api/v1/search/keyword"
        headers = {"Content-Type": "application/json"}
        data = {
            "SearchByKeywordRequest": {
                "keyword": query,
                "records": 5,
                "searchOptions": "InStock"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}?apiKey={MOUSER_API_KEY}",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                result = await response.json()
                
                if "Errors" in result:
                    print(f"Mouser API error: {result['Errors']}")
                    # Fall back to mock data
                    parts = [p for p in MOCK_PARTS if query.lower() in p["mpn"].lower()]
                else:
                    parts = []
                    for item in result.get("SearchResults", {}).get("Parts", []):
                        part = {
                            "mpn": item.get("MfrPartNumber", ""),
                            "manufacturer": item.get("Manufacturer", ""),
                            "description": item.get("Description", ""),
                            "package": item.get("Package", ""),
                            "stock": item.get("Availability", 0),
                            "unit_price": item.get("PriceBreaks", [{}])[0].get("Price", 0) if item.get("PriceBreaks") else 0,
                            "lifecycle": "ACTIVE"
                        }
                        parts.append(part)
    except Exception as e:
        print(f"Mouser API failed: {e}")
        parts = [p for p in MOCK_PARTS if query.lower() in p["mpn"].lower()]
    
    if not parts:
        return f"No parts found for query: {query}"
    
    result = f"Found {len(parts)} parts for '{query}':\n\n"
    for i, part in enumerate(parts, 1):
        result += f"**{i}.** {part['mpn']} - {part['manufacturer']}\n"
        result += f"    {part['description']}\n"
        result += f"    Package: {part['package']}, Stock: {part['stock']}, Price: ${part['unit_price']}\n\n"
    
    return result

def create_interface():
    with gr.Blocks(title="PartSync - Parts Search") as interface:
        gr.Markdown("# 🔧 PartSync - Electronic Parts Search")
        
        with gr.Row():
            search_query = gr.Textbox(
                label="Search Query",
                placeholder="Enter part number or keywords",
                value="LM358"
            )
            search_btn = gr.Button("🔍 Search", variant="primary")
        
        search_output = gr.Markdown(label="Results")
        
        search_btn.click(
            fn=search_parts,
            inputs=[search_query],
            outputs=search_output
        )
        
        gr.Markdown(f"""
        **API Status:**
        - Mouser API: {'✅ Configured' if MOUSER_API_KEY else '❌ Not configured'}
        - LlamaCloud API: {'✅ Configured' if LLAMA_CLOUD_API_KEY else '❌ Not configured'}
        """)
    
    return interface

def main():
    print("🚀 Starting PartSync Gradio App...")
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
