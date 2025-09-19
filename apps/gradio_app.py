import gradio as gr
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
import asyncio
import httpx

# Mock data for demonstration
SAMPLE_PARTS = [
    {
        "mpn": "LM358DR",
        "manufacturer": "Texas Instruments",
        "category": "OpAmp",
        "package": "SOIC-8",
        "pins": 8,
        "v_range": "3V to 32V",
        "temp_range": "-40°C to 85°C",
        "rohs": True,
        "lifecycle": "ACTIVE",
        "stock": 1500,
        "unit_price": 0.45,
        "description": "Dual Operational Amplifier"
    },
    {
        "mpn": "LM358N",
        "manufacturer": "Texas Instruments", 
        "category": "OpAmp",
        "package": "DIP-8",
        "pins": 8,
        "v_range": "3V to 32V",
        "temp_range": "-40°C to 85°C",
        "rohs": True,
        "lifecycle": "ACTIVE",
        "stock": 800,
        "unit_price": 0.52,
        "description": "Dual Operational Amplifier"
    },
    {
        "mpn": "TL072CP",
        "manufacturer": "Texas Instruments",
        "category": "OpAmp", 
        "package": "DIP-8",
        "pins": 8,
        "v_range": "3V to 36V",
        "temp_range": "0°C to 70°C",
        "rohs": True,
        "lifecycle": "ACTIVE",
        "stock": 2000,
        "unit_price": 0.38,
        "description": "Low-Noise JFET-Input Operational Amplifier"
    }
]

SAMPLE_REPLACEMENTS = [
    {
        "mpn": "LM358DR",
        "manufacturer": "Texas Instruments",
        "score": 0.95,
        "form_score": 1.0,
        "fit_score": 0.9,
        "func_score": 0.95,
        "reasons": ["Exact package match", "Same electrical specs", "Same manufacturer"],
        "stock": 1500,
        "unit_price": 0.45
    },
    {
        "mpn": "LM358N", 
        "manufacturer": "Texas Instruments",
        "score": 0.85,
        "form_score": 0.7,
        "fit_score": 0.9,
        "func_score": 0.95,
        "reasons": ["Different package (DIP vs SOIC)", "Same electrical specs", "Same manufacturer"],
        "stock": 800,
        "unit_price": 0.52
    },
    {
        "mpn": "TL072CP",
        "manufacturer": "Texas Instruments", 
        "score": 0.75,
        "form_score": 0.7,
        "fit_score": 0.8,
        "func_score": 0.75,
        "reasons": ["Different package", "Similar electrical specs", "JFET input vs BJT"],
        "stock": 2000,
        "unit_price": 0.38
    }
]

class PartSyncGradioApp:
    def __init__(self):
        self.parts_db = SAMPLE_PARTS.copy()
        self.replacements_db = SAMPLE_REPLACEMENTS.copy()
        
    def search_parts(self, query: str, category: str, manufacturer: str) -> pd.DataFrame:
        """Search for parts based on query and filters"""
        if not query and not category and not manufacturer:
            return pd.DataFrame(self.parts_db)
        
        filtered_parts = []
        for part in self.parts_db:
            match = True
            
            if query and query.lower() not in part["mpn"].lower() and query.lower() not in part["description"].lower():
                match = False
                
            if category and category != "All" and part["category"] != category:
                match = False
                
            if manufacturer and manufacturer != "All" and part["manufacturer"] != manufacturer:
                match = False
                
            if match:
                filtered_parts.append(part)
        
        return pd.DataFrame(filtered_parts)
    
    def find_replacements(self, mpn: str, weights: Dict[str, float]) -> pd.DataFrame:
        """Find replacement candidates for a given MPN"""
        if not mpn:
            return pd.DataFrame()
        
        # Find the target part
        target_part = None
        for part in self.parts_db:
            if part["mpn"].upper() == mpn.upper():
                target_part = part
                break
        
        if not target_part:
            return pd.DataFrame({"Error": [f"Part {mpn} not found"]})
        
        # Mock replacement logic with FFF scoring
        replacements = []
        for replacement in self.replacements_db:
            if replacement["mpn"].upper() != mpn.upper():
                # Apply weights
                weighted_score = (
                    weights["form"] * replacement["form_score"] +
                    weights["fit"] * replacement["fit_score"] + 
                    weights["func"] * replacement["func_score"]
                )
                
                replacement_copy = replacement.copy()
                replacement_copy["weighted_score"] = round(weighted_score, 3)
                replacements.append(replacement_copy)
        
        # Sort by weighted score
        replacements.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        return pd.DataFrame(replacements)
    
    def crawl_mouser(self, query: str, max_pages: int) -> str:
        """Simulate Mouser crawling"""
        if not query:
            return "Please enter a search query"
        
        # Simulate crawling process
        import time
        time.sleep(2)  # Simulate processing time
        
        return f"""
🔄 **Crawling Mouser for: {query}**

✅ **Crawl Complete!**
- **Pages Crawled**: {max_pages}
- **Parts Found**: {len(self.parts_db)}
- **Datasheets Downloaded**: 5
- **Processing Status**: Complete

**New Parts Added:**
- LM358DR (Texas Instruments)
- LM358N (Texas Instruments) 
- TL072CP (Texas Instruments)

**Next Steps:**
1. Parts have been processed with LlamaParse
2. Specifications extracted using LlamaIndex
3. Data enriched with Mouser API
4. Ready for replacement search!
        """
    
    def process_bom(self, bom_text: str) -> str:
        """Process BOM and find replacements"""
        if not bom_text.strip():
            return "Please enter BOM data"
        
        lines = bom_text.strip().split('\n')
        results = []
        
        for i, line in enumerate(lines[:5], 1):  # Process first 5 lines
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    mpn = parts[0]
                    qty = parts[1] if len(parts) > 1 else "1"
                    
                    # Find replacements
                    replacements = self.find_replacements(mpn, {"form": 0.5, "fit": 0.3, "func": 0.2})
                    
                    if not replacements.empty and "Error" not in replacements.columns:
                        best_replacement = replacements.iloc[0]
                        results.append(f"**Line {i}**: {mpn} (Qty: {qty})")
                        results.append(f"  → Best Replacement: {best_replacement['mpn']} (Score: {best_replacement['weighted_score']})")
                        results.append(f"  → Price: ${best_replacement['unit_price']} | Stock: {best_replacement['stock']}")
                        results.append("")
        
        if not results:
            return "No valid BOM lines found. Please format as: MPN QTY"
        
        return "**BOM Replacement Analysis:**\n\n" + "\n".join(results)
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="PartSync - AI-Powered Part Replacement Engine",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                margin: 10px 0;
            }
            """
        ) as demo:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🔧 PartSync</h1>
                <p>AI-Powered Part Replacement Engine with LlamaIndex + Mouser API</p>
                <p>Never lose weeks waiting on obsolete parts - repair your BOM in seconds!</p>
            </div>
            """)
            
            with gr.Tabs():
                
                # Search Tab
                with gr.Tab("🔍 Part Search"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="Enter MPN, manufacturer, or description...",
                                value="LM358"
                            )
                            
                            with gr.Row():
                                category_filter = gr.Dropdown(
                                    choices=["All", "OpAmp", "LDO", "Logic", "MOSFET", "Resistor", "Capacitor"],
                                    value="All",
                                    label="Category"
                                )
                                manufacturer_filter = gr.Dropdown(
                                    choices=["All", "Texas Instruments", "STMicroelectronics", "Vishay", "Microchip"],
                                    value="All", 
                                    label="Manufacturer"
                                )
                            
                            search_btn = gr.Button("🔍 Search Parts", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="metric-card">
                                <h4>📊 Search Stats</h4>
                                <p><strong>Total Parts:</strong> 1,247</p>
                                <p><strong>Categories:</strong> 12</p>
                                <p><strong>Manufacturers:</strong> 45</p>
                                <p><strong>Last Updated:</strong> 2 min ago</p>
                            </div>
                            """)
                    
                    search_results = gr.Dataframe(
                        headers=["MPN", "Manufacturer", "Category", "Package", "Pins", "Voltage Range", "Temperature", "RoHS", "Lifecycle", "Stock", "Price"],
                        datatype=["str", "str", "str", "str", "number", "str", "str", "bool", "str", "number", "number"],
                        interactive=False,
                        wrap=True
                    )
                
                # Replacement Tab
                with gr.Tab("🔄 Find Replacements"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            replacement_mpn = gr.Textbox(
                                label="Target Part MPN",
                                placeholder="Enter MPN to find replacements for...",
                                value="LM358DR"
                            )
                            
                            gr.HTML("<h4>🎛️ FFF Scoring Weights</h4>")
                            with gr.Row():
                                form_weight = gr.Slider(0, 1, 0.5, step=0.1, label="Form Factor")
                                fit_weight = gr.Slider(0, 1, 0.3, step=0.1, label="Electrical Fit") 
                                func_weight = gr.Slider(0, 1, 0.2, step=0.1, label="Function")
                            
                            replace_btn = gr.Button("🔄 Find Replacements", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="metric-card">
                                <h4>🎯 FFF Scoring</h4>
                                <p><strong>Form:</strong> Package, pins, pitch</p>
                                <p><strong>Fit:</strong> Voltage, temp, power</p>
                                <p><strong>Function:</strong> Category-specific specs</p>
                                <br>
                                <p><strong>Hard Gates:</strong></p>
                                <p>• RoHS Compliance</p>
                                <p>• Lifecycle Status</p>
                                <p>• Confidence Score</p>
                            </div>
                            """)
                    
                    replacement_results = gr.Dataframe(
                        headers=["MPN", "Manufacturer", "Total Score", "Form", "Fit", "Function", "Reasons", "Stock", "Price"],
                        datatype=["str", "str", "number", "number", "number", "number", "str", "number", "number"],
                        interactive=False,
                        wrap=True
                    )
                
                # BOM Processing Tab
                with gr.Tab("📋 BOM Processing"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            bom_input = gr.Textbox(
                                label="BOM Data",
                                placeholder="Enter BOM data (one part per line):\nLM358DR 10\nTL072CP 5\n...",
                                lines=10,
                                value="LM358DR 10\nTL072CP 5\nLM358N 2"
                            )
                            
                            process_bom_btn = gr.Button("📋 Process BOM", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="metric-card">
                                <h4>📋 BOM Features</h4>
                                <p>• Auto-replacement suggestions</p>
                                <p>• Risk assessment flags</p>
                                <p>• Stock availability check</p>
                                <p>• Price comparison</p>
                                <p>• Export to CSV/Excel</p>
                            </div>
                            """)
                    
                    bom_output = gr.Markdown()
                
                # Data Ingestion Tab
                with gr.Tab("🔄 Live Ingestion"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            crawl_query = gr.Textbox(
                                label="Mouser Search Query",
                                placeholder="Enter search terms for Mouser crawling...",
                                value="LM358 op-amp"
                            )
                            
                            max_pages = gr.Slider(1, 10, 3, step=1, label="Max Pages to Crawl")
                            
                            crawl_btn = gr.Button("🔄 Start Crawling", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class="metric-card">
                                <h4>🔄 Ingestion Pipeline</h4>
                                <p>1. <strong>Apify Crawler</strong> → Mouser.com</p>
                                <p>2. <strong>PDF Download</strong> → Datasheets</p>
                                <p>3. <strong>LlamaParse</strong> → Extract specs</p>
                                <p>4. <strong>LlamaIndex</strong> → Structure data</p>
                                <p>5. <strong>Mouser API</strong> → Enrich metadata</p>
                            </div>
                            """)
                    
                    crawl_output = gr.Markdown()
                
                # Analytics Tab
                with gr.Tab("📊 Analytics"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("""
                            <div class="metric-card">
                                <h4>📈 System Performance</h4>
                                <p><strong>Search Latency:</strong> 156ms avg</p>
                                <p><strong>Extraction Accuracy:</strong> 94.2%</p>
                                <p><strong>Replacement Quality:</strong> 89.7%</p>
                                <p><strong>Database Size:</strong> 1,247 parts</p>
                            </div>
                            """)
                        
                        with gr.Column():
                            gr.HTML("""
                            <div class="metric-card">
                                <h4>🎯 Recent Activity</h4>
                                <p>• 15 parts searched today</p>
                                <p>• 8 BOMs processed</p>
                                <p>• 3 Mouser crawls completed</p>
                                <p>• 47 replacements suggested</p>
                            </div>
                            """)
                    
                    # Mock charts
                    gr.HTML("""
                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                        <h4>📊 Replacement Success Rate by Category</h4>
                        <p>OpAmp: 92% | LDO: 88% | Logic: 85% | MOSFET: 90%</p>
                        <br>
                        <h4>⏱️ Average Processing Time</h4>
                        <p>Search: 156ms | Replacement: 234ms | BOM: 1.2s | Crawl: 45s</p>
                    </div>
                    """)
            
            # Event handlers
            search_btn.click(
                fn=self.search_parts,
                inputs=[search_query, category_filter, manufacturer_filter],
                outputs=search_results
            )
            
            replace_btn.click(
                fn=self.find_replacements,
                inputs=[replacement_mpn, gr.State({"form": 0.5, "fit": 0.3, "func": 0.2})],
                outputs=replacement_results
            )
            
            process_bom_btn.click(
                fn=self.process_bom,
                inputs=bom_input,
                outputs=bom_output
            )
            
            crawl_btn.click(
                fn=self.crawl_mouser,
                inputs=[crawl_query, max_pages],
                outputs=crawl_output
            )
            
            # Update weights when sliders change
            def update_weights(form, fit, func):
                return {"form": form, "fit": fit, "func": func}
            
            form_weight.change(
                fn=update_weights,
                inputs=[form_weight, fit_weight, func_weight],
                outputs=gr.State()
            )
            
            fit_weight.change(
                fn=update_weights,
                inputs=[form_weight, fit_weight, func_weight],
                outputs=gr.State()
            )
            
            func_weight.change(
                fn=update_weights,
                inputs=[form_weight, fit_weight, func_weight],
                outputs=gr.State()
            )
        
        return demo

def main():
    app = PartSyncGradioApp()
    demo = app.create_interface()
    
    print("🚀 Starting PartSync Gradio App...")
    print("📱 Access the interface at: http://localhost:7860")
    print("🔧 Features: Part Search, FFF Replacement, BOM Processing, Live Ingestion")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
