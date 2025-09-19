import gradio as gr
import pandas as pd

# Sample data
SAMPLE_PARTS = [
    ["LM358DR", "Texas Instruments", "OpAmp", "SOIC-8", 8, "3V-32V", "-40°C to 85°C", True, "ACTIVE", 1500, 0.45],
    ["LM358N", "Texas Instruments", "OpAmp", "DIP-8", 8, "3V-32V", "-40°C to 85°C", True, "ACTIVE", 800, 0.52],
    ["TL072CP", "Texas Instruments", "OpAmp", "DIP-8", 8, "3V-36V", "0°C to 70°C", True, "ACTIVE", 2000, 0.38]
]

def search_parts(query, category, manufacturer):
    """Search for parts"""
    if not query and category == "All" and manufacturer == "All":
        return pd.DataFrame(SAMPLE_PARTS, columns=["MPN", "Manufacturer", "Category", "Package", "Pins", "Voltage", "Temperature", "RoHS", "Lifecycle", "Stock", "Price"])
    
    filtered = []
    for part in SAMPLE_PARTS:
        if (not query or query.lower() in part[0].lower()) and \
           (category == "All" or part[2] == category) and \
           (manufacturer == "All" or part[1] == manufacturer):
            filtered.append(part)
    
    return pd.DataFrame(filtered, columns=["MPN", "Manufacturer", "Category", "Package", "Pins", "Voltage", "Temperature", "RoHS", "Lifecycle", "Stock", "Price"])

def find_replacements(mpn):
    """Find replacements for a part"""
    if not mpn:
        return pd.DataFrame()
    
    # Mock replacement data
    replacements = [
        ["LM358N", "Texas Instruments", 0.85, 0.7, 0.9, 0.95, "Different package, same specs", 800, 0.52],
        ["TL072CP", "Texas Instruments", 0.75, 0.7, 0.8, 0.75, "JFET input, similar specs", 2000, 0.38],
        ["LM358DR", "Texas Instruments", 0.95, 1.0, 0.9, 0.95, "Exact match", 1500, 0.45]
    ]
    
    return pd.DataFrame(replacements, columns=["MPN", "Manufacturer", "Total Score", "Form", "Fit", "Function", "Reasons", "Stock", "Price"])

def crawl_mouser(query, pages):
    """Simulate Mouser crawling"""
    return f"""
🔄 **Crawling Mouser for: {query}**

✅ **Crawl Complete!**
- **Pages Crawled**: {pages}
- **Parts Found**: 15
- **Datasheets Downloaded**: 8
- **Processing Status**: Complete

**New Parts Added:**
- LM358DR (Texas Instruments)
- LM358N (Texas Instruments) 
- TL072CP (Texas Instruments)

**Next Steps:**
1. Parts processed with LlamaParse
2. Specifications extracted using LlamaIndex
3. Data enriched with Mouser API
4. Ready for replacement search!
    """

# Create the interface
with gr.Blocks(title="PartSync - AI-Powered Part Replacement Engine") as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
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
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter MPN, manufacturer, or description...", value="LM358")
                    with gr.Row():
                        category = gr.Dropdown(["All", "OpAmp", "LDO", "Logic", "MOSFET"], value="All", label="Category")
                        manufacturer = gr.Dropdown(["All", "Texas Instruments", "STMicroelectronics", "Vishay"], value="All", label="Manufacturer")
                    search_btn = gr.Button("🔍 Search Parts", variant="primary")
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                        <h4>📊 Search Stats</h4>
                        <p><strong>Total Parts:</strong> 1,247</p>
                        <p><strong>Categories:</strong> 12</p>
                        <p><strong>Manufacturers:</strong> 45</p>
                        <p><strong>Last Updated:</strong> 2 min ago</p>
                    </div>
                    """)
            
            search_results = gr.Dataframe(
                headers=["MPN", "Manufacturer", "Category", "Package", "Pins", "Voltage", "Temperature", "RoHS", "Lifecycle", "Stock", "Price"],
                datatype=["str", "str", "str", "str", "number", "str", "str", "bool", "str", "number", "number"],
                interactive=False
            )
        
        # Replacement Tab
        with gr.Tab("🔄 Find Replacements"):
            with gr.Row():
                with gr.Column(scale=2):
                    replacement_mpn = gr.Textbox(label="Target Part MPN", placeholder="Enter MPN to find replacements for...", value="LM358DR")
                    gr.HTML("<h4>🎛️ FFF Scoring Weights</h4>")
                    with gr.Row():
                        form_weight = gr.Slider(0, 1, 0.5, step=0.1, label="Form Factor")
                        fit_weight = gr.Slider(0, 1, 0.3, step=0.1, label="Electrical Fit") 
                        func_weight = gr.Slider(0, 1, 0.2, step=0.1, label="Function")
                    replace_btn = gr.Button("🔄 Find Replacements", variant="primary")
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
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
                interactive=False
            )
        
        # Data Ingestion Tab
        with gr.Tab("🔄 Live Ingestion"):
            with gr.Row():
                with gr.Column(scale=2):
                    crawl_query = gr.Textbox(label="Mouser Search Query", placeholder="Enter search terms for Mouser crawling...", value="LM358 op-amp")
                    max_pages = gr.Slider(1, 10, 3, step=1, label="Max Pages to Crawl")
                    crawl_btn = gr.Button("🔄 Start Crawling", variant="primary")
                
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                        <h4>🔄 Ingestion Pipeline</h4>
                        <p>1. <strong>Apify Crawler</strong> → Mouser.com</p>
                        <p>2. <strong>PDF Download</strong> → Datasheets</p>
                        <p>3. <strong>LlamaParse</strong> → Extract specs</p>
                        <p>4. <strong>LlamaIndex</strong> → Structure data</p>
                        <p>5. <strong>Mouser API</strong> → Enrich metadata</p>
                    </div>
                    """)
            
            crawl_output = gr.Markdown()
    
    # Event handlers
    search_btn.click(search_parts, [search_query, category, manufacturer], search_results)
    replace_btn.click(find_replacements, replacement_mpn, replacement_results)
    crawl_btn.click(crawl_mouser, [crawl_query, max_pages], crawl_output)

if __name__ == "__main__":
    print("🚀 Starting PartSync Gradio App...")
    print("📱 Access the interface at: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
