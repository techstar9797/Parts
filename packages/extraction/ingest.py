import os, glob
from typing import List
from llama_index.readers.llama_parse import LlamaParseReader

def load_docs_from_dir(path: str):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    reader = LlamaParseReader(api_key=api_key, use_vendor_multimodal_ocr=True)
    pdfs = []
    for ext in ("*.pdf", "*.PDF"):
        pdfs.extend(glob.glob(os.path.join(path, ext)))
    if not pdfs:
        return []
    docs = reader.load_data(pdfs)
    return docs
