import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.extractors import PydanticProgramExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.parsers.file import PDFParser
from llama_parse import LlamaParse

from .schemas_partsync import PartRecordPS
from .extract import extract_part_from_text

logger = logging.getLogger(__name__)

class LlamaIndexExtractor:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.llama_cloud_api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable is required")
        
        # Configure LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", api_key=self.openai_api_key)
        
        # Initialize LlamaParse
        self.parser = LlamaParse(
            api_key=self.llama_cloud_api_key,
            result_type="markdown",
            parsing_instruction="Extract all technical specifications, pinout information, and part details from electronic component datasheets. Preserve tables and structured data."
        )
        
        # Initialize PDF parser as fallback
        self.pdf_parser = PDFParser()
        
        # Initialize Pydantic program extractor
        self.extractor = PydanticProgramExtractor(
            program_cls=PartRecordPS,
            llm=Settings.llm
        )
    
    async def parse_documents(self, file_paths: List[str]) -> List[Document]:
        """Parse PDF documents using LlamaParse"""
        documents = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Parsing document: {file_path}")
                
                # Try LlamaParse first
                try:
                    parsed_docs = await self.parser.aload_data(file_path)
                    documents.extend(parsed_docs)
                    logger.info(f"Successfully parsed with LlamaParse: {file_path}")
                except Exception as e:
                    logger.warning(f"LlamaParse failed for {file_path}: {e}, trying fallback")
                    # Fallback to regular PDF parser
                    parsed_docs = self.pdf_parser.load_data(file_path)
                    documents.extend(parsed_docs)
                    logger.info(f"Successfully parsed with fallback: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue
        
        return documents
    
    def extract_parts_from_documents(self, documents: List[Document]) -> List[PartRecordPS]:
        """Extract PartRecordPS from parsed documents"""
        parts = []
        
        for doc in documents:
            try:
                # Extract using Pydantic program
                extracted_data = self.extractor.extract([doc])
                
                if extracted_data and len(extracted_data) > 0:
                    for data in extracted_data[0]:
                        if isinstance(data, PartRecordPS):
                            # Add provenance information
                            data.provenance.append(f"pdf://{doc.metadata.get('file_path', 'unknown')}")
                            parts.append(data)
                            logger.info(f"Extracted part: {data.mpn}")
                
                # Fallback to custom extraction if Pydantic program fails
                if not extracted_data or len(extracted_data) == 0:
                    logger.info("Pydantic extraction failed, trying custom extractor")
                    custom_part = extract_part_from_text(doc.text)
                    if custom_part:
                        custom_part.provenance.append(f"pdf://{doc.metadata.get('file_path', 'unknown')}")
                        parts.append(custom_part)
                        logger.info(f"Extracted part with custom extractor: {custom_part.mpn}")
                        
            except Exception as e:
                logger.error(f"Failed to extract part from document: {e}")
                continue
        
        return parts
    
    async def process_datasheets(self, file_paths: List[str]) -> List[PartRecordPS]:
        """Complete workflow: parse PDFs and extract parts"""
        # Parse documents
        documents = await self.parse_documents(file_paths)
        logger.info(f"Parsed {len(documents)} documents")
        
        # Extract parts
        parts = self.extract_parts_from_documents(documents)
        logger.info(f"Extracted {len(parts)} parts")
        
        return parts
    
    def create_vector_index(self, parts: List[PartRecordPS]) -> VectorStoreIndex:
        """Create a vector index for semantic search"""
        # Convert parts to documents
        documents = []
        for part in parts:
            # Create a text representation for indexing
            text = f"""
            MPN: {part.mpn}
            Manufacturer: {part.manufacturer}
            Category: {part.category}
            Description: {part.value_display or 'N/A'}
            Package: {part.package.name}
            Pins: {part.package.pins or 'N/A'}
            Voltage Range: {part.v_range.min}V - {part.v_range.max}V
            Temperature Range: {part.temp_range_c.min}°C - {part.temp_range_c.max}°C
            RoHS: {part.rohs or 'Unknown'}
            Lifecycle: {part.lifecycle.status if part.lifecycle else 'Unknown'}
            """
            
            doc = Document(
                text=text,
                metadata={
                    "mpn": part.mpn,
                    "manufacturer": part.manufacturer,
                    "category": part.category,
                    "package": part.package.name,
                    "pins": part.package.pins,
                    "voltage_min": part.v_range.min,
                    "voltage_max": part.v_range.max,
                    "temp_min": part.temp_range_c.min,
                    "temp_max": part.temp_range_c.max,
                    "rohs": part.rohs,
                    "lifecycle": part.lifecycle.status if part.lifecycle else None
                }
            )
            documents.append(doc)
        
        # Create vector index
        index = VectorStoreIndex.from_documents(documents)
        return index

# Convenience function
async def extract_from_datasheets(file_paths: List[str]) -> List[PartRecordPS]:
    """Convenience function to extract parts from datasheet PDFs"""
    extractor = LlamaIndexExtractor()
    return await extractor.process_datasheets(file_paths)
