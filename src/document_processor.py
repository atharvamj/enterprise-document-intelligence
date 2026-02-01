"""
Document processing module for loading and preprocessing financial documents.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
import PyPDF2
from docx import Document
from loguru import logger
from tqdm import tqdm


class DocumentChunk:
    """Represents a chunk of document text with metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
    
    def __repr__(self):
        return f"DocumentChunk(text_length={len(self.text)}, metadata={self.metadata})"


class DocumentProcessor:
    """
    Process and chunk documents for RAG pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.doc_config = config['document_processing']
        self.chunk_size = self.doc_config['chunk_size']
        self.chunk_overlap = self.doc_config['chunk_overlap']
        self.min_chunk_size = self.doc_config['min_chunk_size']
        logger.info("DocumentProcessor initialized")
    
    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load PDF file and extract text with metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page dictionaries with text and metadata
        """
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages.append({
                            'text': text,
                            'page_number': page_num,
                            'filename': os.path.basename(file_path)
                        })
            logger.info(f"Loaded {len(pages)} pages from {file_path}")
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            pages.append({
                                'text': text,
                                'page_number': page_num,
                                'filename': os.path.basename(file_path)
                            })
                logger.info(f"Loaded {len(pages)} pages from {file_path} using PyPDF2")
            except Exception as e2:
                logger.error(f"Failed to load PDF with both libraries: {e2}")
        
        return pages
    
    def load_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List with single text document
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [{
                'text': text,
                'page_number': 1,
                'filename': os.path.basename(file_path)
            }]
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def load_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load Word document.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List with document paragraphs
        """
        try:
            doc = Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs if para.text])
            return [{
                'text': text,
                'page_number': 1,
                'filename': os.path.basename(file_path)
            }]
        except Exception as e:
            logger.error(f"Error loading DOCX file {file_path}: {e}")
            return []
    
    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load document based on file extension.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of page/section dictionaries
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext == '.txt':
            return self.load_txt(file_path)
        elif ext == '.docx':
            return self.load_docx(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return []
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings within window
                for punct in ['. ', '.\n', '! ', '?\n', '? ']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1:
                        end = last_punct + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            # Only add chunk if it meets minimum size
            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_start'] = start
                chunk_metadata['chunk_end'] = end
                chunks.append(DocumentChunk(chunk_text, chunk_metadata))
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start <= 0 or start >= text_length:
                start = end
        
        return chunks
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a document into chunks.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of DocumentChunk objects
        """
        pages = self.load_document(file_path)
        all_chunks = []
        
        for page in pages:
            chunks = self.chunk_text(page['text'], page)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {file_path} into {len(all_chunks)} chunks")
        return all_chunks
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[DocumentChunk]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of all DocumentChunk objects
        """
        all_chunks = []
        supported_formats = self.doc_config['supported_formats']
        
        # Get all files
        path = Path(directory_path)
        if recursive:
            files = [f for f in path.rglob('*') if f.is_file()]
        else:
            files = [f for f in path.glob('*') if f.is_file()]
        
        # Filter by supported formats
        files = [f for f in files if f.suffix.lower().strip('.') in supported_formats]
        
        logger.info(f"Processing {len(files)} documents from {directory_path}")
        
        for file_path in tqdm(files, desc="Processing documents"):
            chunks = self.process_document(str(file_path))
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
