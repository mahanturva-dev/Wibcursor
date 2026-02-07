"""
PDF parsing utilities for academic research papers using PyMuPDF.
Handles text extraction, metadata, and proper formatting for academic papers.
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import fitz  # PyMuPDF
from typing import List, Dict, Optional
import re


class PDFParser:
    """Parser for extracting text and metadata from research paper PDFs."""
    
    def __init__(self):
        self.papers: Dict[str, Dict] = {}
    
    def extract_text(self, pdf_path: str, paper_name: str) -> Dict:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            paper_name: Name/identifier for the paper
            
        Returns:
            Dictionary with text, metadata, and page info
        """
        doc = fitz.open(pdf_path)
        
        full_text = []
        pages_text = []
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean and process text
            cleaned_text = self._clean_text(text)
            
            if cleaned_text.strip():
                full_text.append(cleaned_text)
                pages_text.append({
                    'page': page_num + 1,
                    'text': cleaned_text
                })
        
        # Extract metadata
        metadata = doc.metadata
        doc.close()
        
        # Try to extract title and authors from first page
        title, authors = self._extract_title_authors(full_text[0] if full_text else "")
        
        result = {
            'paper_name': paper_name,
            'full_text': '\n\n'.join(full_text),
            'pages': pages_text,
            'num_pages': len(pages_text),
            'metadata': metadata,
            'title': title,
            'authors': authors,
            'file_path': pdf_path
        }
        
        self.papers[paper_name] = result
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text from PDF."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common PDF extraction issues
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def _extract_title_authors(self, first_page_text: str) -> tuple:
        """Extract title and authors from first page."""
        lines = first_page_text.split('\n')[:20]  # First 20 lines usually contain title/authors
        
        title = ""
        authors = ""
        
        # Look for title (usually first few lines, all caps or title case)
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if len(line) > 20 and len(line) < 200:
                # Likely a title
                if not title or len(line) > len(title):
                    title = line
        
        # Look for authors (usually contains "et al" or multiple names)
        for line in lines[5:15]:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['et al', '@', 'university', 'department']):
                authors = line
                break
            elif ',' in line and len(line.split(',')) >= 2:
                authors = line
                break
        
        return title[:200] if title else "Unknown Title", authors[:300] if authors else "Unknown Authors"
    
    def get_paper(self, paper_name: str) -> Optional[Dict]:
        """Get stored paper data."""
        return self.papers.get(paper_name)
    
    def get_all_papers(self) -> Dict[str, Dict]:
        """Get all stored papers."""
        return self.papers
    
    def clear_papers(self):
        """Clear all stored papers."""
        self.papers.clear()
