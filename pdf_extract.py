"""
PDF extraction utilities using PyMuPDF (fitz).
Extracts text from PDF files page by page with metadata.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple
import re


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in extracted text while keeping it readable.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Normalized text with consistent spacing
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with max 2 newlines
    text = re.sub(r'\n\n+', '\n\n', text)
    # Remove spaces at line ends
    text = re.sub(r' +\n', '\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, any]]:
    """
    Extract text from a PDF file page by page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries, one per page with:
        - page_num: 1-based page number
        - text: extracted and normalized text
        - char_count: number of characters
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be opened or read
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Normalize whitespace
            text = normalize_whitespace(text)
            
            # Only add pages with actual content
            if text.strip():
                pages.append({
                    "page_num": page_num + 1,  # 1-based page numbering
                    "text": text,
                    "char_count": len(text)
                })
            else:
                # Empty page - add placeholder
                pages.append({
                    "page_num": page_num + 1,
                    "text": "",
                    "char_count": 0
                })
        
        doc.close()
        
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    return pages


def extract_state_from_filename(filename: str) -> str:
    """
    Extract state code from filename.
    Expected format: {STATE}_*.pdf (e.g., CA_Caltrans_Maintenance_Manual.pdf)
    
    Args:
        filename: Name of the PDF file
        
    Returns:
        State code (CA, TX, or WA)
        
    Raises:
        ValueError: If state cannot be determined
    """
    # Get first part before underscore
    parts = filename.split('_')
    if parts:
        state = parts[0].upper()
        if state in ["CA", "TX", "WA"]:
            return state
    
    raise ValueError(f"Cannot determine state from filename: {filename}")


def extract_title_from_filename(filename: str) -> str:
    """
    Extract a friendly title from the PDF filename.
    
    Args:
        filename: Name of the PDF file (without extension)
        
    Returns:
        Friendly title
    """
    # Remove .pdf extension if present
    name = filename.replace('.pdf', '')
    
    # Replace underscores with spaces
    title = name.replace('_', ' ')
    
    return title


def extract_all_pdfs(pdf_dir: Path) -> List[Dict[str, any]]:
    """
    Extract text from all PDF files in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of documents, each containing:
        - state: State code
        - source_file: Filename
        - title: Friendly title
        - pages: List of page dictionaries
        - total_pages: Total page count
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    documents = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    for pdf_path in pdf_files:
        try:
            # Extract metadata from filename
            filename = pdf_path.name
            state = extract_state_from_filename(filename)
            title = extract_title_from_filename(filename)
            
            # Extract pages
            pages = extract_pdf_pages(pdf_path)
            
            documents.append({
                "state": state,
                "source_file": filename,
                "title": title,
                "pages": pages,
                "total_pages": len(pages),
                "pdf_path": str(pdf_path)
            })
            
            print(f"✓ Extracted {len(pages)} pages from {filename} (State: {state})")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {str(e)}")
            continue
    
    return documents

