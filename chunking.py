"""
Text chunking utilities with keyword tagging for maintenance manuals.
"""
from typing import List, Dict, Tuple
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP_CHARS, TIME_KEYWORDS


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size per chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # If this is not the last chunk and we can find a good break point
        if end < text_length:
            # Try to break at paragraph boundary (double newline)
            break_pos = text.rfind('\n\n', start, end)
            
            # If no paragraph break, try sentence boundary
            if break_pos == -1 or break_pos <= start:
                # Look for period followed by space and capital letter
                for i in range(end, start + chunk_size // 2, -1):
                    if i < text_length and text[i] == '.' and i + 1 < text_length:
                        if text[i + 1] in [' ', '\n']:
                            break_pos = i + 1
                            break
            
            # If no good break point found, just use chunk_size
            if break_pos == -1 or break_pos <= start:
                break_pos = end
            
            end = break_pos
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < text_length else text_length
        
        # Ensure we make progress
        if start <= (end - overlap):
            start = end
    
    return chunks


def detect_time_keywords(text: str) -> Tuple[bool, List[str]]:
    """
    Detect time-related keywords in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (has_keywords, matched_keywords)
    """
    text_lower = text.lower()
    matched = []
    
    for keyword in TIME_KEYWORDS:
        if keyword.lower() in text_lower:
            matched.append(keyword)
    
    has_keywords = len(matched) > 0
    return has_keywords, matched


def create_chunk_id(state: str, source_file: str, page_start: int, page_end: int, chunk_index: int) -> str:
    """
    Create a stable, unique ID for a chunk.
    
    Args:
        state: State code (CA, TX, WA)
        source_file: Source PDF filename
        page_start: Starting page number
        page_end: Ending page number
        chunk_index: Index of chunk within this page range
        
    Returns:
        Unique chunk ID string
    """
    return f"{state}:{source_file}:{page_start}-{page_end}:{chunk_index}"


def chunk_document_pages(
    pages: List[Dict[str, any]],
    state: str,
    source_file: str,
    title: str,
    doc_type: str = "maintenance_manual"
) -> List[Dict[str, any]]:
    """
    Chunk document pages into smaller pieces with metadata.
    Never crosses document boundaries.
    
    Args:
        pages: List of page dictionaries from pdf_extract
        state: State code
        source_file: Source filename
        title: Document title
        doc_type: Document type
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    chunk_global_index = 0
    
    # Combine consecutive pages into larger segments, then chunk
    # This allows chunks to span pages naturally
    current_text = ""
    current_page_start = None
    current_page_end = None
    
    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"]
        
        # Skip empty pages
        if not page_text.strip():
            continue
        
        # Initialize page range
        if current_page_start is None:
            current_page_start = page_num
        
        current_page_end = page_num
        current_text += "\n\n" + page_text if current_text else page_text
        
        # If accumulated text is large enough, chunk it
        if len(current_text) >= CHUNK_SIZE * 2:
            page_chunks = _chunk_and_create_metadata(
                current_text,
                state,
                source_file,
                title,
                doc_type,
                current_page_start,
                current_page_end,
                chunk_global_index
            )
            chunks.extend(page_chunks)
            chunk_global_index += len(page_chunks)
            
            # Reset accumulators
            current_text = ""
            current_page_start = None
            current_page_end = None
    
    # Process remaining text
    if current_text.strip():
        page_chunks = _chunk_and_create_metadata(
            current_text,
            state,
            source_file,
            title,
            doc_type,
            current_page_start,
            current_page_end,
            chunk_global_index
        )
        chunks.extend(page_chunks)
    
    return chunks


def _chunk_and_create_metadata(
    text: str,
    state: str,
    source_file: str,
    title: str,
    doc_type: str,
    page_start: int,
    page_end: int,
    start_index: int
) -> List[Dict[str, any]]:
    """
    Helper to chunk text and create metadata dictionaries.
    
    Args:
        text: Text to chunk
        state: State code
        source_file: Source filename
        title: Document title
        doc_type: Document type
        page_start: Starting page number
        page_end: Ending page number
        start_index: Starting index for chunk IDs
        
    Returns:
        List of chunk dictionaries
    """
    text_chunks = chunk_text(text)
    result = []
    
    for i, text_chunk in enumerate(text_chunks):
        # Detect time keywords
        has_time_keywords, matched_keywords = detect_time_keywords(text_chunk)
        
        # Create chunk ID
        chunk_id = create_chunk_id(state, source_file, page_start, page_end, start_index + i)
        
        # Build metadata
        chunk_dict = {
            "id": chunk_id,
            "text": text_chunk,
            "state": state,
            "doc_type": doc_type,
            "title": title,
            "source_file": source_file,
            "page_start": page_start,
            "page_end": page_end,
            "has_time_keywords": has_time_keywords,
            "matched_time_keywords": matched_keywords,
            "char_count": len(text_chunk)
        }
        
        result.append(chunk_dict)
    
    return result

