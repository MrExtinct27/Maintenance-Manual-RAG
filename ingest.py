#!/usr/bin/env python3
"""
Document ingestion script for maintenance manuals.
Extracts, chunks, embeds, and stores documents in ChromaDB.

Usage:
    python ingest.py
"""
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    PDF_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    DOC_TYPE,
    SUPPORTED_STATES
)
from pdf_extract import extract_all_pdfs
from chunking import chunk_document_pages


def main():
    """Main ingestion pipeline."""
    print("=" * 70)
    print("MAINTENANCE MANUAL INGESTION PIPELINE")
    print("=" * 70)
    print()
    
    # Check if PDF directory exists and has files
    if not PDF_DIR.exists():
        print(f"❌ Error: PDF directory not found: {PDF_DIR}")
        print(f"Please create the directory and add PDF files.")
        sys.exit(1)
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {PDF_DIR}")
        print(f"\nExpected files:")
        print(f"  - CA_Caltrans_Maintenance_Manual.pdf")
        print(f"  - TX_TxDOT_Maintenance_Management_Manual.pdf")
        print(f"  - WA_WSDOT_Maintenance_Manual.pdf")
        sys.exit(1)
    
    print(f"📂 PDF Directory: {PDF_DIR}")
    print(f"📦 ChromaDB Path: {CHROMA_DIR}")
    print(f"🔤 Embedding Model: {EMBED_MODEL}")
    print(f"📚 Collection Name: {COLLECTION_NAME}")
    print()
    
    # Step 1: Extract PDFs
    print("STEP 1: Extracting PDFs")
    print("-" * 70)
    try:
        documents = extract_all_pdfs(PDF_DIR)
        print(f"\n✓ Extracted {len(documents)} document(s)")
        
        # Show summary
        for doc in documents:
            print(f"  - {doc['state']}: {doc['title']} ({doc['total_pages']} pages)")
    except Exception as e:
        print(f"\n❌ Error during extraction: {str(e)}")
        sys.exit(1)
    
    print()
    
    # Step 2: Chunk documents
    print("STEP 2: Chunking documents")
    print("-" * 70)
    all_chunks = []
    
    for doc in documents:
        print(f"Processing {doc['state']}: {doc['source_file']}")
        try:
            chunks = chunk_document_pages(
                pages=doc['pages'],
                state=doc['state'],
                source_file=doc['source_file'],
                title=doc['title'],
                doc_type=DOC_TYPE
            )
            all_chunks.extend(chunks)
            
            # Count chunks with time keywords
            time_chunks = sum(1 for c in chunks if c['has_time_keywords'])
            print(f"  ✓ Created {len(chunks)} chunks ({time_chunks} with time keywords)")
            
        except Exception as e:
            print(f"  ❌ Error chunking document: {str(e)}")
            continue
    
    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    print()
    
    if not all_chunks:
        print("❌ No chunks created. Exiting.")
        sys.exit(1)
    
    # Step 3: Initialize embedding model
    print("STEP 3: Loading embedding model")
    print("-" * 70)
    try:
        embedding_model = SentenceTransformer(EMBED_MODEL)
        print(f"✓ Loaded model: {EMBED_MODEL}")
    except Exception as e:
        print(f"❌ Error loading embedding model: {str(e)}")
        sys.exit(1)
    
    print()
    
    # Step 4: Create embeddings
    print("STEP 4: Creating embeddings")
    print("-" * 70)
    try:
        texts = [chunk['text'] for chunk in all_chunks]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        print(f"✓ Created {len(embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Error creating embeddings: {str(e)}")
        sys.exit(1)
    
    print()
    
    # Step 5: Store in ChromaDB
    print("STEP 5: Storing in ChromaDB")
    print("-" * 70)
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"🗑️  Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "State DOT maintenance manuals"}
        )
        print(f"✓ Created collection: {COLLECTION_NAME}")
        
        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for chunk, embedding in zip(all_chunks, embeddings):
            ids.append(chunk['id'])
            documents.append(chunk['text'])
            
            # Prepare metadata (ChromaDB requires simple types)
            metadata = {
                'state': chunk['state'],
                'doc_type': chunk['doc_type'],
                'title': chunk['title'],
                'source_file': chunk['source_file'],
                'page_start': int(chunk['page_start']),
                'page_end': int(chunk['page_end']),
                'has_time_keywords': bool(chunk['has_time_keywords']),
                'matched_time_keywords': ','.join(chunk['matched_time_keywords']) if chunk['matched_time_keywords'] else '',
                'char_count': int(chunk['char_count'])
            }
            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())
        
        # Insert in batches
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="Inserting batches"):
            batch_end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                embeddings=embeddings_list[i:batch_end]
            )
        
        print(f"✓ Stored {len(ids)} chunks in ChromaDB")
        
        # Verify counts per state
        print("\nState breakdown:")
        for state in SUPPORTED_STATES:
            state_chunks = [c for c in all_chunks if c['state'] == state]
            time_keyword_chunks = [c for c in state_chunks if c['has_time_keywords']]
            print(f"  {state}: {len(state_chunks)} chunks ({len(time_keyword_chunks)} with time keywords)")
        
    except Exception as e:
        print(f"❌ Error storing in ChromaDB: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("✅ INGESTION COMPLETE")
    print("=" * 70)
    print(f"\nYou can now run the application with:")
    print(f"  streamlit run app.py")
    print()


if __name__ == "__main__":
    main()

