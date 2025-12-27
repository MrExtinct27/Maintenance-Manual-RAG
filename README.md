# State DOT Maintenance Manual RAG Application

A Phase-1 prototype RAG (Retrieval-Augmented Generation) application for querying state maintenance manuals. Currently supports California (CA), Texas (TX), and Washington (WA).

## 🎯 Purpose

This tool helps users query state Department of Transportation (DOT) maintenance manuals to find information about:
- Time-of-day constraints for maintenance work (nighttime/daytime/off-peak hours)
- Lane closure requirements and windows
- Traffic control policies
- Safety requirements
- Other maintenance activity guidance

The application provides clear answers with evidence snippets and page number citations from the official manuals.

## 🏗️ Architecture

**Tech Stack:**
- **PDF Extraction:** PyMuPDF (fitz)
- **Embeddings:** HuggingFace sentence-transformers (BAAI/bge-base-en-v1.5)
- **Vector Database:** ChromaDB (persistent storage)
- **LLM:** Groq API for response synthesis
- **Frontend:** Streamlit

**Key Features:**
- Smart chunking with time-keyword tagging for better retrieval
- State-filtered queries (only search selected state's manual)
- Citation tracking with page numbers
- No hallucination: explicitly states when information is not found
- Debug mode to inspect retrieved chunks

## 📋 Prerequisites

- Python 3.10 or higher
- Groq API key ([Get one here](https://console.groq.com))
- State maintenance manual PDFs (see below)

## 🚀 Setup

### 1. Clone or Download

```bash
cd /path/to/RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (these are defaults)
GROQ_MODEL=llama-3.1-70b-versatile
EMBED_MODEL=BAAI/bge-base-en-v1.5
```

**Environment Variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | - | Your Groq API key |
| `GROQ_MODEL` | ❌ No | `llama-3.1-70b-versatile` | Groq model to use |
| `EMBED_MODEL` | ❌ No | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |

### 5. Add PDF Documents

Place your state maintenance manual PDFs in `data/pdfs/` directory:

```
data/pdfs/
├── CA_Caltrans_Maintenance_Manual.pdf
├── TX_TxDOT_Maintenance_Operations_Manual.pdf
└── WA_WSDOT_Maintenance_Manual.pdf
```

**Important:** The filename must start with the state code (`CA_`, `TX_`, or `WA_`) followed by an underscore.

The `data/pdfs/` directory will be created automatically when you run the application for the first time.

## 📥 Ingestion

Before using the application, you need to ingest the PDF documents:

```bash
python ingest.py
```

This will:
1. Extract text from all PDFs page-by-page
2. Chunk documents (~1200 tokens per chunk with 12.5% overlap)
3. Tag chunks containing time-related keywords
4. Generate embeddings using sentence-transformers
5. Store everything in ChromaDB at `data/chroma/`

**Example output:**
```
======================================================================
MAINTENANCE MANUAL INGESTION PIPELINE
======================================================================

📂 PDF Directory: /path/to/data/pdfs
📦 ChromaDB Path: /path/to/data/chroma
🔤 Embedding Model: BAAI/bge-base-en-v1.5
📚 Collection Name: road_maintenance_manuals

STEP 1: Extracting PDFs
----------------------------------------------------------------------
✓ Extracted CA_Caltrans_Maintenance_Manual.pdf (State: CA)
✓ Extracted TX_TxDOT_Maintenance_Operations_Manual.pdf (State: TX)
✓ Extracted WA_WSDOT_Maintenance_Manual.pdf (State: WA)

✓ Extracted 3 document(s)

...

✅ INGESTION COMPLETE
```

## 🖥️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 Usage

1. **Select a state** from the dropdown (CA, TX, or WA)
2. **Ask questions** in the chat interface, such as:
   - "Are there any nighttime restrictions for maintenance work?"
   - "What are the lane closure requirements?"
   - "What time of day can maintenance be performed?"
   - "Are there any off-peak hour requirements?"

3. **Review the answer** with citations showing:
   - Source file name
   - Page numbers
   - Relevant excerpts from the manual

4. **Adjust settings** in the sidebar:
   - **Top K Results:** Number of chunks to retrieve (5-20)
   - **Debug Mode:** Show raw retrieved chunks for troubleshooting

## 📁 Project Structure

```
RAG/
├── app.py                  # Streamlit frontend
├── ingest.py              # CLI ingestion script
├── rag.py                 # Retrieval + LLM logic
├── pdf_extract.py         # PyMuPDF extraction helpers
├── chunking.py            # Chunking + keyword tagging
├── config.py              # Configuration & env vars
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
└── data/
    ├── pdfs/             # Place PDF files here
    └── chroma/           # ChromaDB storage (auto-created)
```

## 🔍 How It Works

### Ingestion Pipeline

1. **PDF Extraction:** Uses PyMuPDF to extract text page-by-page
2. **Text Normalization:** Cleans whitespace while keeping text readable
3. **Chunking:** Splits into ~1200 token chunks with 12.5% overlap
4. **Keyword Tagging:** Identifies chunks containing time-related terms:
   - night, nighttime, daytime, off-peak, peak, curfew
   - hours of work, work hours, lane closure, closure window
5. **Embedding:** Creates vector embeddings using sentence-transformers
6. **Storage:** Stores in ChromaDB with rich metadata

### Query Pipeline

1. **State Filtering:** Only searches the selected state's manual
2. **Semantic Search:** Embeds query and finds similar chunks
3. **Smart Retrieval:** For time-related queries, prioritizes chunks with time keywords
4. **Prompt Assembly:** Builds context-rich prompt with retrieved chunks
5. **LLM Synthesis:** Groq API generates answer with strict instructions:
   - Only use provided context
   - Explicitly state if time-of-day rules exist
   - Include citations
   - Never hallucinate

## ⚠️ Important Notes

- **Verification Required:** This tool is for document lookup only. Always verify critical information with the official manual.
- **No Hallucination Policy:** If information isn't found in the manual, the system will explicitly say so.
- **State-Specific:** Each query only searches the selected state's manual.
- **Phase 1 Scope:** Currently limited to 3 states (CA, TX, WA) and maintenance manuals only.

## 🐛 Troubleshooting

### "ChromaDB collection not found"
- Run `python ingest.py` to create and populate the collection

### "GROQ_API_KEY not set"
- Create a `.env` file with your Groq API key
- Or set the environment variable: `export GROQ_API_KEY=your_key`

### "No PDF files found"
- Ensure PDFs are placed in `data/pdfs/` directory
- Check that filenames start with state codes (CA_, TX_, WA_)

### Empty or poor results
- Try increasing "Top K Results" in the sidebar
- Check if your query is clearly worded
- Enable debug mode to see what chunks were retrieved

## 🔧 Advanced Configuration

### Custom Chunking

Edit `config.py`:
```python
CHUNK_SIZE = 5000  # characters (~1200 tokens)
CHUNK_OVERLAP = 0.125  # 12.5% overlap
```

### Custom Time Keywords

Edit `config.py`:
```python
TIME_KEYWORDS = [
    "night", "nighttime", "daytime",
    # Add your keywords here
]
```

### Different Embedding Model

Set in `.env`:
```
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 📝 Example Queries

**Time-of-Day Constraints:**
- "When can roadwork be performed?"
- "Are there nighttime maintenance restrictions?"
- "What are the allowed hours for lane closures?"

**Lane Closures:**
- "What are the lane closure requirements?"
- "How many lanes can be closed at once?"
- "What traffic control is needed for closures?"

**General Maintenance:**
- "What safety equipment is required?"
- "What are the traffic control requirements?"
- "How should work zones be set up?"

## 📄 License

This is a prototype application. Please ensure you have the right to use and distribute the state maintenance manual PDFs.

## 🤝 Contributing

This is a Phase-1 prototype. Future enhancements could include:
- More states
- Additional document types (design manuals, construction specs)
- Multi-state comparison queries
- Export functionality for citations
- Advanced filtering (by topic, section, etc.)

---

**Built with:** PyMuPDF • HuggingFace • ChromaDB • Groq • Streamlit

