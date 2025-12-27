#!/usr/bin/env python3
"""
Setup verification script.
Run this to check if your environment is properly configured.
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro}")
        print("  Required: Python 3.10 or higher")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        "streamlit",
        "fitz",  # PyMuPDF
        "sentence_transformers",
        "chromadb",
        "groq",
        "dotenv",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ Package installed: {package}")
        except ImportError:
            print(f"✗ Package missing: {package}")
            missing.append(package)
    
    if missing:
        print(f"\nRun: pip install -r requirements.txt")
        return False
    return True


def check_env_file():
    """Check if .env file exists and has GROQ_API_KEY."""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("✗ .env file not found")
        print("  Copy env.example to .env and add your GROQ_API_KEY")
        return False
    
    print("✓ .env file exists")
    
    # Check for GROQ_API_KEY
    with open(env_path) as f:
        content = f.read()
        if "GROQ_API_KEY" in content and "your_groq_api_key_here" not in content:
            # Check if it's not empty
            for line in content.split('\n'):
                if line.startswith('GROQ_API_KEY=') and len(line.split('=')[1].strip()) > 0:
                    print("✓ GROQ_API_KEY is set")
                    return True
    
    print("✗ GROQ_API_KEY not set in .env")
    return False


def check_data_directory():
    """Check data directory structure."""
    pdf_dir = Path("data/pdfs")
    
    if not pdf_dir.exists():
        print("✗ data/pdfs/ directory not found")
        return False
    
    print("✓ data/pdfs/ directory exists")
    
    # Check for PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠ No PDF files found in data/pdfs/")
        print("  Expected files:")
        print("    - CA_Caltrans_Maintenance_Manual.pdf")
        print("    - TX_TxDOT_Maintenance_Operations_Manual.pdf")
        print("    - WA_WSDOT_Maintenance_Manual.pdf")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"    - {pdf.name}")
    
    return True


def check_chroma_collection():
    """Check if ChromaDB collection exists."""
    chroma_dir = Path("data/chroma")
    
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        print("⚠ ChromaDB not initialized")
        print("  Run: python ingest.py")
        return False
    
    print("✓ ChromaDB directory exists")
    
    # Try to check collection
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            collection = client.get_collection(name="road_maintenance_manuals")
            count = collection.count()
            print(f"✓ ChromaDB collection exists with {count} chunks")
            return True
        except:
            print("⚠ ChromaDB collection not found")
            print("  Run: python ingest.py")
            return False
            
    except ImportError:
        print("⚠ Cannot verify ChromaDB (chromadb not installed)")
        return False


def main():
    """Run all checks."""
    print("=" * 70)
    print("SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Data Directory", check_data_directory),
        ("ChromaDB Collection", check_chroma_collection),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 70)
        results.append(check_func())
    
    print()
    print("=" * 70)
    
    if all(results[:4]):  # Python, deps, env, data
        if results[4]:  # ChromaDB
            print("✅ ALL CHECKS PASSED - Ready to run!")
            print("\nStart the app with:")
            print("  streamlit run app.py")
        else:
            print("⚠ SETUP COMPLETE - Need to run ingestion")
            print("\nNext step:")
            print("  python ingest.py")
            print("\nThen start the app with:")
            print("  streamlit run app.py")
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nPlease fix the issues above and run this script again.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

