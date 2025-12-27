"""
Streamlit frontend for State DOT Maintenance Manual RAG application.

Usage:
    streamlit run app.py
"""
import streamlit as st
from pathlib import Path
import sys

from config import (
    SUPPORTED_STATES,
    STATE_NAMES,
    MIN_TOP_K,
    MAX_TOP_K,
    DEFAULT_TOP_K,
    GROQ_API_KEY,
    CHROMA_DIR,
    COLLECTION_NAME
)
from rag import RAGPipeline, get_collection_stats


# Page configuration
st.set_page_config(
    page_title="State DOT Maintenance Manual RAG",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_prerequisites():
    """Check if all prerequisites are met before running the app."""
    errors = []
    
    # Check GROQ_API_KEY
    if not GROQ_API_KEY:
        errors.append("❌ GROQ_API_KEY not set. Please set it in your .env file or environment.")
    
    # Check if ChromaDB collection exists
    if not CHROMA_DIR.exists():
        errors.append(f"❌ ChromaDB directory not found: {CHROMA_DIR}")
        errors.append("   Please run: python ingest.py")
    else:
        # Check collection stats
        stats = get_collection_stats()
        if 'error' in stats:
            errors.append(f"❌ ChromaDB collection '{COLLECTION_NAME}' not found.")
            errors.append("   Please run: python ingest.py")
        elif stats.get('total_chunks', 0) == 0:
            errors.append(f"❌ ChromaDB collection is empty.")
            errors.append("   Please run: python ingest.py")
    
    return errors


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = "CA"


def load_rag_pipeline():
    """Load RAG pipeline (cached in session state)."""
    if st.session_state.rag_pipeline is None:
        with st.spinner("Loading RAG pipeline..."):
            try:
                st.session_state.rag_pipeline = RAGPipeline()
                return True
            except Exception as e:
                st.error(f"Error loading RAG pipeline: {str(e)}")
                return False
    return True


def display_citation(citation, index):
    """Display a single citation in an expander."""
    source = citation['source_file']
    page_start = citation['page_start']
    page_end = citation['page_end']
    snippet = citation['snippet']
    has_time = citation.get('has_time_keywords', False)
    keywords = citation.get('matched_keywords', [])
    
    # Format page reference
    if page_start == page_end:
        page_ref = f"p.{page_start}"
    else:
        page_ref = f"p.{page_start}-{page_end}"
    
    # Create expander title
    title = f"📄 {source} {page_ref}"
    if has_time:
        title += " ⏰"
    
    with st.expander(title):
        st.markdown(f"**Source:** {source}")
        st.markdown(f"**Pages:** {page_ref}")
        
        if has_time and keywords:
            if isinstance(keywords, str):
                keywords = keywords.split(',')
            st.markdown(f"**Time Keywords Found:** {', '.join(keywords)}")
        
        st.markdown("**Excerpt:**")
        st.text(snippet)


def main():
    """Main application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Maintenance Manual RAG")
    
    
    # Check prerequisites
    errors = check_prerequisites()
    if errors:
        st.error("**Prerequisites not met:**")
        for error in errors:
            st.text(error)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # State selection
        selected_state = st.selectbox(
            "Select State",
            options=SUPPORTED_STATES,
            format_func=lambda x: f"{x} - {STATE_NAMES[x]}",
            key="state_selector"
        )
        
        # Update session state
        if selected_state != st.session_state.selected_state:
            st.session_state.selected_state = selected_state
            st.session_state.messages = []  # Clear chat history on state change
        
        st.divider()
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Top K Results",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            help="Number of document chunks to retrieve"
        )
        
        show_debug = st.checkbox(
            "Debug: Show retrieved chunks",
            value=False,
            help="Display raw retrieved chunks for debugging"
        )
        
        st.divider()
        
        # Collection info
        stats = get_collection_stats()
        if 'total_chunks' in stats:
            st.metric("📊 Total Chunks", stats['total_chunks'])
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Load RAG pipeline
    if not load_rag_pipeline():
        st.stop()
    
    # Main chat interface
    st.subheader(f"💬 {STATE_NAMES[selected_state]} ({selected_state})")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations and debug info only if debug mode is enabled
            if message["role"] == "assistant" and show_debug:
                # Display citations
                if "citations" in message and message["citations"]:
                    st.markdown("---")
                    st.markdown("**📚 Citations:**")
                    for i, citation in enumerate(message["citations"]):
                        display_citation(citation, i)
                
                # Display debug info
                if "debug_chunks" in message:
                    with st.expander("🔧 Debug: Retrieved Chunks"):
                        for i, chunk in enumerate(message["debug_chunks"]):
                            st.markdown(f"**Chunk {i+1}** (distance: {chunk.get('distance', 'N/A'):.4f})")
                            st.text(chunk['text'][:300] + "...")
                            st.json(chunk['metadata'])
    
    # Helper function to process questions
    def process_question(question_text):
        """Process a user question and generate response."""
        try:
            # Query RAG pipeline
            response = st.session_state.rag_pipeline.answer_question(
                state=selected_state,
                question=question_text,
                k=top_k,
                return_debug=show_debug
            )
            
            # Extract data
            answer = response['final_answer']
            citations = response.get('citations', [])
            
            # Add assistant message to chat
            message_data = {
                "role": "assistant",
                "content": answer,
                "citations": citations
            }
            
            if show_debug and 'retrieved_chunks' in response:
                message_data['debug_chunks'] = response['retrieved_chunks']
            
            st.session_state.messages.append(message_data)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
    
    # Check if there's a pending question from suggested questions
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        last_message = st.session_state.messages[-1]["content"]
        # Check if this is a new question that hasn't been answered yet
        if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Searching manual and generating answer..."):
                    process_question(last_message)
                    st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about maintenance policies..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching manual and generating answer..."):
                process_question(prompt)
                st.rerun()
    
    # Suggested questions (only show if chat is empty)
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("**💡 Suggested Questions:**")
        
        suggestions = [
            "Are there any nighttime restrictions for maintenance work?",
            "What are the lane closure requirements?",
            "What time of day can maintenance be performed?",
            "What are the traffic control requirements?",
            "Are there any off-peak hour requirements?"
        ]
        
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(suggestion, use_container_width=True):
                    # Trigger question
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.rerun()


if __name__ == "__main__":
    main()

