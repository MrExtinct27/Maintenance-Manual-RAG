"""
RAG pipeline using LangChain: retrieval, prompt assembly, and Groq LLM completion.
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    DEFAULT_TOP_K,
    LANGCHAIN_API_KEY,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_PROJECT
)


class RAGPipeline:
    """RAG pipeline for querying state maintenance manuals using LangChain."""
    
    def __init__(self):
        """Initialize the RAG pipeline with embedding model, Chroma client, and LangChain LLM."""
        # Initialize embedding model
        print(f"Loading embedding model: {EMBED_MODEL}")
        self.embedding_model = SentenceTransformer(EMBED_MODEL)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            print(f"✓ Connected to collection: {COLLECTION_NAME}")
        except Exception as e:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Please run 'python ingest.py' first to create the collection."
            )
        
        # Initialize LangChain LLM with Groq
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in your environment or .env file."
            )
        
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY
        )
        
        print(f"✓ Initialized LangChain with Groq model: {GROQ_MODEL}")
        
    def _embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(query).tolist()
    
    def _is_time_related_query(self, query: str) -> bool:
        """
        Check if query is asking about time-of-day constraints.
        
        Args:
            query: User query
            
        Returns:
            True if query appears to be about time constraints
        """
        query_lower = query.lower()
        time_indicators = [
            "night", "day", "time", "hours", "off-peak",
            "curfew", "lane closure", "when", "schedule"
        ]
        return any(indicator in query_lower for indicator in time_indicators)
    
    def retrieve_chunks(
        self,
        query: str,
        state: str,
        k: int = DEFAULT_TOP_K,
        boost_time_keywords: bool = True
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            state: State filter (CA, TX, or WA)
            k: Number of results to retrieve
            boost_time_keywords: Whether to boost chunks with time keywords
            
        Returns:
            List of retrieved chunk dictionaries with metadata
        """
        # Embed query
        query_embedding = self._embed_query(query)
        
        # Check if query is time-related
        is_time_query = self._is_time_related_query(query)
        
        # Base filter for state
        where_filter = {"state": state}
        
        # Strategy: If time-related query, get both time-keyword chunks and general chunks
        all_results = []
        
        if is_time_query and boost_time_keywords:
            # First, get chunks WITH time keywords
            try:
                time_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, 15),
                    where={**where_filter, "has_time_keywords": True}
                )
                
                # Parse results
                if time_results['ids'][0]:
                    for i in range(len(time_results['ids'][0])):
                        all_results.append({
                            'id': time_results['ids'][0][i],
                            'text': time_results['documents'][0][i],
                            'metadata': time_results['metadatas'][0][i],
                            'distance': time_results['distances'][0][i],
                            'boost_score': 0.1  # Boost for time keywords
                        })
            except Exception as e:
                # If no results with time keywords, that's okay
                pass
            
            # Then get general results
            general_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter
            )
            
            # Add general results (avoiding duplicates)
            existing_ids = {r['id'] for r in all_results}
            if general_results['ids'][0]:
                for i in range(len(general_results['ids'][0])):
                    chunk_id = general_results['ids'][0][i]
                    if chunk_id not in existing_ids:
                        all_results.append({
                            'id': chunk_id,
                            'text': general_results['documents'][0][i],
                            'metadata': general_results['metadatas'][0][i],
                            'distance': general_results['distances'][0][i],
                            'boost_score': 0.0
                        })
            
            # Sort by adjusted distance (lower is better)
            all_results.sort(key=lambda x: x['distance'] - x['boost_score'])
            all_results = all_results[:k]
            
        else:
            # Simple retrieval without boosting
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter
            )
            
            # Parse results
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'boost_score': 0.0
                    })
        
        return all_results
    
    def _format_context(self, chunks: List[Dict[str, any]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            text = chunk['text']
            page_start = metadata.get('page_start', '?')
            page_end = metadata.get('page_end', '?')
            source = metadata.get('source_file', 'Unknown')
            
            # Format page range
            if page_start == page_end:
                page_ref = f"p.{page_start}"
            else:
                page_ref = f"p.{page_start}-{page_end}"
            
            context_parts.append(
                f"[Excerpt {i}] ({source} {page_ref}):\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _call_llm(self, query: str, context: str, state: str) -> str:
        """
        Call LangChain LLM for completion.
        
        Args:
            query: User query
            context: Formatted context from retrieved chunks
            state: State code
            
        Returns:
            LLM response text
        """
        try:
            # Create prompt template
            system_message = SystemMessage(
                content="You are a helpful assistant specialized in analyzing transportation maintenance manuals. You provide accurate, cited answers based only on provided context."
            )
            
            # Build the user message
            user_prompt = f"""You are an expert assistant helping users understand state Department of Transportation (DOT) maintenance manual policies.

You will be provided with excerpts from the {state} maintenance manual and a user question. Your task is to answer the question based ONLY on the provided excerpts.

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided excerpts below
2. If the question asks about time-of-day constraints (nighttime, daytime, off-peak, work hours, lane closure windows, etc.), you MUST explicitly state whether such requirements exist in the provided excerpts
3. If no explicit time-of-day requirement is found in the excerpts, you MUST say: "No explicit time-of-day requirement found in the provided manual excerpts."
4. DO NOT make up or infer information not present in the excerpts
5. Include citations in the format: (source_file p.X) or (source_file p.X-Y)
6. Keep quotes SHORT - only essential snippets
7. Be concise but complete

CONTEXT FROM {state} MAINTENANCE MANUAL:
{context}

USER QUESTION:
{query}

Please provide a clear, accurate answer with citations:"""
            
            human_message = HumanMessage(content=user_prompt)
            
            # Call LLM using LangChain
            response = self.llm.invoke([system_message, human_message])
            
            return response.content
            
        except Exception as e:
            raise Exception(f"Error calling LangChain LLM: {str(e)}")
    
    def _extract_citations(self, answer: str, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Extract citation information from retrieved chunks.
        
        Args:
            answer: LLM answer text
            chunks: Retrieved chunks
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for chunk in chunks:
            metadata = chunk['metadata']
            text = chunk['text']
            
            # Extract a snippet (first 200 chars)
            snippet = text[:200] + "..." if len(text) > 200 else text
            
            citations.append({
                'source_file': metadata.get('source_file', 'Unknown'),
                'page_start': metadata.get('page_start', '?'),
                'page_end': metadata.get('page_end', '?'),
                'snippet': snippet,
                'has_time_keywords': metadata.get('has_time_keywords', False),
                'matched_keywords': metadata.get('matched_time_keywords', [])
            })
        
        return citations
    
    def answer_question(
        self,
        state: str,
        question: str,
        k: int = DEFAULT_TOP_K,
        return_debug: bool = False
    ) -> Dict[str, any]:
        """
        Answer a question about a state's maintenance manual using LangChain.
        
        Args:
            state: State code (CA, TX, or WA)
            question: User question
            k: Number of chunks to retrieve
            return_debug: Whether to return debug information
            
        Returns:
            Dictionary containing:
            - final_answer: The LLM's answer
            - citations: List of citation dictionaries
            - retrieved_chunks: (optional) Retrieved chunks for debugging
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_chunks(query=question, state=state, k=k)
        
        if not chunks:
            return {
                'final_answer': f"No relevant information found in the {state} maintenance manual for this query.",
                'citations': [],
                'retrieved_chunks': [] if return_debug else None
            }
        
        # Format context
        context = self._format_context(chunks)
        
        # Call LLM via LangChain
        answer = self._call_llm(question, context, state)
        
        # Extract citations
        citations = self._extract_citations(answer, chunks)
        
        # Build response
        response = {
            'final_answer': answer,
            'citations': citations
        }
        
        if return_debug:
            response['retrieved_chunks'] = chunks
        
        return response


def get_collection_stats() -> Dict[str, any]:
    """
    Get statistics about the ChromaDB collection.
    
    Returns:
        Dictionary with collection statistics
    """
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
        # Get count
        count = collection.count()
        
        # Try to get state breakdown
        stats = {
            'total_chunks': count,
            'collection_name': COLLECTION_NAME
        }
        
        # Get counts per state
        for state in ['CA', 'TX', 'WA']:
            try:
                results = collection.get(where={"state": state}, limit=1)
                # This is a rough check - actual count would require querying all
                stats[f'{state}_exists'] = len(results['ids']) > 0
            except:
                stats[f'{state}_exists'] = False
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}
