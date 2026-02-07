"""
RAG Engine using LlamaIndex and FAISS for research paper question answering.
Supports multi-paper queries, comparisons, and structured summaries.
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from typing import List, Dict, Optional
from pathlib import Path
import pickle

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core.node_parser import SimpleNodeParser
import faiss
import numpy as np


class ResearchPaperRAG:
    """RAG system for research paper question answering and summarization."""
    
    def __init__(self, provider: str = "groq", api_key: str = None, index_dir: str = "./faiss_index", model: str = None):
        """
        Initialize RAG engine.
        
        Args:
            provider: LLM provider - "groq" (free) or "openai" (paid)
            api_key: API key for the provider (Groq API key or OpenAI API key)
            index_dir: Directory to store FAISS indices
            model: Model name (optional). For Groq, uses mixtral-8x7b-32768 by default
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.provider = provider.lower()
        
        # Set default models if not specified
        if model is None:
            model = "mixtral-8x7b-32768" if provider.lower() == "groq" else "gpt-4o-mini"
        
        # Initialize LLM based on provider
        if self.provider == "groq":
            if not api_key:
                # Try to get from environment
                api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                Settings.llm = GroqLLM(
                    api_key=api_key,
                    model=model,  # Use provided model or default
                    temperature=0.1
                )
            else:
                raise ValueError("Groq API key required. Get free key from https://console.groq.com/keys")
        else:  # OpenAI
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                Settings.llm = OpenAI(
                    api_key=api_key,
                    model=model,  # Use provided model or default
                    temperature=0.1
                )
            else:
                raise ValueError("OpenAI API key required")
        
        # Use free Hugging Face embeddings (no API key needed)
        # Initialize lazily on first use to avoid blocking
        self._embed_model_initialized = False
        self._embed_model = None
        
        # Node parser for chunking documents
        Settings.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1024,
            chunk_overlap=200
        )
        
        # Store indices per paper
        self.paper_indices: Dict[str, VectorStoreIndex] = {}
        self.paper_docs: Dict[str, List[Document]] = {}
    
    def _initialize_embeddings(self):
        """Initialize embeddings lazily on first use."""
        if self._embed_model_initialized:
            return
        
        try:
            print("Initializing HuggingFace embeddings (this may take a moment)...")
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"  # Free, no API key needed
            )
            Settings.embed_model = embed_model
            self._embed_model = embed_model
            self._embed_model_initialized = True
            print("✓ Embeddings initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing embeddings: {e}")
            raise
        
    def add_paper(self, paper_name: str, text: str, metadata: Optional[Dict] = None):
        """
        Add a research paper to the RAG system.
        
        Args:
            paper_name: Name/identifier for the paper
            text: Full text of the paper
            metadata: Optional metadata (title, authors, etc.)
        """
        # Create document with metadata
        doc_metadata = metadata or {}
        doc_metadata['paper_name'] = paper_name
        
        document = Document(
            text=text,
            metadata=doc_metadata
        )
        
        # Store document
        if paper_name not in self.paper_docs:
            self.paper_docs[paper_name] = []
        self.paper_docs[paper_name].append(document)
        
        # Create or update index for this paper
        self._create_index_for_paper(paper_name)
    
    def _create_index_for_paper(self, paper_name: str):
        """Create or update FAISS index for a specific paper."""
        # Initialize embeddings on first use
        self._initialize_embeddings()
        
        documents = self.paper_docs[paper_name]
        
        # Create FAISS vector store with correct dimension for HuggingFace embeddings
        # BAAI/bge-small-en-v1.5 has dimension 384
        dimension = 384  # HuggingFace BAAI/bge-small-en-v1.5 dimension
        faiss_index = faiss.IndexFlatL2(dimension)
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        try:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=False
            )
            self.paper_indices[paper_name] = index
            
            # Save index
            self._save_index(paper_name, index, storage_context)
            print(f"✓ Successfully created index for {paper_name}")
        except Exception as e:
            print(f"✗ Error creating index for {paper_name}: {str(e)}")
            raise
    
    def _save_index(self, paper_name: str, index: VectorStoreIndex, storage_context: StorageContext):
        """Save FAISS index to disk."""
        paper_dir = self.index_dir / paper_name
        paper_dir.mkdir(exist_ok=True)
        
        storage_context.persist(persist_dir=str(paper_dir))
    
    def query_paper(self, paper_name: str, question: str, top_k: int = 5) -> str:
        """
        Query a specific paper.
        
        Args:
            paper_name: Name of the paper to query
            question: Question to ask
            top_k: Number of top chunks to retrieve
            
        Returns:
            Answer string
        """
        if paper_name not in self.paper_indices:
            return f"Paper '{paper_name}' not found in the system."
        
        index = self.paper_indices[paper_name]
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        
        response = query_engine.query(question)
        return str(response)
    
    def query_all_papers(self, question: str, top_k: int = 5) -> Dict[str, str]:
        """
        Query all papers and return answers for each.
        
        Args:
            question: Question to ask
            top_k: Number of top chunks to retrieve per paper
            
        Returns:
            Dictionary mapping paper names to answers
        """
        results = {}
        for paper_name in self.paper_indices.keys():
            results[paper_name] = self.query_paper(paper_name, question, top_k)
        return results
    
    def compare_papers(self, question: str, paper_names: Optional[List[str]] = None, top_k: int = 5) -> str:
        """
        Compare multiple papers on a specific question.
        
        Args:
            question: Comparison question
            paper_names: List of papers to compare (None = all papers)
            top_k: Number of top chunks to retrieve per paper
            
        Returns:
            Comparative answer
        """
        if paper_names is None:
            paper_names = list(self.paper_indices.keys())
        
        if len(paper_names) < 2:
            return "Need at least 2 papers for comparison."
        
        # Get answers from each paper
        individual_answers = {}
        for paper_name in paper_names:
            if paper_name in self.paper_indices:
                individual_answers[paper_name] = self.query_paper(paper_name, question, top_k)
        
        # Create comparison prompt
        comparison_text = f"Question: {question}\n\n"
        for paper_name, answer in individual_answers.items():
            comparison_text += f"Paper: {paper_name}\nAnswer: {answer}\n\n"
        
        comparison_text += "Please provide a comprehensive comparison synthesizing the above answers."
        
        # Use LLM to synthesize comparison
        response = Settings.llm.complete(comparison_text)
        return str(response)
    
    def get_paper_summary(self, paper_name: str) -> Dict[str, str]:
        """
        Generate structured summary for a paper.
        
        Returns:
            Dictionary with Problem, Method, Results, Conclusion
        """
        if paper_name not in self.paper_indices:
            return {
                'Problem': 'Paper not found',
                'Method': '',
                'Results': '',
                'Conclusion': ''
            }
        
        # Query different aspects
        problem = self.query_paper(paper_name, "What problem or research question does this paper address?")
        method = self.query_paper(paper_name, "What methodology or approach does this paper use?")
        results = self.query_paper(paper_name, "What are the main results or findings of this paper?")
        conclusion = self.query_paper(paper_name, "What are the conclusions and limitations of this paper?")
        
        return {
            'Problem': problem,
            'Method': method,
            'Results': results,
            'Conclusion': conclusion
        }
    
    def get_all_papers(self) -> List[str]:
        """Get list of all paper names in the system."""
        return list(self.paper_indices.keys())
    
    def clear_all(self):
        """Clear all papers and indices."""
        self.paper_indices.clear()
        self.paper_docs.clear()
        # Optionally delete index directory
        import shutil
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(exist_ok=True)
