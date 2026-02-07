"""
Comprehensive system diagnostic test
"""
import sys

print("=" * 60)
print("SYSTEM DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Check Python version
print("\n[1/6] Python Version:")
print(f"  ✓ {sys.version}")

# Test 2: Check main imports
print("\n[2/6] Main Imports:")
try:
    import streamlit
    print(f"  ✓ Streamlit: {streamlit.__version__}")
except Exception as e:
    print(f"  ✗ Streamlit: {e}")

try:
    import llama_index
    print(f"  ✓ LlamaIndex imported")
except Exception as e:
    print(f"  ✗ LlamaIndex: {e}")

try:
    from llama_index.llms.groq import Groq
    print(f"  ✓ Groq LLM imported")
except Exception as e:
    print(f"  ✗ Groq LLM: {e}")

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print(f"  ✓ HuggingFace Embeddings imported")
except Exception as e:
    print(f"  ✗ HuggingFace: {e}")

try:
    import fitz  # PyMuPDF
    print(f"  ✓ PyMuPDF imported")
except Exception as e:
    print(f"  ✗ PyMuPDF: {e}")

try:
    import faiss
    print(f"  ✓ FAISS imported")
except Exception as e:
    print(f"  ✗ FAISS: {e}")

# Test 3: Check local modules
print("\n[3/6] Local Modules:")
try:
    from pdf_utils import PDFParser
    print(f"  ✓ PDFParser imported")
except Exception as e:
    print(f"  ✗ PDFParser: {e}")

try:
    from rag_engine import ResearchPaperRAG
    print(f"  ✓ ResearchPaperRAG imported")
except Exception as e:
    print(f"  ✗ ResearchPaperRAG: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check HuggingFace embeddings dimension
print("\n[4/6] HuggingFace Embeddings Setup:")
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    
    # Get dimension by embedding a test string
    test_embedding = embed_model.get_text_embedding("test")
    dimension = len(test_embedding)
    print(f"  ✓ HuggingFace model loaded: dimension = {dimension}")
except Exception as e:
    print(f"  ✗ HuggingFace setup failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test Groq initialization (without API call)
print("\n[5/6] Groq LLM Initialization:")
try:
    from llama_index.llms.groq import Groq as GroqLLM
    from llama_index.core import Settings
    
    # Just test object creation, not API call
    llm = GroqLLM(api_key="test_key", model="mixtral-8x7b-32768")
    print(f"  ✓ Groq LLM object created successfully")
except Exception as e:
    print(f"  ✗ Groq LLM creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test RAG engine initialization structure
print("\n[6/6] RAG Engine Structure:")
try:
    from rag_engine import ResearchPaperRAG
    # Just check the class exists and has the right methods
    methods = ['add_paper', 'query_paper', 'get_paper_summary', 'compare_papers']
    for method in methods:
        if hasattr(ResearchPaperRAG, method):
            print(f"  ✓ Method '{method}' exists")
        else:
            print(f"  ✗ Method '{method}' missing")
except Exception as e:
    print(f"  ✗ RAG Engine check failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
