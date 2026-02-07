# 📚 Research Paper Summarizer

A hackathon-ready RAG (Retrieval-Augmented Generation) system for analyzing and comparing academic research papers. Built with Streamlit, LlamaIndex, FAISS, and PyMuPDF.

## 🎯 Features

- **📄 PDF Upload**: Upload 3-5 research papers (PDF format)
- **❓ Question Answering**: Ask questions about individual papers or compare across papers
- **📋 Structured Summaries**: Generate Problem, Method, Results, Conclusion summaries
- **🔄 Paper Comparison**: Side-by-side comparison of multiple papers
- **📊 Multi-Paper Synthesis**: Get answers across all papers simultaneously
- **🔍 Smart Extraction**: Proper text extraction from academic papers with metadata

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- **Groq API key (FREE)** - Recommended, or OpenAI API key (paid)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd research-paper-summarie
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get a FREE Groq API key:
   - Visit: https://console.groq.com/keys
   - Sign up (completely free)
   - Create an API key
   - Copy the key

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## 📖 Usage

1. **Choose Provider**: In the sidebar, select "Groq (FREE)" or "OpenAI (Paid)"
2. **Enter API Key**: Enter your Groq API key (free) or OpenAI API key
3. **Upload Papers**: Upload 3-5 research papers in PDF format
4. **Process Papers**: Click "Process Papers" to extract text and build indices
5. **Ask Questions**: Use the "Question Answering" tab to query papers
6. **Generate Summaries**: Use the "Paper Summaries" tab for structured summaries
7. **Compare Papers**: Use the "Compare Papers" tab for side-by-side comparisons
8. **Multi-Paper Query**: Use the "Multi-Paper Query" tab to query across all papers

## 🏗️ Architecture

- **PDF Parser** (`pdf_utils.py`): Extracts text and metadata from PDFs using PyMuPDF
- **RAG Engine** (`rag_engine.py`): Manages vector indices, embeddings, and querying using LlamaIndex and FAISS
- **Streamlit App** (`app.py`): Main UI for user interactions
- **Prompts** (`prompts.py`): Template prompts for summarization and comparison

## 🛠️ Tech Stack

- **UI**: Streamlit
- **RAG Framework**: LlamaIndex
- **Vector Database**: FAISS
- **PDF Parser**: PyMuPDF (fitz)
- **LLM**: Groq (FREE) using Llama 3.1 70B, or OpenAI GPT-4o-mini (paid)
- **Embeddings**: Hugging Face BAAI/bge-small-en-v1.5 (FREE, no API key needed)

## 📝 Example Workflows

### Example 1: Dataset Comparison
- **Upload**: 3 papers on RAG systems
- **Question**: "What datasets did these papers use?"
- **Output**: Table showing datasets used by each paper

### Example 2: Paper Summary
- **Select**: Paper 1
- **Action**: Generate Summary
- **Output**: 
  - Problem: RAG retrieval accuracy issues
  - Method: Hybrid search approach
  - Results: 15% improvement
  - Conclusion: Limitations with small dataset

### Example 3: Method Comparison
- **Select**: Paper 1, Paper 2, Paper 3
- **Question**: "Compare the methods used in these papers"
- **Output**: Synthesized comparison highlighting similarities and differences

## 🎓 Hackathon Evaluation Criteria

This project addresses all evaluation criteria:

- ✅ **Summary quality and structure**: Structured Problem/Method/Results/Conclusion format
- ✅ **Comparison accuracy**: Multi-paper comparison with synthesis
- ✅ **Multi-paper synthesis ability**: Query across all papers simultaneously
- ✅ **PDF handling**: Robust extraction using PyMuPDF with metadata support

## 📁 Project Structure

```
research-paper-summarie/
├── app.py                 # Main Streamlit application
├── rag_engine.py          # RAG engine with LlamaIndex and FAISS
├── pdf_utils.py           # PDF parsing utilities
├── prompts.py             # Prompt templates
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── faiss_index/          # Generated FAISS indices (created at runtime)
```

## 🔧 Configuration

The app supports both free and paid options:
- **Free Option**: Groq API (recommended) - Get free key from https://console.groq.com/keys
- **Paid Option**: OpenAI API - Requires paid account
- **Embeddings**: Uses free Hugging Face embeddings (no API key needed)
- **Chunk Size**: 1024 tokens with 200 token overlap (configurable in `rag_engine.py`)

## 🐛 Troubleshooting

- **API Key Issues**: 
  - For Groq: Get free key from https://console.groq.com/keys
  - For OpenAI: Make sure your API key is valid and has credits
- **PDF Processing Errors**: Ensure PDFs are not corrupted and contain extractable text
- **Memory Issues**: For large papers, consider reducing chunk size or using GPU FAISS
- **Groq Rate Limits**: Free tier has rate limits, but sufficient for hackathon use

## 📄 License

This project is created for hackathon purposes.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
