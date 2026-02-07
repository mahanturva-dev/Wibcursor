"""
Research Paper Summarizer - Streamlit App
A hackathon-ready RAG system for analyzing and comparing research papers.
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
import groq

from pdf_utils import PDFParser
from rag_engine import ResearchPaperRAG
import prompts

# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'papers' not in st.session_state:
        st.session_state.papers = {}
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'pdf_parser' not in st.session_state:
        st.session_state.pdf_parser = PDFParser()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_provider' not in st.session_state:
        st.session_state.current_provider = None


def setup_rag_engine():
    """Initialize RAG engine with API key."""
    st.sidebar.markdown("### 🔑 API Configuration")
    
    # Provider selection
    provider = st.sidebar.radio(
        "Choose LLM Provider",
        ["Groq (FREE)", "OpenAI (Paid)"],
        help="Groq offers free tier, OpenAI requires paid API"
    )
    
    provider_name = "groq" if provider == "Groq (FREE)" else "openai"
    
    # Model selection
    if provider_name == "groq":
        groq_models = [
            "mixtral-8x7b-32768",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma-7b-it"
        ]
        selected_model = st.sidebar.selectbox(
            "Select Groq Model",
            groq_models,
            help="Choose the model to use for text generation"
        )
    else:
        openai_models = [
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]
        selected_model = st.sidebar.selectbox(
            "Select OpenAI Model",
            openai_models,
            help="Choose the model to use for text generation"
        )
    
    # Get API key based on provider
    if provider_name == "groq":
        # Check for environment variable first
        api_key = os.getenv("GROQ_API_KEY")
        
        # If not in env, get from sidebar input
        if not api_key:
            api_key = st.sidebar.text_input(
                "Groq API Key (FREE)",
                type="password",
                help="Get free API key from https://console.groq.com/keys",
                value=""
            )
        
        if api_key:
            # Check if we need to reinitialize (provider or model changed)
            needs_reinit = (st.session_state.rag_engine is None or 
                           st.session_state.current_provider != provider_name or
                           st.session_state.current_model != selected_model)
            
            if needs_reinit:
                with st.spinner(f"Initializing RAG engine with Groq ({selected_model})..."):
                    try:
                        st.session_state.rag_engine = ResearchPaperRAG(provider=provider_name, api_key=api_key, model=selected_model)
                        st.session_state.rag_engine.provider = provider_name
                        st.session_state.current_provider = provider_name
                        st.session_state.current_model = selected_model
                        st.sidebar.success(f"✅ Using Groq - {selected_model}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
                        st.sidebar.info("Try a different model from the dropdown above")
                        return False
            return True
        else:
            st.sidebar.warning("⚠️ Please enter your Groq API key")
            st.sidebar.markdown("""
            **Get FREE Groq API Key:**
            1. Visit: https://console.groq.com/keys
            2. Sign up (free)
            3. Create API key
            4. Paste it above
            """)
            return False
    else:  # OpenAI
        # Check for environment variable first
        api_key = os.getenv("OPENAI_API_KEY")
        
        # If not in env, get from sidebar input
        if not api_key:
            api_key = st.sidebar.text_input(
                "OpenAI API Key (Paid)",
                type="password",
                help="Get API key from https://platform.openai.com/api-keys",
                value=""
            )
        
        if api_key:
            # Check if we need to reinitialize (provider or model changed)
            needs_reinit = (st.session_state.rag_engine is None or 
                           st.session_state.current_provider != provider_name or
                           st.session_state.current_model != selected_model)
            
            if needs_reinit:
                with st.spinner(f"Initializing RAG engine with OpenAI ({selected_model})..."):
                    try:
                        st.session_state.rag_engine = ResearchPaperRAG(provider=provider_name, api_key=api_key, model=selected_model)
                        st.session_state.rag_engine.provider = provider_name
                        st.session_state.current_provider = provider_name
                        st.session_state.current_model = selected_model
                        st.sidebar.success(f"✅ Using OpenAI - {selected_model}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
                        return False
            return True
        else:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key")
            st.sidebar.info("💡 Tip: Use Groq (FREE) option above for no cost!")
            return False


def process_uploaded_pdfs(uploaded_files: List) -> Dict[str, Dict]:
    """Process uploaded PDF files."""
    # Check if RAG engine is initialized
    if not st.session_state.rag_engine:
        st.error("❌ Please set your API key in the Configuration section first!")
        st.stop()
    
    processed_papers = {}
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract paper name from filename
            paper_name = Path(uploaded_file.name).stem
            
            # Parse PDF
            try:
                paper_data = st.session_state.pdf_parser.extract_text(tmp_path, paper_name)
                processed_papers[paper_name] = paper_data
                
                # Add to RAG engine (always exists because of check above)
                st.session_state.rag_engine.add_paper(
                    paper_name=paper_name,
                    text=paper_data['full_text'],
                    metadata={
                        'title': paper_data.get('title', ''),
                        'authors': paper_data.get('authors', ''),
                        'num_pages': paper_data.get('num_pages', 0)
                    }
                )
            except Exception as e:
                st.error(f"Error processing {paper_name}: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    return processed_papers


def display_paper_info(papers: Dict[str, Dict]):
    """Display information about uploaded papers."""
    if not papers:
        return
    
    st.subheader("📄 Uploaded Papers")
    
    cols = st.columns(min(len(papers), 3))
    for idx, (paper_name, paper_data) in enumerate(papers.items()):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <h4>{paper_name}</h4>
                    <p><strong>Title:</strong> {paper_data.get('title', 'N/A')[:50]}...</p>
                    <p><strong>Pages:</strong> {paper_data.get('num_pages', 0)}</p>
                </div>
                """, unsafe_allow_html=True)


def generate_summary(paper_name: str):
    """Generate structured summary for a paper."""
    if not st.session_state.rag_engine:
        st.error("RAG engine not initialized. Please enter API key.")
        return
    
    try:
        with st.spinner(f"Generating summary for {paper_name}... This may take a minute."):
            st.info(f"📊 Querying paper: {paper_name}")
            summary = st.session_state.rag_engine.get_paper_summary(paper_name)
            
            st.markdown(f"### 📋 Summary: {paper_name}")
            
            # Show paper metadata if available
            if paper_name in st.session_state.papers:
                paper_data = st.session_state.papers[paper_name]
                if paper_data.get('title'):
                    st.caption(f"**Title:** {paper_data['title']}")
                if paper_data.get('authors'):
                    st.caption(f"**Authors:** {paper_data['authors'][:100]}...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Problem")
                st.info(summary['Problem'])
                
                st.markdown("#### 🧪 Method")
                st.info(summary['Method'])
            
            with col2:
                st.markdown("#### 📊 Results")
                st.success(summary['Results'])
                
                st.markdown("#### 💡 Conclusion")
                st.warning(summary['Conclusion'])
            
            st.success("✅ Summary generated successfully!")
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")
        st.error("Full error details:")
        st.code(str(e))
        st.info("💡 Try checking:\n1. API key is valid\n2. PDF was uploaded successfully\n3. Internet connection is working")


def create_comparison_table(question: str, paper_names: List[str]):
    """Create a comparison table for multiple papers."""
    if not st.session_state.rag_engine:
        st.error("RAG engine not initialized. Please enter API key.")
        return
    
    if len(paper_names) < 2:
        st.warning("Please select at least 2 papers for comparison.")
        return
    
    try:
        with st.spinner("Generating comparison table..."):
            # Get answers for each paper
            paper_answers = {}
            progress_bar = st.progress(0)
            for idx, paper_name in enumerate(paper_names):
                answer = st.session_state.rag_engine.query_paper(paper_name, question)
                paper_answers[paper_name] = answer
                progress_bar.progress((idx + 1) / len(paper_names))
            
            # Create comparison
            comparison = st.session_state.rag_engine.compare_papers(question, paper_names)
            progress_bar.empty()
            
            st.markdown("### 📊 Comparison Results")
            st.markdown(f"**Question:** {question}")
            
            # Display individual answers
            st.markdown("#### Individual Paper Answers")
            for paper_name, answer in paper_answers.items():
                with st.expander(f"📄 {paper_name}"):
                    st.write(answer)
            
            # Display synthesized comparison
            st.markdown("#### 🔄 Synthesized Comparison")
            st.write(comparison)
            
            # Create a simple table format
            st.markdown("#### 📋 Comparison Table")
            table_data = {
                'Paper': paper_names,
                'Answer': [paper_answers[p][:200] + "..." if len(paper_answers[p]) > 200 else paper_answers[p] for p in paper_names]
            }
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error generating comparison: {str(e)}")
        st.info("Please try again or check your API key.")


def main():
    """Main application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 Research Paper Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar setup
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Clear cache button
        if st.button("🔄 Clear Cache & Reset", help="Clear session data and reinitialize"):
            st.session_state.rag_engine = None
            st.session_state.current_model = None
            st.session_state.current_provider = None
            st.rerun()
        
        rag_ready = setup_rag_engine()
        
        st.markdown("---")
        st.header("📤 Upload Papers")
        
        if not rag_ready:
            st.warning("⚠️ Set your API key above first!")
            st.markdown("Steps:")
            st.markdown("1. Choose LLM Provider (Groq or OpenAI)")
            st.markdown("2. Enter API key")
            st.markdown("3. Once configured, upload your PDFs below")
        else:
            st.success("✅ Ready to upload papers!")
            st.markdown("Upload 3-5 research papers (PDF format)")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select multiple PDF files to upload",
            disabled=not rag_ready
        )
        
        if uploaded_files:
            if st.button("Process Papers", type="primary", disabled=not rag_ready):
                with st.spinner("Processing PDFs..."):
                    processed = process_uploaded_pdfs(uploaded_files)
                    st.session_state.papers.update(processed)
                    st.session_state.uploaded_files = [f.name for f in uploaded_files]
                    st.success(f"✅ Processed {len(processed)} paper(s)!")
                    st.rerun()
        
        # Display uploaded papers count
        if st.session_state.papers:
            st.markdown("---")
            st.markdown(f"**Papers loaded:** {len(st.session_state.papers)}")
            for paper_name in st.session_state.papers.keys():
                st.markdown(f"- {paper_name}")
    
    # Main content area
    if not st.session_state.papers:
        st.info("👆 Please upload research papers using the sidebar to get started.")
        st.markdown("""
        ### Features:
        - 📄 **PDF Upload**: Upload multiple research papers
        - ❓ **Question Answering**: Ask questions about individual papers or compare across papers
        - 📋 **Structured Summaries**: Generate Problem, Method, Results, Conclusion summaries
        - 📊 **Comparison Tables**: Side-by-side comparison of papers
        - 🔍 **Multi-Paper Synthesis**: Get answers across all papers
        """)
    else:
        display_paper_info(st.session_state.papers)
        st.markdown("---")
        
        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "❓ Question Answering",
            "📋 Paper Summaries",
            "🔄 Compare Papers",
            "📊 Multi-Paper Query"
        ])
        
        with tab1:
            st.header("Ask Questions About Papers")
            
            # Example questions
            st.markdown("**💡 Example Questions:**")
            example_questions = [
                "What datasets did this paper use?",
                "What methods were employed?",
                "What are the main contributions?",
                "What are the limitations mentioned?",
                "What evaluation metrics were used?"
            ]
            selected_example = st.selectbox("Or choose an example:", [""] + example_questions)
            
            paper_selection = st.selectbox(
                "Select a paper (or 'All Papers')",
                ["All Papers"] + list(st.session_state.papers.keys())
            )
            
            question = st.text_input(
                "Enter your question",
                value=selected_example if selected_example else "",
                placeholder="e.g., What datasets did this paper use? What methods were employed?"
            )
            
            if st.button("Get Answer", type="primary") and question:
                if not rag_ready:
                    st.error("❌ Please enter API key in the Configuration section first.")
                else:
                    try:
                        if paper_selection == "All Papers":
                            with st.spinner("Querying all papers..."):
                                answers = st.session_state.rag_engine.query_all_papers(question)
                                
                                st.markdown(f"### Answer for: {question}")
                                for paper_name, answer in answers.items():
                                    with st.expander(f"📄 {paper_name}"):
                                        st.write(answer)
                                st.success("✅ Query completed!")
                        else:
                            with st.spinner("Generating answer..."):
                                st.info(f"📊 Querying: {paper_selection}")
                                answer = st.session_state.rag_engine.query_paper(paper_selection, question)
                                st.markdown(f"### Answer")
                                st.write(answer)
                                st.success("✅ Answer generated!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.error("Full error details:")
                        st.code(str(e))
                        st.info("💡 Try checking:\n1. API key is valid\n2. PDF was uploaded and processed\n3. Internet connection works")
        
        with tab2:
            st.header("Generate Structured Summaries")
            
            summary_paper = st.selectbox(
                "Select a paper to summarize",
                list(st.session_state.papers.keys()),
                key="summary_select"
            )
            
            if st.button("Generate Summary", type="primary"):
                if not rag_ready:
                    st.error("Please enter OpenAI API key in the sidebar.")
                else:
                    try:
                        generate_summary(summary_paper)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        st.info("Please check your API key and try again.")
        
        with tab3:
            st.header("Compare Multiple Papers")
            
            st.markdown("Select papers to compare and ask a comparison question.")
            
            comparison_papers = st.multiselect(
                "Select papers to compare",
                list(st.session_state.papers.keys()),
                default=list(st.session_state.papers.keys())[:2] if len(st.session_state.papers) >= 2 else []
            )
            
            comparison_question = st.text_input(
                "Enter comparison question",
                placeholder="e.g., What datasets did these papers use? Compare the methods used.",
                key="comparison_q"
            )
            
            if st.button("Compare Papers", type="primary") and comparison_question:
                if not rag_ready:
                    st.error("Please enter OpenAI API key in the sidebar.")
                else:
                    create_comparison_table(comparison_question, comparison_papers)
        
        with tab4:
            st.header("Query Across All Papers")
            
            st.markdown("Ask a question that will be answered using information from all papers.")
            
            multi_question = st.text_input(
                "Enter your question",
                placeholder="e.g., What are the common datasets used across all papers?",
                key="multi_q"
            )
            
            if st.button("Query All Papers", type="primary") and multi_question:
                if not rag_ready:
                    st.error("Please enter OpenAI API key in the sidebar.")
                else:
                    try:
                        with st.spinner("Querying all papers and synthesizing..."):
                            # Get individual answers
                            individual_answers = st.session_state.rag_engine.query_all_papers(multi_question)
                            
                            # Create synthesis
                            synthesis = st.session_state.rag_engine.compare_papers(multi_question)
                            
                            st.markdown(f"### Question: {multi_question}")
                            
                            st.markdown("#### Individual Answers")
                            for paper_name, answer in individual_answers.items():
                                with st.expander(f"📄 {paper_name}"):
                                    st.write(answer)
                            
                            st.markdown("#### 🔄 Synthesized Answer")
                            st.write(synthesis)
                    except Exception as e:
                        st.error(f"Error querying papers: {str(e)}")
                        st.info("Please check your API key and try again.")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>Built with ❤️ using Streamlit, LlamaIndex, FAISS, and PyMuPDF</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
