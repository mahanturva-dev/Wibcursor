"""
Prompt templates for research paper summarization and comparison.
"""

SUMMARY_PROMPT = """You are an expert academic researcher. Analyze the following research paper and provide a structured summary.

Paper Text:
{paper_text}

Please provide a comprehensive summary with the following sections:

1. **Problem**: What problem or research question does this paper address?
2. **Method**: What methodology, approach, or techniques does this paper use?
3. **Results**: What are the main results, findings, or contributions?
4. **Conclusion**: What are the conclusions, implications, and limitations?

Be concise but thorough. Focus on key technical details."""

COMPARISON_PROMPT = """You are an expert academic researcher. Compare the following research papers based on the question asked.

Question: {question}

Papers and their answers:
{paper_answers}

Please provide a comprehensive comparison that:
1. Synthesizes information from all papers
2. Highlights similarities and differences
3. Provides a clear, structured comparison
4. Uses specific details from each paper

Format your response in a clear, organized manner."""

TABLE_COMPARISON_PROMPT = """Create a side-by-side comparison table for the following research papers.

Question/Topic: {question}

Papers and their information:
{paper_info}

Generate a markdown table comparing these papers. Include columns for each paper and rows for different aspects being compared."""

QUESTION_ANSWER_PROMPT = """Answer the following question about the research paper(s) based on the provided context.

Question: {question}

Context from paper(s):
{context}

Provide a clear, accurate answer based only on the provided context. If the information is not available in the context, state that clearly."""

EXTRACTION_PROMPT = """Extract specific information from the following research paper text.

Information to extract: {extraction_type}

Paper Text:
{paper_text}

Provide a concise, structured response with the requested information."""
