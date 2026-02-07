# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key (you'll need it in the app)

## Step 3: Run the Application

```bash
streamlit run app.py
```

## Step 4: Use the App

1. **Enter API Key**: Paste your OpenAI API key in the sidebar
2. **Upload Papers**: Upload 3-5 PDF research papers
3. **Process Papers**: Click "Process Papers" button
4. **Start Querying**: Use the tabs to ask questions, generate summaries, or compare papers

## 💡 Example Questions

### Single Paper Questions:
- "What datasets did this paper use?"
- "What methods were employed?"
- "What are the main contributions?"
- "What are the limitations?"

### Comparison Questions:
- "What datasets did these papers use?"
- "Compare the methods used in these papers"
- "What are the common evaluation metrics?"

### Multi-Paper Questions:
- "What are the common datasets used across all papers?"
- "Summarize the key findings from all papers"

## 🎯 Demo Workflow

1. Upload 3 papers on RAG systems
2. Go to "Question Answering" tab
3. Ask: "What datasets did these papers use?"
4. Select "All Papers" and get answers
5. Go to "Paper Summaries" tab
6. Select a paper and generate summary
7. Go to "Compare Papers" tab
8. Select 2-3 papers and ask: "Compare the methods used"
9. View the comparison table

## ⚠️ Troubleshooting

- **API Key Error**: Make sure your OpenAI API key is valid and has credits
- **PDF Processing Error**: Ensure PDFs contain extractable text (not scanned images)
- **Slow Processing**: Large papers may take time. Be patient!
- **Memory Issues**: Close other applications if you run out of memory

## 📊 Tips for Best Results

1. **Paper Quality**: Use PDFs with extractable text (not scanned images)
2. **Question Clarity**: Ask specific, clear questions
3. **Paper Count**: 3-5 papers work best for comparisons
4. **API Costs**: Using gpt-4o-mini keeps costs low for hackathons

## 🎓 Hackathon Presentation Tips

1. **Demo Flow**: 
   - Show PDF upload → Processing → Question answering → Summary → Comparison
2. **Highlight Features**:
   - Multi-paper synthesis
   - Structured summaries
   - Comparison tables
3. **Show Use Cases**:
   - Literature review
   - Research comparison
   - Quick paper understanding
