# 🚀 Setup and Run Guide

Quick guide to get the RAG system up and running.

---

## 📋 Prerequisites

- Python 3.8 or higher
- (Optional) Weaviate running on localhost:8080
- (Optional) Ollama running on localhost:11434
- (Optional) Google Gemini API key

---

## 🔧 Installation

### Step 1: Install Dependencies

```bash
# Make sure you're in the project root directory
cd rag_system_refactored

# Install all required packages
pip install -r requirements.txt
```

This will install:
- ✅ Core dependencies (numpy, pydantic, python-dotenv)
- ✅ LLM providers (google-generativeai, langchain-ollama)
- ✅ Vector database (weaviate-client)
- ✅ ML models (sentence-transformers, torch)
- ✅ Testing tools (pytest)

### Step 2: Configure Environment (Optional)

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env

# Edit with your values
nano .env  # or use any text editor
```

Example `.env` file:
```env
# Google Gemini API (optional - for LLM generation)
GOOGLE_API_KEY=your_api_key_here

# Weaviate (optional - defaults to localhost)
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080

# Ollama (optional - defaults to localhost)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma:2b-instruct
```

---

## ✅ Verify Installation

Run the validation scripts to ensure everything is set up correctly:

```bash
# Validate each phase
python validate-phase1.py  # Core layer
python validate-phase2.py  # Infrastructure layer (may fail without services)
python validate-phase3.py  # Components layer
python validate-phase4.py  # Application layer (may fail without services)
```

**Expected output:**
```
✅ PASS: Import Tests
✅ PASS: Interface Compliance Tests
✅ PASS: ...
🎉 PHASE X COMPLETE!
```

---

## 🎮 Run the Demo

### Option 1: Quick Demo (with mocks)

Run without any external services:

```bash
python demo.py
```

This will:
- ✅ Check dependencies
- ✅ Run with mock components
- ✅ Show a sample query/response
- ✅ Display performance metrics

### Option 2: Full Demo (with real services)

If you have Weaviate, Ollama, or Gemini API set up:

```bash
python demo.py
```

The script automatically detects available services and uses them.

---

## 🧪 Run Tests

### Integration Tests

```bash
python tests/test_full_pipeline.py
```

This tests:
- ✅ Full pipeline integration
- ✅ Strategy switching
- ✅ Decorator stacking
- ✅ Builder pattern

**Expected output:**
```
✅ PASS: Full Pipeline Integration
✅ PASS: Builder Integration
✅ PASS: Strategy Switching
✅ PASS: Decorator Stacking
🎉 ALL INTEGRATION TESTS PASSED!
```

---

## 🔍 What to Expect

### With Mock Components (No services running)

```
🧪 Running Demo with Mock Components
✅ Pipeline built successfully!
📝 Running test query...
💬 Query: What are positive economic policies?
📝 Answer: This is a mock response...
📚 Sources: 1
⏱️  Total Time: ~50ms
✅ DEMO COMPLETED SUCCESSFULLY
```

### With Real Services (Weaviate + Ollama/Gemini)

```
🚀 Running Demo with Real Services
📦 Building pipeline with real services...
✅ Pipeline built successfully!
📝 Query 1: What are positive economic policies?
📝 Answer: Based on the retrieved statements...
📚 Sources: 3
  [1] Score: 0.923
      We must invest in infrastructure...
⏱️  Performance:
      Total: 1250ms
      Retrieval: 350ms
      Generation: 800ms
✅ DEMO COMPLETED SUCCESSFULLY
```

---

## 🛠️ Troubleshooting

### Issue: ImportError for dependencies

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Weaviate not available"

**Options:**
1. **Use mocks** - The demo will automatically fall back to mocks
2. **Start Weaviate** - If you have Docker:
   ```bash
   docker run -d -p 8080:8080 semitechnologies/weaviate:latest
   ```
3. **Skip Weaviate** - The validation will still work with mocks

### Issue: "Gemini API key not configured"

**Options:**
1. **Use Ollama** - Set up Ollama locally (no API key needed)
2. **Add API key** - Get key from https://makersuite.google.com/app/apikey
3. **Use mocks** - Demo works fine with mocks

### Issue: Models downloading slowly

**Note:** First run downloads models (~100MB):
- sentence-transformers model
- cross-encoder model

This is normal and only happens once. Models are cached locally.

---

## 📊 Understanding the Output

### Performance Metrics

```
⏱️  Performance:
    Total: 1250ms          ← End-to-end time
    Filter extraction: 150ms   ← LLM extracts filters
    Embedding: 50ms            ← Query → vector
    Retrieval: 350ms           ← Database search
    Reranking: 100ms           ← Re-score results
    Generation: 600ms          ← LLM generates answer
```

### Response Structure

```python
response = pipeline.query("Your query", top_k=5)

# Access results:
response.answer              # Generated answer text
response.sources            # List of Document objects
response.filters_applied    # Active filters
response.metrics            # Performance metrics
```

---

## 🎯 Next Steps

### For Development

1. **Explore the architecture:**
   ```bash
   cat README.md
   ```

2. **Review phase docs:**
   - `PHASE_1_COMPLETED.md` - Core layer
   - `PHASE_2_COMPLETED.md` - Infrastructure
   - `PHASE_3_COMPLETED.md` - Components
   - `PHASE_4_COMPLETED.md` - Application
   - `PHASE_5_COMPLETED.md` - Testing

3. **Customize the system:**
   - Add new LLM providers
   - Create custom strategies
   - Add new decorators

### For Testing

```bash
# Run all validations
python validate-phase1.py
python validate-phase2.py
python validate-phase3.py
python validate-phase4.py

# Run integration tests
python tests/test_full_pipeline.py

# Run demo
python demo.py
```

### For Production

1. Set up real Weaviate instance
2. Configure production API keys
3. Load your actual data
4. Optimize performance settings
5. Add monitoring and logging

---

## 📚 Quick Reference

### Build a Pipeline

```python
from application.builders import RAGPipelineBuilder

pipeline = (RAGPipelineBuilder()
    .with_llm(primary="gemini", fallback="ollama")
    .with_embedder("sentence-transformers")
    .with_vector_store(host="localhost", port=8080)
    .with_retrieval_strategy("hybrid")
    .with_caching(cache_size=100)
    .with_reranker("cross-encoder")
    .build())
```

### Query the System

```python
response = pipeline.query("Your question here", top_k=5)
print(response.answer)
print(f"Found {len(response.sources)} sources")
print(f"Time: {response.metrics.total_time_ms:.1f}ms")
```

### Switch Strategies

```python
from components.retrieval.strategies import VectorOnlyStrategy

retriever = pipeline.get_retriever()
retriever.set_strategy(VectorOnlyStrategy(vector_store))
```

---

## 🎉 You're Ready!

The system is now set up and ready to use. Run the demo to see it in action:

```bash
python demo.py
```

For questions or issues, check the documentation in the phase completion files.

**Happy querying! 🚀**