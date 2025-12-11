# RAG System - Refactored Architecture

A production-ready Retrieval-Augmented Generation (RAG) system built with **clean architecture principles**, **SOLID principles**, and **design patterns**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-clean%20architecture-brightgreen.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![Patterns](https://img.shields.io/badge/patterns-6%20implemented-orange.svg)](#design-patterns)

---

## 🏗️ Architecture

This system follows a **5-layer clean architecture** with strict dependency rules:

```
┌─────────────────────────────────────────────────────────────┐
│  5. Application Layer (Orchestration)                       │
│     Factories, Builders, Facades                            │
├─────────────────────────────────────────────────────────────┤
│  4. Components Layer (Business Logic)                       │
│     Strategies, Decorators, Domain Services                 │
├─────────────────────────────────────────────────────────────┤
│  3. Infrastructure Layer (External Adapters)                │
│     Gemini, Ollama, Weaviate, SentenceTransformer          │
├─────────────────────────────────────────────────────────────┤
│  2. Core Interfaces (Contracts)                             │
│     Abstract base classes defining contracts                │
├─────────────────────────────────────────────────────────────┤
│  1. Domain Layer (Pure Data Models)                         │
│     Query, Document, Response, Metrics                      │
└─────────────────────────────────────────────────────────────┘

Dependencies flow INWARD: 5→4→3→2→1 (never outward!)
```

---

## 🎭 Design Patterns

This system demonstrates **6 design patterns**:

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| **Adapter** | Unify external APIs | Gemini, Ollama, Weaviate adapters |
| **Strategy** | Swappable algorithms | Vector, Keyword, Hybrid retrieval |
| **Decorator** | Add cross-cutting concerns | Caching, Timing decorators |
| **Factory** | Centralized object creation | LLM & Model factories with fallback |
| **Builder** | Fluent pipeline construction | RAGPipelineBuilder |
| **Facade** | Simple interface to complex system | RAGPipeline |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd rag_system_refactored

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from application.builders import RAGPipelineBuilder

# Build pipeline with fluent API
pipeline = (RAGPipelineBuilder()
    .with_llm(primary="gemini", fallback="ollama")
    .with_embedder("sentence-transformers")
    .with_vector_store(host="localhost", port=8080)
    .with_retrieval_strategy("hybrid")
    .with_caching(cache_size=100)
    .with_reranker("cross-encoder")
    .build())

# Query the system
response = pipeline.query("What are positive economic policies?", top_k=5)

# Access results
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
print(f"Time: {response.metrics.total_time_ms:.1f}ms")
```

---

## 📂 Project Structure

```
rag_system_refactored/
├── core/                          # Layer 1 & 2: Domain + Interfaces
│   ├── domain/                    # Pure data models
│   │   ├── query.py
│   │   ├── document.py
│   │   ├── response.py
│   │   └── metrics.py
│   └── interfaces/                # Abstract contracts
│       ├── llm_provider.py
│       ├── vector_store.py
│       ├── embedder.py
│       ├── reranker.py
│       └── retrieval_strategy.py
│
├── infrastructure/                # Layer 3: External Adapters
│   ├── adapters/
│   │   ├── gemini_adapter.py      # Google Gemini LLM
│   │   ├── ollama_adapter.py      # Local Ollama LLM
│   │   ├── weaviate_adapter.py    # Weaviate vector DB
│   │   ├── sentence_transformer_embedder.py
│   │   └── cross_encoder_reranker.py
│   └── config/
│       └── settings.py            # Configuration management
│
├── components/                    # Layer 4: Business Logic
│   ├── retrieval/
│   │   ├── strategies/            # Strategy Pattern
│   │   │   ├── vector_only_strategy.py
│   │   │   ├── keyword_only_strategy.py
│   │   │   └── hybrid_strategy.py
│   │   ├── decorators/            # Decorator Pattern
│   │   │   ├── caching_retriever.py
│   │   │   └── timing_retriever.py
│   │   └── retriever.py           # Context class
│   └── filters/
│       └── filter_manager.py      # Filter state management
│
├── application/                   # Layer 5: Orchestration
│   ├── factories/                 # Factory Pattern
│   │   ├── llm_factory.py
│   │   └── model_factory.py
│   ├── builders/                  # Builder Pattern
│   │   └── pipeline_builder.py
│   └── pipeline/                  # Facade Pattern
│       └── rag_pipeline.py
│
├── tests/                         # Test suite
│   └── test_full_pipeline.py      # Integration tests
│
├── validate-phase1.py             # Validation scripts
├── validate-phase2.py
├── validate-phase3.py
├── validate-phase4.py
│
└── README.md                      # This file
```

---

## 🎯 SOLID Principles

Every component follows SOLID principles:

- **S**ingle Responsibility: Each class has one job
- **O**pen/Closed: Extensible without modification
- **L**iskov Substitution: Implementations are interchangeable
- **I**nterface Segregation: Focused, minimal interfaces
- **D**ependency Inversion: Depend on abstractions, not concretions

---

## 💡 Key Features

### 🔄 **Swappable Components**

Change LLM providers, retrieval strategies, or any component without modifying code:

```python
# Switch LLM at runtime
llm = LLMFactory.create_with_fallback(primary="gemini", fallback="ollama")

# Switch retrieval strategy
pipeline.get_retriever().set_strategy(VectorOnlyStrategy(vector_store))
```

### 📦 **Caching & Performance**

Built-in caching and performance tracking:

```python
# Enable caching
.with_caching(cache_size=100)

# Enable timing
.with_timing(threshold_ms=1000)

# Access metrics
print(response.metrics.retrieval_time_ms)  # Time for each stage
```

### 🎨 **Fluent API**

Readable, self-documenting pipeline construction:

```python
pipeline = (RAGPipelineBuilder()
    .with_llm(primary="gemini", fallback="ollama")
    .with_embedder("sentence-transformers")
    .with_vector_store(host="localhost", port=8080)
    .with_retrieval_strategy("hybrid", rrf_k=60)
    .with_caching(cache_size=100)
    .with_timing(threshold_ms=1000)
    .with_reranker("cross-encoder")
    .build())
```

### 🔍 **Multiple Retrieval Strategies**

- **Vector-Only**: Pure semantic search
- **Keyword-Only**: Pure BM25 search
- **Hybrid**: Combines both with RRF fusion

```python
# Different strategies for different queries
retriever.set_strategy(VectorOnlyStrategy(store))    # Semantic
retriever.set_strategy(KeywordOnlyStrategy(store))   # Exact terms
retriever.set_strategy(HybridStrategy(store))        # Best of both
```

---

## 🧪 Testing

### Run Validation Scripts

```bash
# Validate each phase
python validate-phase1.py  # Core layer
python validate-phase2.py  # Infrastructure layer
python validate-phase3.py  # Components layer
python validate-phase4.py  # Application layer
```

### Run Integration Tests

```bash
# Full pipeline integration test
python tests/test_full_pipeline.py
```

### Expected Output

```
✅ PASS: Import Tests
✅ PASS: Factory Pattern Tests
✅ PASS: Builder Pattern Tests
✅ PASS: Strategy Pattern Tests
✅ PASS: Decorator Pattern Tests
✅ PASS: Full Pipeline Integration
🎉 ALL TESTS PASSED!
```

---

## 📚 Usage Examples

### Example 1: Basic Query

```python
from application.builders import RAGPipelineBuilder

pipeline = (RAGPipelineBuilder()
    .with_llm(primary="gemini", fallback="ollama")
    .with_embedder("sentence-transformers")
    .with_vector_store(host="localhost", port=8080)
    .with_retrieval_strategy("hybrid")
    .build())

response = pipeline.query("What is machine learning?")
print(response.answer)
```

### Example 2: With Caching & Timing

```python
pipeline = (RAGPipelineBuilder()
    .with_llm(primary="ollama")
    .with_embedder("sentence-transformers")
    .with_vector_store(host="localhost", port=8080)
    .with_retrieval_strategy("hybrid")
    .with_caching(cache_size=100)      # Enable caching
    .with_timing(threshold_ms=1000)    # Enable timing
    .build())

response = pipeline.query("What is AI?")

# Access performance metrics
print(f"Total: {response.metrics.total_time_ms:.1f}ms")
print(f"Retrieval: {response.metrics.retrieval_time_ms:.1f}ms")
print(f"Generation: {response.metrics.generation_time_ms:.1f}ms")
```

### Example 3: Strategy Switching

```python
# Start with hybrid
retriever = pipeline.get_retriever()

# Query with hybrid strategy
response1 = pipeline.query("complex semantic query")

# Switch to vector-only for semantic queries
from components.retrieval.strategies import VectorOnlyStrategy
retriever.set_strategy(VectorOnlyStrategy(vector_store))

response2 = pipeline.query("another semantic query")
```

### Example 4: Context Manager

```python
# Automatic resource cleanup
with pipeline:
    response = pipeline.query("What is deep learning?")
    print(response.answer)
# Resources automatically closed
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# LLM Configuration
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
OLLAMA_MODEL=gemma:2b-instruct
OLLAMA_BASE_URL=http://localhost:11434

# Weaviate Configuration
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_COLLECTION=PoliticalStatements

# Model Configuration
EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
DEVICE=cuda  # or 'cpu'
```

### Settings Class

```python
from infrastructure.config import get_settings

settings = get_settings()
print(settings.gemini_model)
print(settings.weaviate_host)
```

---

## 🔧 Extending the System

### Adding a New LLM Provider

1. Create adapter implementing `LLMProvider`:

```python
# infrastructure/adapters/new_llm_adapter.py
from core.interfaces import LLMProvider

class NewLLMAdapter(LLMProvider):
    def generate(self, prompt: str) -> str:
        # Your implementation
        pass
    
    def is_available(self) -> bool:
        # Check availability
        pass
```

2. Add to factory:

```python
# application/factories/llm_factory.py
@staticmethod
def _create_single(provider_type, settings):
    if provider_type == "new_llm":
        return NewLLMAdapter(settings.new_llm_api_key)
    # ... existing providers
```

3. Use it:

```python
pipeline = (RAGPipelineBuilder()
    .with_llm(primary="new_llm", fallback="ollama")
    .build())
```

**No other code changes needed!** ✨

### Adding a New Retrieval Strategy

1. Create strategy implementing `RetrievalStrategy`:

```python
# components/retrieval/strategies/custom_strategy.py
from core.interfaces import RetrievalStrategy

class CustomStrategy(RetrievalStrategy):
    def retrieve(self, query_vector, query_text, top_k, filters):
        # Your custom retrieval logic
        pass
```

2. Use it:

```python
from components.retrieval.strategies import CustomStrategy

retriever.set_strategy(CustomStrategy(vector_store))
```

---

## 📈 Performance

Typical performance metrics:

- **Filter Extraction**: 50-200ms
- **Query Embedding**: 20-50ms
- **Retrieval**: 100-500ms
- **Reranking**: 50-150ms
- **Answer Generation**: 500-2000ms

**Total End-to-End**: 700-3000ms

Caching can reduce repeated queries to <50ms!

---

## 🤝 Contributing

This is an academic project demonstrating software design principles. Key areas:

1. **Design Patterns**: All 6 patterns properly implemented
2. **SOLID Principles**: Strict adherence throughout
3. **Clean Architecture**: Clear layer separation
4. **Type Safety**: Full type hints
5. **Documentation**: Comprehensive docstrings

---

## 📝 Documentation

- **Architecture**: See `docs/architecture.md`
- **Phase 1 (Core)**: See `PHASE_1_COMPLETED.md`
- **Phase 2 (Infrastructure)**: See `PHASE_2_COMPLETED.md`
- **Phase 3 (Components)**: See `PHASE_3_COMPLETED.md`
- **Phase 4 (Application)**: See `PHASE_4_COMPLETED.md`

---

## 🎓 Learning Resources

This project demonstrates:

- **Clean Architecture** (Robert C. Martin)
- **SOLID Principles** (Object-oriented design)
- **Design Patterns** (Gang of Four)
- **Domain-Driven Design** (Eric Evans)
- **Dependency Inversion Principle**

---

## 🏆 Key Achievements

✅ **Modular Design**: Swap any component without breaking others  
✅ **Testable**: All components mockable and unit-testable  
✅ **Extensible**: Add new providers/strategies easily  
✅ **Maintainable**: Clear structure, comprehensive docs  
✅ **Type-Safe**: Full type hints throughout  
✅ **Production-Ready**: Error handling, logging, metrics  

---

## 📧 Contact

For questions about this architecture or implementation:
- Review the phase completion documents
- Check the validation scripts
- See the integration tests

---

**Built with ❤️ using Clean Architecture & Design Patterns**