# RAG System - Refactored Architecture

A production-ready Retrieval-Augmented Generation (RAG) system built with clean architecture principles, SOLID principles, and design patterns.

## 🏗️ Architecture

This system follows a **5-layer clean architecture**:

1. **Domain** (`core/domain/`) - Pure data models
2. **Core** (`core/interfaces/`) - Abstract interfaces/contracts
3. **Infrastructure** (`infrastructure/`) - External service adapters
4. **Components** (`components/`) - Business logic (strategies, decorators)
5. **Application** (`application/`) - High-level orchestration (factories, builders, facade)

## 🎭 Design Patterns

- **Adapter Pattern** - Unify different APIs (Gemini, Ollama, Weaviate)
- **Strategy Pattern** - Swappable retrieval algorithms
- **Decorator Pattern** - Add caching, timing without modifying core logic
- **Factory Pattern** - Centralized object creation with fallback
- **Builder Pattern** - Fluent API for pipeline construction
- **Facade Pattern** - Simple interface to complex subsystem

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
from application.builders.pipeline_builder import RAGPipelineBuilder

# Build pipeline with fluent API
pipeline = (RAGPipelineBuilder()
    .with_llm(primary="ollama", fallback="gemini")
    .with_embedder("sentence-transformers")
    .with_retrieval_strategy("hybrid")
    .with_caching(cache_size=100)
    .with_reranker("cross-encoder")
    .build())

# Query the system
response = pipeline.query("Your question here", top_k=5)
print(response.answer)
```

## 📂 Project Structure

```
rag_system_refactored/
├── core/                    # Layer 1 & 2: Domain + Interfaces
│   ├── domain/              # Data classes
│   └── interfaces/          # Abstract base classes
├── infrastructure/          # Layer 3: External integrations
│   ├── adapters/            # API adapters
│   └── config/              # Configuration
├── components/              # Layer 4: Business logic
│   ├── retrieval/           # Strategies & decorators
│   └── filters/             # Filter management
├── application/             # Layer 5: Orchestration
│   ├── factories/           # Object creation
│   ├── builders/            # Pipeline construction
│   └── pipeline/            # Main facade
├── tests/                   # Test suite
└── legacy/                  # Old code (during migration)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# With coverage
pytest --cov=. --cov-report=html
```

## 📝 Development Guidelines

### Adding a New LLM Provider

1. Create adapter in `infrastructure/adapters/`
2. Implement `LLMProvider` interface
3. Add factory method in `LLMFactory`
4. No changes needed to existing code!

### Adding a New Retrieval Strategy

1. Create strategy in `components/retrieval/strategies/`
2. Implement `RetrievalStrategy` interface
3. Can be used immediately with existing pipeline

## 🔧 Configuration

Set environment variables in `.env`:

```env
GOOGLE_API_KEY=your_gemini_key
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
```

## 📚 Documentation

- Architecture decisions: See `docs/architecture.md` (TODO)
- API documentation: Run `make docs` (TODO)
- Design patterns: See `docs/patterns.md` (TODO)

## 🤝 Contributing

1. Follow the layer dependency rules (Application → Core ← Infrastructure)
2. Write tests for new features
3. Update documentation
4. Run linters: `black .` and `pylint src/`

## 👥 Authors

Arnav Patil
