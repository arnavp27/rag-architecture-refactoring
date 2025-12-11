#!/usr/bin/env python3
"""
RAG System Demo - Test the refactored system

This script demonstrates the complete RAG pipeline with real components.
It will attempt to use real services if available, otherwise fall back to mocks.

Usage:
    python demo.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n" + "="*70)
    print("🔍 Checking Dependencies")
    print("="*70)
    
    missing = []
    
    try:
        import weaviate
        print("✅ weaviate-client installed")
    except ImportError:
        missing.append("weaviate-client")
        print("❌ weaviate-client not installed")
    
    try:
        import google.generativeai as genai
        print("✅ google-generativeai installed")
    except ImportError:
        missing.append("google-generativeai")
        print("❌ google-generativeai not installed")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers installed")
    except ImportError:
        missing.append("sentence-transformers")
        print("❌ sentence-transformers not installed")
    
    try:
        import torch
        print("✅ torch installed")
    except ImportError:
        missing.append("torch")
        print("❌ torch not installed")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_services():
    """Check if required services are running"""
    print("\n" + "="*70)
    print("🔍 Checking Services")
    print("="*70)
    
    services_ok = True
    
    # Check Weaviate
    try:
        import weaviate
        client = weaviate.connect_to_local(host="localhost", port=8080)
        if client.is_ready():
            print("✅ Weaviate is running (localhost:8080)")
            client.close()
        else:
            print("⚠️  Weaviate connected but not ready")
            services_ok = False
    except Exception as e:
        print(f"❌ Weaviate not available: {e}")
        services_ok = False
    
    # Check Ollama
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running (localhost:11434)")
        else:
            print("⚠️  Ollama responded but with error")
            services_ok = False
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        # Ollama is optional (can use Gemini)
    
    # Check for API keys
    from infrastructure.config import get_settings
    settings = get_settings()
    
    if settings.google_api_key:
        print("✅ Gemini API key configured")
    else:
        print("⚠️  Gemini API key not configured (set GOOGLE_API_KEY in .env)")
        services_ok = False
    
    return services_ok


def run_demo_with_mocks():
    """Run demo with mock components"""
    print("\n" + "="*70)
    print("🧪 Running Demo with Mock Components")
    print("="*70)
    
    from application.builders import RAGPipelineBuilder
    from core.interfaces import LLMProvider, Embedder, VectorStore
    import numpy as np
    
    # Mock components
    class MockLLM(LLMProvider):
        def generate(self, prompt: str) -> str:
            return "This is a mock response. The real system would analyze political statements here."
        def generate_structured(self, prompt: str, schema: dict) -> dict:
            return {}
        def is_available(self) -> bool:
            return True
    
    class MockEmbedder(Embedder):
        def embed_query(self, text: str) -> np.ndarray:
            return np.random.rand(384)
        def embed_batch(self, texts: list) -> np.ndarray:
            return np.random.rand(len(texts), 384)
        def get_dimension(self) -> int:
            return 384
    
    class MockVectorStore(VectorStore):
        def vector_search(self, query_vector, top_k, filters=None):
            return [{
                "content": "Mock political statement about economic policy.",
                "score": 0.95,
                "embedding_index": 1,
                "metadata": {"theme": "Economy", "sentiment": "Positive"}
            }]
        def keyword_search(self, query_text, top_k, filters=None):
            return []
        def hybrid_search(self, query_vector, query_text, top_k, alpha=0.5, filters=None):
            return []
        def close(self):
            pass
    
    try:
        # Build pipeline with mocks
        from components.retrieval import Retriever
        from components.retrieval.strategies import HybridStrategy
        from application.pipeline import RAGPipeline
        
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder()
        mock_store = MockVectorStore()
        retriever = Retriever(strategy=HybridStrategy(mock_store))
        
        pipeline = RAGPipeline(
            llm=mock_llm,
            embedder=mock_embedder,
            retriever=retriever,
            reranker=None,
            vector_store=mock_store
        )
        
        print("\n✅ Pipeline built successfully!")
        
        # Test query
        print("\n📝 Running test query...")
        query = "What are positive economic policies?"
        
        response = pipeline.query(query, top_k=3)
        
        print(f"\n✅ Query executed successfully!")
        print(f"\n💬 Query: {query}")
        print(f"\n📝 Answer:\n{response.answer}")
        print(f"\n📚 Sources: {len(response.sources)}")
        print(f"\n⏱️  Total Time: {response.metrics.total_time_ms:.1f}ms")
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETED SUCCESSFULLY (with mocks)")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_demo_with_real_services():
    """Run demo with real services"""
    print("\n" + "="*70)
    print("🚀 Running Demo with Real Services")
    print("="*70)
    
    try:
        from application.builders import RAGPipelineBuilder
        
        print("\n📦 Building pipeline with real services...")
        print("   (This may take a moment to load models)")
        
        # Build pipeline
        pipeline = (RAGPipelineBuilder()
            .with_llm(primary="ollama", fallback="gemini")
            .with_embedder("sentence-transformers")
            .with_vector_store(host="localhost", port=8080)
            .with_retrieval_strategy("hybrid")
            .with_caching(cache_size=10)
            .build())
        
        print("\n✅ Pipeline built successfully!")
        
        # Test queries
        queries = [
            "What are positive economic policies?",
            "What statements discuss healthcare?",
            "Show me recent political statements"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*70}")
            print(f"📝 Query {i}: {query}")
            print('='*70)
            
            response = pipeline.query(query, top_k=3)
            
            print(f"\n📝 Answer:\n{response.answer}\n")
            print(f"📚 Sources: {len(response.sources)}")
            
            for j, doc in enumerate(response.sources, 1):
                print(f"\n  [{j}] Score: {doc.score:.3f}")
                print(f"      {doc.content[:100]}...")
            
            print(f"\n⏱️  Performance:")
            print(f"      Total: {response.metrics.total_time_ms:.1f}ms")
            print(f"      Retrieval: {response.metrics.retrieval_time_ms:.1f}ms")
            print(f"      Generation: {response.metrics.generation_time_ms:.1f}ms")
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETED SUCCESSFULLY (with real services)")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo with real services failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo function"""
    print("\n" + "="*70)
    print("🚀 RAG System Demo - Refactored Architecture")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Check services
    services_available = check_services()
    
    if services_available:
        print("\n✅ All services available! Running with real components...")
        success = run_demo_with_real_services()
    else:
        print("\n⚠️  Some services unavailable. Running with mock components...")
        success = run_demo_with_mocks()
    
    if success:
        print("\n" + "="*70)
        print("🎉 Demo completed successfully!")
        print("="*70)
        print("\n📚 Next steps:")
        print("   1. Check the validation scripts: python validate-phase4.py")
        print("   2. Run integration tests: python tests/test_full_pipeline.py")
        print("   3. Review the architecture: see README.md")
        print("   4. Customize for your use case")
        return 0
    else:
        print("\n❌ Demo failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())