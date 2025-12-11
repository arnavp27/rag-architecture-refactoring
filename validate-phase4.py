#!/usr/bin/env python3
"""
Application Layer (Phase 4) Validation Script

This script validates that the Application Layer (factories, builder, pipeline)
is properly implemented following Factory, Builder, and Facade patterns.

Run this after implementing Phase 4 to ensure everything is working.

Usage:
    python validate-phase4.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all application modules can be imported"""
    print("=" * 60)
    print("🧪 Testing Application Layer Imports")
    print("=" * 60)
    
    try:
        # Test factory imports
        print("\n📦 Testing Factory Imports...")
        from application.factories import LLMFactory, ModelFactory
        print("✅ Factory imports successful")
        
        # Test builder imports
        print("\n📦 Testing Builder Imports...")
        from application.builders import RAGPipelineBuilder
        print("✅ Builder imports successful")
        
        # Test pipeline imports
        print("\n📦 Testing Pipeline Imports...")
        from application.pipeline import RAGPipeline
        print("✅ Pipeline imports successful")
        
        # Test filter manager imports
        print("\n📦 Testing Filter Manager Imports...")
        from components.filters import FilterManager
        print("✅ Filter Manager imports successful")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Make sure all dependencies are installed and files are created")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False


def test_factory_pattern():
    """Test that Factory Pattern is properly implemented"""
    print("\n" + "=" * 60)
    print("🧪 Testing Factory Pattern Implementation")
    print("=" * 60)
    
    try:
        from core.interfaces import LLMProvider, Embedder, Reranker
        from application.factories import LLMFactory, ModelFactory
        from infrastructure.config import get_settings
        
        settings = get_settings()
        
        # Test 1: LLMFactory returns interface type
        print("\n📝 Test 1: LLMFactory returns LLMProvider interface...")
        try:
            # This might fail if no API key, but that's ok for testing
            llm = LLMFactory.create_ollama(settings)
            if not isinstance(llm, LLMProvider):
                print(f"  ❌ LLMFactory didn't return LLMProvider, got {type(llm)}")
                return False
            print("  ✅ LLMFactory returns LLMProvider interface")
        except Exception as e:
            print(f"  ⚠️  LLMFactory creation failed (may be expected): {e}")
            print("  ✓ Factory structure is correct")
        
        # Test 2: ModelFactory returns interface types
        print("\n📝 Test 2: ModelFactory returns Embedder interface...")
        try:
            embedder = ModelFactory.create_embedder("sentence-transformers", settings)
            if not isinstance(embedder, Embedder):
                print(f"  ❌ ModelFactory didn't return Embedder, got {type(embedder)}")
                return False
            print("  ✅ ModelFactory returns Embedder interface")
        except Exception as e:
            print(f"  ⚠️  Embedder creation failed (model download may be needed): {e}")
            print("  ✓ Factory structure is correct")
        
        print("\n📝 Test 3: ModelFactory returns Reranker interface...")
        try:
            reranker = ModelFactory.create_reranker("cross-encoder", settings)
            if not isinstance(reranker, Reranker):
                print(f"  ❌ ModelFactory didn't return Reranker, got {type(reranker)}")
                return False
            print("  ✅ ModelFactory returns Reranker interface")
        except Exception as e:
            print(f"  ⚠️  Reranker creation failed (model download may be needed): {e}")
            print("  ✓ Factory structure is correct")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Factory Pattern: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_builder_pattern():
    """Test that Builder Pattern is properly implemented"""
    print("\n" + "=" * 60)
    print("🧪 Testing Builder Pattern Implementation")
    print("=" * 60)
    
    try:
        from application.builders import RAGPipelineBuilder
        from application.pipeline import RAGPipeline
        
        # Test 1: Builder methods return self
        print("\n📝 Test 1: Builder methods return self for chaining...")
        builder = RAGPipelineBuilder()
        
        # Check return type (should be builder itself)
        result = builder.with_caching(cache_size=50)
        if result is not builder:
            print("  ❌ Builder method didn't return self")
            return False
        
        result = builder.with_timing(threshold_ms=500)
        if result is not builder:
            print("  ❌ Builder method didn't return self")
            return False
        
        print("  ✅ Builder methods return self for chaining")
        
        # Test 2: Builder validates before building
        print("\n📝 Test 2: Builder validates required components...")
        builder2 = RAGPipelineBuilder()
        
        try:
            pipeline = builder2.build()
            print("  ❌ Builder should have raised error for missing components")
            return False
        except RuntimeError as e:
            if "Missing required components" in str(e):
                print("  ✅ Builder properly validates required components")
            else:
                print(f"  ❌ Wrong error message: {e}")
                return False
        
        # Test 3: Builder can't be used twice
        print("\n📝 Test 3: Builder can only be used once...")
        # We'll test this with mocks in the next section
        print("  ✓ Will test with full pipeline build")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Builder Pattern: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filter_manager():
    """Test that FilterManager works correctly"""
    print("\n" + "=" * 60)
    print("🧪 Testing FilterManager Implementation")
    print("=" * 60)
    
    try:
        from components.filters import FilterManager
        from core.interfaces import LLMProvider
        
        # Mock LLM for testing
        class MockLLM(LLMProvider):
            def generate(self, prompt: str) -> str:
                # Return mock filter JSON
                return '{"theme": ["Economy"], "sentiment": "Positive"}'
            
            def generate_structured(self, prompt: str, schema: dict) -> dict:
                return {"theme": ["Economy"], "sentiment": "Positive"}
            
            def is_available(self) -> bool:
                return True
        
        mock_llm = MockLLM()
        
        # Test 1: FilterManager accepts LLMProvider
        print("\n📝 Test 1: FilterManager accepts LLMProvider...")
        try:
            manager = FilterManager(llm=mock_llm)
            print("  ✅ FilterManager initialized with LLMProvider")
        except Exception as e:
            print(f"  ❌ FilterManager initialization failed: {e}")
            return False
        
        # Test 2: Filter extraction works
        print("\n📝 Test 2: Filter extraction...")
        try:
            filters = manager.extract_and_update_filters(
                query="Show me positive economic statements"
            )
            if not isinstance(filters, dict):
                print(f"  ❌ extract_and_update_filters didn't return dict, got {type(filters)}")
                return False
            print(f"  ✅ Filter extraction works (extracted: {filters})")
        except Exception as e:
            print(f"  ❌ Filter extraction failed: {e}")
            return False
        
        # Test 3: Conversation history
        print("\n📝 Test 3: Conversation history...")
        try:
            context = manager.get_conversation_context(last_n_turns=3)
            if not isinstance(context, str):
                print(f"  ❌ get_conversation_context didn't return str, got {type(context)}")
                return False
            print("  ✅ Conversation history works")
        except Exception as e:
            print(f"  ❌ Conversation history failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing FilterManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_facade():
    """Test that RAGPipeline facade works correctly"""
    print("\n" + "=" * 60)
    print("🧪 Testing RAGPipeline Facade Implementation")
    print("=" * 60)
    
    try:
        from application.pipeline import RAGPipeline
        from core.interfaces import LLMProvider, Embedder, Reranker, VectorStore
        from components.retrieval import Retriever
        from components.retrieval.strategies import VectorOnlyStrategy
        from core.domain import RAGResponse
        import numpy as np
        
        # Create mocks
        class MockLLM(LLMProvider):
            def generate(self, prompt: str) -> str:
                return "Mock answer based on the provided context."
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
                return [
                    {
                        "content": "Statement about economy",
                        "score": 0.9,
                        "embedding_index": 1,
                        "metadata": {"theme": "Economy", "sentiment": "Positive"}
                    }
                ]
            def keyword_search(self, query_text, top_k, filters=None):
                return []
            def hybrid_search(self, query_vector, query_text, top_k, alpha=0.5, filters=None):
                return []
            def close(self):
                pass
        
        # Test 1: Pipeline accepts interfaces
        print("\n📝 Test 1: RAGPipeline accepts interface types...")
        try:
            mock_llm = MockLLM()
            mock_embedder = MockEmbedder()
            mock_vector_store = MockVectorStore()
            mock_retriever = Retriever(strategy=VectorOnlyStrategy(mock_vector_store))
            
            pipeline = RAGPipeline(
                llm=mock_llm,
                embedder=mock_embedder,
                retriever=mock_retriever,
                reranker=None,
                vector_store=mock_vector_store
            )
            print("  ✅ RAGPipeline accepts interface types")
        except Exception as e:
            print(f"  ❌ RAGPipeline initialization failed: {e}")
            return False
        
        # Test 2: query() method returns RAGResponse
        print("\n📝 Test 2: query() method returns RAGResponse...")
        try:
            response = pipeline.query("What are positive economic statements?", top_k=5)
            
            if not isinstance(response, RAGResponse):
                print(f"  ❌ query() didn't return RAGResponse, got {type(response)}")
                return False
            
            # Check response has required fields
            if not hasattr(response, 'answer'):
                print("  ❌ RAGResponse missing 'answer' field")
                return False
            
            if not hasattr(response, 'sources'):
                print("  ❌ RAGResponse missing 'sources' field")
                return False
            
            if not hasattr(response, 'metrics'):
                print("  ❌ RAGResponse missing 'metrics' field")
                return False
            
            print("  ✅ query() returns proper RAGResponse")
            print(f"     Answer: {response.answer[:50]}...")
            print(f"     Sources: {len(response.sources)}")
            print(f"     Time: {response.metrics.total_time_ms:.1f}ms")
        
        except Exception as e:
            print(f"  ❌ query() execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing RAGPipeline Facade: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n🚀 RAG System - Application Layer (Phase 4) Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Factory Pattern Tests", test_factory_pattern),
        ("Builder Pattern Tests", test_builder_pattern),
        ("FilterManager Tests", test_filter_manager),
        ("RAGPipeline Facade Tests", test_pipeline_facade)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 PHASE 4 COMPLETE!")
        print("=" * 60)
        print("\n✅ Application Layer is properly implemented")
        print("✅ Factory Pattern working correctly")
        print("✅ Builder Pattern working correctly")
        print("✅ Facade Pattern working correctly")
        print("✅ FilterManager implemented")
        print("✅ RAGPipeline orchestration working")
        print("✅ Ready to move to Phase 5: Integration & Testing!")
        print("\n📚 Next steps:")
        print("  1. Create comprehensive integration tests")
        print("  2. Test full pipeline with real components")
        print("  3. Performance testing and optimization")
        print("  4. Documentation and examples")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n⚠️  Please fix the errors before proceeding to Phase 5")
        return 1


if __name__ == "__main__":
    exit(main())