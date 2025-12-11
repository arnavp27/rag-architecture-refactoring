#!/usr/bin/env python3
"""
Components Layer (Phase 3) Validation Script

This script validates that the Components Layer (strategies and decorators)
is properly implemented following the Strategy and Decorator patterns.

Run this after implementing Phase 3 to ensure everything is working.

Usage:
    python validate-phase3.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all component modules can be imported"""
    print("=" * 60)
    print("🧪 Testing Components Layer Imports")
    print("=" * 60)
    
    try:
        # Test strategy imports
        print("\n📦 Testing Strategy Imports...")
        from components.retrieval.strategies import (
            VectorOnlyStrategy,
            KeywordOnlyStrategy,
            HybridStrategy
        )
        print("✅ All strategy imports successful")
        
        # Test decorator imports
        print("\n📦 Testing Decorator Imports...")
        from components.retrieval.decorators import (
            CachingRetriever,
            TimingRetriever
        )
        print("✅ All decorator imports successful")
        
        # Test context class import
        print("\n📦 Testing Context Class Import...")
        from components.retrieval import Retriever
        print("✅ Retriever (context class) import successful")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Make sure all files are created in the correct locations")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False


def test_strategy_pattern():
    """Test that Strategy Pattern is properly implemented"""
    print("\n" + "=" * 60)
    print("🧪 Testing Strategy Pattern Implementation")
    print("=" * 60)
    
    try:
        from core.interfaces import RetrievalStrategy
        from components.retrieval.strategies import (
            VectorOnlyStrategy,
            KeywordOnlyStrategy,
            HybridStrategy
        )
        from components.retrieval import Retriever
        
        from core.interfaces import VectorStore
        
        # Mock vector store for testing
        class MockVectorStore(VectorStore):
            def vector_search(self, query_vector, top_k, filters=None):
                return [
                    {"content": "doc1", "score": 0.9, "embedding_index": 1},
                    {"content": "doc2", "score": 0.8, "embedding_index": 2}
                ]
            
            def keyword_search(self, query_text, top_k, filters=None):
                return [
                    {"content": "doc3", "score": 0.85, "embedding_index": 3},
                    {"content": "doc4", "score": 0.75, "embedding_index": 4}
                ]
            
            def hybrid_search(self, query_vector, query_text, top_k, alpha=0.5, filters=None):
                return [
                    {"content": "doc1", "score": 0.88, "embedding_index": 1}
                ]
            
            def close(self):
                pass
        
        mock_store = MockVectorStore()
        
        # Test 1: All strategies implement RetrievalStrategy
        print("\n📝 Test 1: Checking interface implementation...")
        strategies = [
            VectorOnlyStrategy(mock_store),
            KeywordOnlyStrategy(mock_store),
            HybridStrategy(mock_store)
        ]
        
        for strategy in strategies:
            if not isinstance(strategy, RetrievalStrategy):
                print(f"  ❌ {strategy.__class__.__name__} doesn't implement RetrievalStrategy")
                return False
        print("  ✅ All strategies implement RetrievalStrategy interface")
        
        # Test 2: Strategies are interchangeable
        print("\n📝 Test 2: Testing strategy interchangeability...")
        query_vec = np.random.rand(384)  # Mock embedding
        query_text = "test query"
        
        for strategy in strategies:
            try:
                results = strategy.retrieve(query_vec, query_text, top_k=5)
                if not isinstance(results, list):
                    print(f"  ❌ {strategy.__class__.__name__} didn't return a list")
                    return False
            except Exception as e:
                print(f"  ❌ {strategy.__class__.__name__} failed: {e}")
                return False
        print("  ✅ All strategies have compatible retrieve() signatures")
        
        # Test 3: Context class (Retriever) works
        print("\n📝 Test 3: Testing Retriever (context class)...")
        retriever = Retriever(strategy=VectorOnlyStrategy(mock_store))
        
        # Test initial strategy
        results = retriever.retrieve(query_vec, query_text, top_k=5)
        if not isinstance(results, list):
            print("  ❌ Retriever didn't return results correctly")
            return False
        
        # Test strategy switching
        retriever.set_strategy(HybridStrategy(mock_store))
        results = retriever.retrieve(query_vec, query_text, top_k=5)
        if not isinstance(results, list):
            print("  ❌ Retriever failed after strategy switch")
            return False
        
        print("  ✅ Retriever (context class) works correctly")
        print("  ✅ Strategy switching at runtime works")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Strategy Pattern: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decorator_pattern():
    """Test that Decorator Pattern is properly implemented"""
    print("\n" + "=" * 60)
    print("🧪 Testing Decorator Pattern Implementation")
    print("=" * 60)
    
    try:
        from core.interfaces import RetrievalStrategy, VectorStore
        from components.retrieval.strategies import VectorOnlyStrategy
        from components.retrieval.decorators import (
            CachingRetriever,
            TimingRetriever
        )
        
        # Mock vector store
        class MockVectorStore(VectorStore):
            def __init__(self):
                self.call_count = 0
            
            def vector_search(self, query_vector, top_k, filters=None):
                self.call_count += 1
                return [
                    {"content": f"doc{self.call_count}", "score": 0.9, "embedding_index": self.call_count}
                ]
            
            def keyword_search(self, query_text, top_k, filters=None):
                return []
            
            def hybrid_search(self, query_vector, query_text, top_k, alpha=0.5, filters=None):
                return []
            
            def close(self):
                pass
        
        mock_store = MockVectorStore()
        
        # Test 1: Decorators implement RetrievalStrategy
        print("\n📝 Test 1: Checking decorator interface implementation...")
        base_strategy = VectorOnlyStrategy(mock_store)
        
        decorators = [
            CachingRetriever(base_strategy, cache_size=10),
            TimingRetriever(base_strategy, log_threshold_ms=1000)
        ]
        
        for decorator in decorators:
            if not isinstance(decorator, RetrievalStrategy):
                print(f"  ❌ {decorator.__class__.__name__} doesn't implement RetrievalStrategy")
                return False
        print("  ✅ All decorators implement RetrievalStrategy interface")
        
        # Test 2: Decorators can wrap strategies
        print("\n📝 Test 2: Testing decorator wrapping...")
        query_vec = np.random.rand(384)
        query_text = "test query"
        
        wrapped = CachingRetriever(base_strategy, cache_size=10)
        results = wrapped.retrieve(query_vec, query_text, top_k=5)
        
        if not isinstance(results, list):
            print("  ❌ Decorator didn't return results correctly")
            return False
        print("  ✅ Decorators can wrap strategies")
        
        # Test 3: Caching works
        print("\n📝 Test 3: Testing caching functionality...")
        mock_store_2 = MockVectorStore()
        base_strategy_2 = VectorOnlyStrategy(mock_store_2)
        cached_strategy = CachingRetriever(base_strategy_2, cache_size=10)
        
        # First call - should hit database
        cached_strategy.retrieve(query_vec, "query1", top_k=5)
        first_call_count = mock_store_2.call_count
        
        # Second call with same query - should use cache
        cached_strategy.retrieve(query_vec, "query1", top_k=5)
        second_call_count = mock_store_2.call_count
        
        if second_call_count != first_call_count:
            print(f"  ❌ Cache not working (calls: {first_call_count} -> {second_call_count})")
            return False
        print("  ✅ Caching works correctly")
        
        # Test 4: Decorators can be stacked
        print("\n📝 Test 4: Testing decorator stacking...")
        base = VectorOnlyStrategy(mock_store)
        cached = CachingRetriever(base, cache_size=10)
        timed = TimingRetriever(cached, log_threshold_ms=1000)
        
        results = timed.retrieve(query_vec, query_text, top_k=5)
        if not isinstance(results, list):
            print("  ❌ Stacked decorators failed")
            return False
        print("  ✅ Decorators can be stacked")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Decorator Pattern: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rrf_fusion():
    """Test that RRF fusion algorithm works correctly"""
    print("\n" + "=" * 60)
    print("🧪 Testing RRF Fusion Algorithm")
    print("=" * 60)
    
    try:
        from components.retrieval.strategies import HybridStrategy
        from core.interfaces import VectorStore
        
        # Mock vector store with known results
        class MockVectorStore(VectorStore):
            def vector_search(self, query_vector, top_k, filters=None):
                return [
                    {"content": "doc1", "score": 0.9, "embedding_index": 1},
                    {"content": "doc2", "score": 0.8, "embedding_index": 2},
                    {"content": "doc3", "score": 0.7, "embedding_index": 3}
                ]
            
            def keyword_search(self, query_text, top_k, filters=None):
                return [
                    {"content": "doc2", "score": 0.95, "embedding_index": 2},
                    {"content": "doc4", "score": 0.85, "embedding_index": 4},
                    {"content": "doc1", "score": 0.75, "embedding_index": 1}
                ]
            
            def hybrid_search(self, query_vector, query_text, top_k, alpha=0.5, filters=None):
                return []
            
            def close(self):
                pass
        
        mock_store = MockVectorStore()
        hybrid = HybridStrategy(mock_store, rrf_k=60)
        
        print("\n📝 Testing RRF fusion...")
        query_vec = np.random.rand(384)
        results = hybrid.retrieve(query_vec, "test query", top_k=5)
        
        # Check that results have rrf_score
        if not all("rrf_score" in r for r in results):
            print("  ❌ RRF scores not added to results")
            return False
        
        # Check that results are sorted by rrf_score
        rrf_scores = [r["rrf_score"] for r in results]
        if rrf_scores != sorted(rrf_scores, reverse=True):
            print("  ❌ Results not sorted by RRF score")
            return False
        
        print("  ✅ RRF fusion works correctly")
        print("  ✅ Results contain rrf_score field")
        print("  ✅ Results sorted by RRF score (descending)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing RRF fusion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n🚀 RAG System - Components Layer (Phase 3) Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Strategy Pattern Tests", test_strategy_pattern),
        ("Decorator Pattern Tests", test_decorator_pattern),
        ("RRF Fusion Tests", test_rrf_fusion)
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
        print("🎉 PHASE 3 COMPLETE!")
        print("=" * 60)
        print("\n✅ Components Layer is properly implemented")
        print("✅ Strategy Pattern working correctly")
        print("✅ Decorator Pattern working correctly")
        print("✅ RRF fusion algorithm implemented")
        print("✅ Ready to move to Phase 4: Application Layer (Factories & Builder)")
        print("\n📚 Next steps:")
        print("  1. Implement application/factories/llm_factory.py")
        print("  2. Implement application/factories/model_factory.py")
        print("  3. Implement application/builders/pipeline_builder.py")
        print("  4. Implement application/pipeline/rag_pipeline.py (Facade)")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n⚠️  Please fix the errors before proceeding to Phase 4")
        return 1


if __name__ == "__main__":
    exit(main())