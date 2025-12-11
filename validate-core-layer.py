#!/usr/bin/env python3
"""
Core Layer Validation Script

This script validates that the Core Layer (interfaces and domain models)
is properly implemented and can be imported without errors.

Run this after implementing Phase 1 to ensure everything is working.

Usage:
    python validate_core_layer.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all core modules can be imported"""
    print("=" * 60)
    print("🧪 Testing Core Layer Imports")
    print("=" * 60)
    
    try:
        # Test interface imports
        print("\n📦 Testing Interface Imports...")
        from core.interfaces import (
            LLMProvider,
            VectorStore,
            Embedder,
            Reranker,
            RetrievalStrategy
        )
        print("✅ All interfaces imported successfully")
        
        # Test domain model imports
        print("\n📦 Testing Domain Model Imports...")
        from core.domain import (
            Query,
            Document,
            RAGResponse,
            PerformanceMetrics
        )
        print("✅ All domain models imported successfully")
        
        # Test that interfaces are abstract
        print("\n🔍 Validating Interfaces are Abstract...")
        from abc import ABC
        
        interfaces = [LLMProvider, VectorStore, Embedder, Reranker, RetrievalStrategy]
        for interface in interfaces:
            assert issubclass(interface, ABC), f"{interface.__name__} is not abstract"
            print(f"  ✓ {interface.__name__} is properly abstract")
        
        print("\n✅ All interfaces are properly abstract")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        return False
    except AssertionError as e:
        print(f"\n❌ Validation Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False


def test_domain_models():
    """Test that domain models work correctly"""
    print("\n" + "=" * 60)
    print("🧪 Testing Domain Models")
    print("=" * 60)
    
    try:
        import numpy as np
        from core.domain import Query, Document, RAGResponse, PerformanceMetrics
        
        # Test Query creation
        print("\n📝 Testing Query model...")
        query = Query(
            text="What is economic growth?",
            filters={"theme": ["Economy"]},
            top_k=5
        )
        print(f"  ✓ Query created: {query}")
        
        # Test Document creation
        print("\n📝 Testing Document model...")
        doc = Document(
            content="Economic growth refers to increase in GDP",
            metadata={"theme": "Economy", "sentiment": "Neutral"},
            score=0.85
        )
        print(f"  ✓ Document created: {doc}")
        
        # Test PerformanceMetrics creation
        print("\n📝 Testing PerformanceMetrics model...")
        metrics = PerformanceMetrics(
            total_time_ms=1500.0,
            embedding_time_ms=200.0,
            retrieval_time_ms=500.0,
            generation_time_ms=800.0
        )
        print(f"  ✓ PerformanceMetrics created: {metrics}")
        print(f"  ✓ Bottleneck: {metrics.get_bottleneck()}")
        
        # Test RAGResponse creation
        print("\n📝 Testing RAGResponse model...")
        response = RAGResponse(
            answer="Economic growth is the increase in GDP over time.",
            sources=[doc],
            filters_applied={"theme": ["Economy"]},
            metrics=metrics
        )
        print(f"  ✓ RAGResponse created: {response}")
        
        # Test serialization
        print("\n📝 Testing serialization...")
        response_dict = response.to_dict()
        assert "answer" in response_dict
        assert "sources" in response_dict
        assert "metrics" in response_dict
        print("  ✓ to_dict() works correctly")
        
        print("\n✅ All domain models work correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing domain models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test that domain models validate their data"""
    print("\n" + "=" * 60)
    print("🧪 Testing Data Validation")
    print("=" * 60)
    
    try:
        from core.domain import Query, Document
        
        # Test Query validation
        print("\n📝 Testing Query validation...")
        try:
            Query(text="")  # Empty text should fail
            print("  ❌ Empty text should have raised ValueError")
            return False
        except ValueError:
            print("  ✓ Empty query text correctly rejected")
        
        try:
            Query(text="test", top_k=0)  # Invalid top_k should fail
            print("  ❌ Invalid top_k should have raised ValueError")
            return False
        except ValueError:
            print("  ✓ Invalid top_k correctly rejected")
        
        # Test Document validation
        print("\n📝 Testing Document validation...")
        try:
            Document(content="")  # Empty content should fail
            print("  ❌ Empty content should have raised ValueError")
            return False
        except ValueError:
            print("  ✓ Empty document content correctly rejected")
        
        print("\n✅ Data validation works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n🚀 RAG System - Core Layer Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Domain Model Tests", test_domain_models),
        ("Validation Tests", test_validation)
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
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n✅ Core Layer is properly implemented")
        print("✅ Ready to move to Phase 2: Infrastructure Layer (Adapters)")
        print("\n📚 Next steps:")
        print("  1. Implement infrastructure/adapters/gemini_adapter.py")
        print("  2. Implement infrastructure/adapters/ollama_adapter.py")
        print("  3. Implement infrastructure/adapters/weaviate_adapter.py")
        print("  4. Implement infrastructure/adapters/sentence_transformer_embedder.py")
        print("  5. Implement infrastructure/adapters/cross_encoder_reranker.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n⚠️  Please fix the errors before proceeding")
        return 1


if __name__ == "__main__":
    exit(main())
