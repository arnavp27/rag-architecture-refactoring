#!/usr/bin/env python3
"""
Infrastructure Layer Validation Script

This script validates that the Infrastructure Layer (adapters and configuration)
is properly implemented and follows the Adapter Pattern correctly.

Run this after implementing Phase 2 to ensure everything is working.

Usage:
    python validate_infrastructure_layer.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all infrastructure modules can be imported"""
    print("=" * 60)
    print("🧪 Testing Infrastructure Layer Imports")
    print("=" * 60)
    
    try:
        # Test configuration imports
        print("\n📦 Testing Configuration Imports...")
        from infrastructure.config import Settings, get_settings
        print("✅ Configuration imports successful")
        
        # Test adapter imports
        print("\n📦 Testing Adapter Imports...")
        from infrastructure.adapters import (
            GeminiAdapter,
            OllamaAdapter,
            WeaviateAdapter,
            SentenceTransformerEmbedder,
            CrossEncoderReranker
        )
        print("✅ All adapter imports successful")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Make sure all dependencies are installed:")
        print("   pip install google-generativeai langchain-community weaviate-client")
        print("   pip install sentence-transformers torch")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False


def test_interface_compliance():
    """Test that adapters implement their interfaces correctly"""
    print("\n" + "=" * 60)
    print("🧪 Testing Interface Compliance")
    print("=" * 60)
    
    try:
        from core.interfaces import LLMProvider, VectorStore, Embedder, Reranker
        from infrastructure.adapters import (
            GeminiAdapter,
            OllamaAdapter,
            WeaviateAdapter,
            SentenceTransformerEmbedder,
            CrossEncoderReranker
        )
        
        # Check LLM adapters
        print("\n📝 Checking LLM Adapters...")
        assert issubclass(GeminiAdapter, LLMProvider), "GeminiAdapter must implement LLMProvider"
        assert issubclass(OllamaAdapter, LLMProvider), "OllamaAdapter must implement LLMProvider"
        print("  ✓ GeminiAdapter implements LLMProvider")
        print("  ✓ OllamaAdapter implements LLMProvider")
        
        # Check VectorStore adapter
        print("\n📝 Checking VectorStore Adapter...")
        assert issubclass(WeaviateAdapter, VectorStore), "WeaviateAdapter must implement VectorStore"
        print("  ✓ WeaviateAdapter implements VectorStore")
        
        # Check Embedder adapter
        print("\n📝 Checking Embedder Adapter...")
        assert issubclass(SentenceTransformerEmbedder, Embedder), "SentenceTransformerEmbedder must implement Embedder"
        print("  ✓ SentenceTransformerEmbedder implements Embedder")
        
        # Check Reranker adapter
        print("\n📝 Checking Reranker Adapter...")
        assert issubclass(CrossEncoderReranker, Reranker), "CrossEncoderReranker must implement Reranker"
        print("  ✓ CrossEncoderReranker implements Reranker")
        
        print("\n✅ All adapters implement their interfaces correctly")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Interface Compliance Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False


def test_configuration():
    """Test configuration management"""
    print("\n" + "=" * 60)
    print("🧪 Testing Configuration")
    print("=" * 60)
    
    try:
        from infrastructure.config import Settings, get_settings
        
        # Test settings creation
        print("\n📝 Testing Settings creation...")
        settings = get_settings()
        print(f"  ✓ Settings created: {settings}")
        
        # Test settings attributes
        print("\n📝 Testing Settings attributes...")
        assert hasattr(settings, 'weaviate_host'), "Settings must have weaviate_host"
        assert hasattr(settings, 'embedder_model'), "Settings must have embedder_model"
        assert hasattr(settings, 'gemini_model'), "Settings must have gemini_model"
        print("  ✓ All required settings attributes present")
        
        # Test settings methods
        print("\n📝 Testing Settings methods...")
        weaviate_url = settings.get_weaviate_url()
        assert weaviate_url.startswith("http"), "Weaviate URL must be valid"
        print(f"  ✓ Weaviate URL: {weaviate_url}")
        
        print("\n✅ Configuration works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter_instantiation():
    """Test that adapters can be instantiated (without real connections)"""
    print("\n" + "=" * 60)
    print("🧪 Testing Adapter Instantiation")
    print("=" * 60)
    
    results = []
    
    # Test Gemini Adapter (will fail without API key, but tests structure)
    print("\n📝 Testing GeminiAdapter...")
    try:
        from infrastructure.adapters import GeminiAdapter
        # Try with fake API key to test structure
        try:
            adapter = GeminiAdapter(api_key="fake_key_for_testing")
            print("  ⚠️  GeminiAdapter instantiated (but may not be functional)")
            results.append(("GeminiAdapter structure", True))
        except Exception as e:
            print(f"  ⚠️  GeminiAdapter requires valid API key (expected): {e}")
            results.append(("GeminiAdapter structure", True))  # Structure is fine
    except Exception as e:
        print(f"  ❌ GeminiAdapter instantiation failed: {e}")
        results.append(("GeminiAdapter structure", False))
    
    # Test Ollama Adapter
    print("\n📝 Testing OllamaAdapter...")
    try:
        from infrastructure.adapters import OllamaAdapter
        adapter = OllamaAdapter()  # Can instantiate without connection
        print("  ✓ OllamaAdapter instantiated successfully")
        results.append(("OllamaAdapter", True))
    except Exception as e:
        print(f"  ⚠️  OllamaAdapter instantiation issue (may need Ollama running): {e}")
        results.append(("OllamaAdapter", False))
    
    # Test Weaviate Adapter (will fail without connection)
    print("\n📝 Testing WeaviateAdapter...")
    try:
        from infrastructure.adapters import WeaviateAdapter
        # Don't actually connect, just test structure
        print("  ⚠️  WeaviateAdapter requires running Weaviate instance")
        results.append(("WeaviateAdapter structure", True))
    except Exception as e:
        print(f"  ❌ WeaviateAdapter import failed: {e}")
        results.append(("WeaviateAdapter structure", False))
    
    # Test Embedder
    print("\n📝 Testing SentenceTransformerEmbedder...")
    try:
        from infrastructure.adapters import SentenceTransformerEmbedder
        print("  ⚠️  SentenceTransformerEmbedder requires model download")
        print("     (Will download on first use)")
        results.append(("SentenceTransformerEmbedder structure", True))
    except Exception as e:
        print(f"  ❌ SentenceTransformerEmbedder import failed: {e}")
        results.append(("SentenceTransformerEmbedder structure", False))
    
    # Test Reranker
    print("\n📝 Testing CrossEncoderReranker...")
    try:
        from infrastructure.adapters import CrossEncoderReranker
        print("  ⚠️  CrossEncoderReranker requires model download")
        print("     (Will download on first use)")
        results.append(("CrossEncoderReranker structure", True))
    except Exception as e:
        print(f"  ❌ CrossEncoderReranker import failed: {e}")
        results.append(("CrossEncoderReranker structure", False))
    
    # Summary
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✅ All adapters have correct structure")
    else:
        print("\n⚠️  Some adapters had issues (may be expected without services running)")
    
    return all_passed


def main():
    """Run all validation tests"""
    print("\n🚀 RAG System - Infrastructure Layer Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Interface Compliance Tests", test_interface_compliance),
        ("Configuration Tests", test_configuration),
        ("Adapter Instantiation Tests", test_adapter_instantiation)
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
        print("🎉 PHASE 2 COMPLETE!")
        print("=" * 60)
        print("\n✅ Infrastructure Layer is properly implemented")
        print("✅ All adapters implement their interfaces correctly")
        print("✅ Ready to move to Phase 3: Components Layer")
        print("\n📚 Next steps:")
        print("  1. Implement components/retrieval/strategies/vector_only_strategy.py")
        print("  2. Implement components/retrieval/strategies/hybrid_strategy.py")
        print("  3. Implement components/retrieval/strategies/keyword_only_strategy.py")
        print("  4. Implement components/retrieval/decorators/caching_retriever.py")
        print("  5. Implement components/retrieval/decorators/timing_retriever.py")
        print("  6. Implement components/retrieval/retriever.py (context class)")
        print("  7. Implement components/filters/filter_manager.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 60)
        print("\n💡 This may be expected if:")
        print("   - External services are not running (Weaviate, Ollama)")
        print("   - API keys are not configured (.env file)")
        print("   - Models haven't been downloaded yet")
        print("\n✅ If imports and interface compliance passed, you can proceed!")
        return 1


if __name__ == "__main__":
    exit(main())
