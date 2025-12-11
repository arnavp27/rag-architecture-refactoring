#!/usr/bin/env python3
"""
Main entry point for the RAG system

This is the primary way to run the RAG system. It provides:
1. Interactive query mode
2. Single query mode
3. Batch query mode

Usage:
    # Interactive mode
    python main.py
    
    # Single query
    python main.py --query "What are positive economic policies?"
    
    # From file
    python main.py --file queries.txt
"""

import sys
import argparse
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


def build_pipeline():
    """
    Build the RAG pipeline with default configuration.
    
    Returns:
        RAGPipeline: Configured pipeline ready to use
    """
    from application.builders import RAGPipelineBuilder
    
    print("🔧 Building RAG Pipeline...")
    print("   (This may take a moment to load models)")
    
    try:
        pipeline = (RAGPipelineBuilder()
            .with_llm(primary="ollama", fallback="gemini")
            .with_embedder("sentence-transformers")
            .with_vector_store()
            .with_retrieval_strategy("hybrid")
            .with_caching(cache_size=100)
            .with_timing(threshold_ms=1000)
            .with_reranker("cross-encoder")
            .build())
        
        print("✅ Pipeline built successfully!\n")
        return pipeline
        
    except Exception as e:
        print(f"❌ Failed to build pipeline: {e}")
        print("\n💡 Tip: Make sure services are running:")
        print("   - Weaviate: localhost:8080")
        print("   - Ollama: localhost:11434")
        print("   - Or set GOOGLE_API_KEY in .env for Gemini")
        print("\n   Alternatively, run demo.py which uses mocks")
        sys.exit(1)


def process_query(pipeline, query: str, verbose: bool = True):
    """
    Process a single query through the pipeline.
    
    Args:
        pipeline: RAGPipeline instance
        query: Query text
        verbose: Whether to print detailed output
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"💬 Query: {query}")
        print('='*70)
    
    try:
        response = pipeline.query(query, top_k=5)
        
        if verbose:
            print(f"\n📝 Answer:\n{response.answer}\n")
            
            print(f"📚 Sources ({len(response.sources)}):")
            for i, doc in enumerate(response.sources, 1):
                print(f"\n  [{i}] Score: {doc.score:.3f}")
                print(f"      {doc.content[:100]}...")
                if doc.rerank_score:
                    print(f"      Rerank: {doc.rerank_score:.3f}")
            
            if response.filters_applied:
                print(f"\n🔍 Filters: {response.filters_applied}")
            
            print(f"\n⏱️  Performance:")
            print(f"      Total: {response.metrics.total_time_ms:.1f}ms")
            print(f"      - Filters: {response.metrics.filter_extraction_time_ms:.1f}ms")
            print(f"      - Embedding: {response.metrics.embedding_time_ms:.1f}ms")
            print(f"      - Retrieval: {response.metrics.retrieval_time_ms:.1f}ms")
            print(f"      - Reranking: {response.metrics.rerank_time_ms:.1f}ms")
            print(f"      - Generation: {response.metrics.generation_time_ms:.1f}ms")
        else:
            print(f"\nQ: {query}")
            print(f"A: {response.answer}")
        
        return response
        
    except Exception as e:
        print(f"\n❌ Error processing query: {e}")
        logger.error(f"Query failed: {e}", exc_info=True)
        return None


def interactive_mode(pipeline):
    """
    Run interactive query mode.
    
    Args:
        pipeline: RAGPipeline instance
    """
    print("\n" + "="*70)
    print("🤖 RAG System - Interactive Mode")
    print("="*70)
    print("\nType your queries below. Commands:")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'clear' to clear screen")
    print("  - 'stats' to show pipeline stats")
    print("  - 'switch [strategy]' to change retrieval strategy")
    print("\nStrategies: vector, keyword, hybrid")
    print("="*70)
    
    while True:
        try:
            query = input("\n💬 Query: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif query.lower() == 'clear':
                print("\n" * 100)
                continue
            
            elif query.lower() == 'stats':
                retriever = pipeline.get_retriever()
                strategy = retriever.get_current_strategy()
                print(f"\n📊 Current Strategy: {strategy.get_strategy_name()}")
                print(f"   Info: {strategy.get_strategy_info()}")
                continue
            
            elif query.lower().startswith('switch'):
                parts = query.split()
                if len(parts) != 2:
                    print("Usage: switch [vector|keyword|hybrid]")
                    continue
                
                strategy_name = parts[1].lower()
                try:
                    from components.retrieval.strategies import (
                        VectorOnlyStrategy, KeywordOnlyStrategy, HybridStrategy
                    )
                    
                    retriever = pipeline.get_retriever()
                    vector_store = pipeline._vector_store
                    
                    if strategy_name == 'vector':
                        retriever.set_strategy(VectorOnlyStrategy(vector_store))
                    elif strategy_name == 'keyword':
                        retriever.set_strategy(KeywordOnlyStrategy(vector_store))
                    elif strategy_name == 'hybrid':
                        retriever.set_strategy(HybridStrategy(vector_store))
                    else:
                        print(f"Unknown strategy: {strategy_name}")
                        continue
                    
                    print(f"✅ Switched to {strategy_name} strategy")
                except Exception as e:
                    print(f"❌ Failed to switch strategy: {e}")
                continue
            
            # Process query
            process_query(pipeline, query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Interactive mode error: {e}", exc_info=True)


def single_query_mode(pipeline, query: str):
    """
    Process a single query and exit.
    
    Args:
        pipeline: RAGPipeline instance
        query: Query text
    """
    print("\n" + "="*70)
    print("🤖 RAG System - Single Query Mode")
    print("="*70)
    
    process_query(pipeline, query)


def batch_mode(pipeline, queries_file: str):
    """
    Process queries from a file.
    
    Args:
        pipeline: RAGPipeline instance
        queries_file: Path to file with queries (one per line)
    """
    print("\n" + "="*70)
    print("🤖 RAG System - Batch Mode")
    print("="*70)
    
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"\n📄 Processing {len(queries)} queries from {queries_file}\n")
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}]")
            response = process_query(pipeline, query, verbose=False)
            if response:
                results.append({
                    'query': query,
                    'answer': response.answer,
                    'num_sources': len(response.sources),
                    'time_ms': response.metrics.total_time_ms
                })
        
        # Summary
        print("\n" + "="*70)
        print("📊 Batch Summary")
        print("="*70)
        print(f"Total queries: {len(queries)}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(queries) - len(results)}")
        
        if results:
            avg_time = sum(r['time_ms'] for r in results) / len(results)
            print(f"Average time: {avg_time:.1f}ms")
        
    except FileNotFoundError:
        print(f"❌ File not found: {queries_file}")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        logger.error(f"Batch mode error: {e}", exc_info=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RAG System - Refactored Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Single query
  python main.py --query "What are positive economic policies?"
  
  # Batch mode
  python main.py --file queries.txt
  
  # Quiet mode
  python main.py --query "test" --quiet
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File with queries (one per line)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Build pipeline
    pipeline = build_pipeline()
    
    try:
        # Determine mode
        if args.query:
            single_query_mode(pipeline, args.query)
        elif args.file:
            batch_mode(pipeline, args.file)
        else:
            interactive_mode(pipeline)
    
    finally:
        # Cleanup
        try:
            pipeline.close()
        except:
            pass


if __name__ == "__main__":
    main()