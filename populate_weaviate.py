#!/usr/bin/env python3
"""
Populate Weaviate with existing embeddings and articles
Migrates data from .npy/.json files to Weaviate vector database
"""

import sys
import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_files(embeddings_path, embedding_map_path, articles_path):
    """Load all data files"""
    logger.info(f"Loading embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded {embeddings.shape[0]} embeddings, dimension={embeddings.shape[1]}")
    
    logger.info(f"Loading embedding map from: {embedding_map_path}")
    with open(embedding_map_path, 'r', encoding='utf-8') as f:
        embedding_map = json.load(f)
    logger.info(f"Loaded mapping for {len(embedding_map)} articles")
    
    logger.info(f"Loading articles from: {articles_path}")
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles")
    
    return embeddings, embedding_map, articles


def create_collection(client):
    """Create or recreate Weaviate collection"""
    collection_name = "PoliticalStatement"
    
    # Delete existing collection if it exists
    try:
        client.collections.delete(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection with proper schema
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="article_id", data_type=DataType.TEXT),
            Property(name="article_title", data_type=DataType.TEXT),
            Property(name="statement_index", data_type=DataType.INT),
            Property(name="sentiment", data_type=DataType.TEXT),
            Property(name="theme", data_type=DataType.TEXT_ARRAY),
            Property(name="classification", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="politician", data_type=DataType.TEXT),
            Property(name="date", data_type=DataType.TEXT),
        ]
    )
    
    logger.info(f"Created collection: {collection_name}")
    return client.collections.get(collection_name)


def populate_weaviate(collection, embeddings, embedding_map, articles):
    """Populate Weaviate with all statements"""
    
    # Create article lookup
    article_lookup = {article['article_id']: article for article in articles}
    
    total_statements = sum(len(info['statement_indices']) for info in embedding_map.values())
    logger.info(f"Populating {total_statements} statements into Weaviate...")
    
    # Batch insert
    with collection.batch.dynamic() as batch:
        for article_id, info in tqdm(embedding_map.items(), desc="Processing articles"):
            article = article_lookup.get(article_id)
            if not article:
                logger.warning(f"Article not found: {article_id}")
                continue
            
            statements = article.get('statements', [])
            statement_indices = info['statement_indices']
            
            for stmt_idx in statement_indices:
                if stmt_idx >= len(statements):
                    continue
                
                statement = statements[stmt_idx]
                
                # Get embedding vector
                embedding_idx = statement.get('embedding_index')
                if embedding_idx is None or embedding_idx >= len(embeddings):
                    continue
                
                vector = embeddings[embedding_idx].tolist()
                
                # Prepare properties
                properties = {
                    "content": statement.get('statement_text', ''),
                    "article_id": article_id,
                    "article_title": article.get('title', ''),
                    "statement_index": stmt_idx,
                    "sentiment": statement.get('sentiment', 'Unknown'),
                    "theme": statement.get('theme', []),
                    "classification": statement.get('classification', 'Unknown'),
                    "source": article.get('source', ''),
                    "politician": article.get('politician', ''),
                    "date": article.get('date', ''),
                }
                
                # Add to batch
                batch.add_object(
                    properties=properties,
                    vector=vector
                )
    
    logger.info("✅ Weaviate population completed!")


def main():
    """Main population function"""
    
    # File paths from your .env
    embeddings_path = Path(r"C:\db\Gemini_Generated_Json\mk_stalin\mk_stalin_embeddings.npy")
    embedding_map_path = Path(r"C:\db\Gemini_Generated_Json\mk_stalin\mk_stalin_embedding_map.json")
    articles_path = Path(r"C:\db\Gemini_Generated_Json\mk_stalin\mk_stalin_consolidated_articles_excluding_none.json")
    
    # Validate paths
    for path in [embeddings_path, embedding_map_path, articles_path]:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return 1
    
    # Load data
    embeddings, embedding_map, articles = load_data_files(
        embeddings_path, embedding_map_path, articles_path
    )
    
    # Connect to Weaviate
    logger.info("Connecting to Weaviate at http://localhost:8081")
    client = weaviate.connect_to_local(
        host="localhost",
        port=8081
    )
    
    try:
        # Create collection
        collection = create_collection(client)
        
        # Populate data
        populate_weaviate(collection, embeddings, embedding_map, articles)
        
        # Verify
        result = collection.aggregate.over_all(total_count=True)
        logger.info(f"✅ Total objects in Weaviate: {result.total_count}")
        
    finally:
        client.close()
    
    logger.info("🎉 Migration completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())