"""
WeaviateAdapter - Adapter for Weaviate vector database (v3 Client)

This adapter implements the VectorStore interface using the Weaviate v3 Client API.
This is required for compatibility with Weaviate Server versions < 1.25.0 when
using weaviate-client v4.x.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from core.interfaces import VectorStore

# Weaviate imports (external dependency)
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


class WeaviateAdapter(VectorStore):
    """
    Adapter for Weaviate vector database using v3 Client API.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,  # Standard default, overridden by settings
        scheme: str = "http",
        collection_name: str = "PoliticalStatement"
    ):
        """
        Initialize Weaviate adapter.
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client package not installed.")
        
        self.host = host
        self.port = port
        self.scheme = scheme
        self.collection_name = collection_name
        
        self.logger = logging.getLogger(__name__)
        
        # Connect to Weaviate using v3 Client (Compatibility Layer)
        try:
            url = f"{scheme}://{host}:{port}"
            self._client = weaviate.Client(url)
            
            # Simple connectivity check
            if not self._client.is_ready():
                self.logger.warning(f"Weaviate at {url} is not ready yet.")
            
            self.logger.info(f"WeaviateAdapter connected to {url}, collection: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {e}")
            raise RuntimeError(f"Weaviate connection failed: {e}")
    
    def vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using v3 GraphQL API.
        """
        if query_vector is None or len(query_vector) == 0:
            raise ValueError("query_vector cannot be None or empty")
        
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        
        try:
            # Build query
            query = (
                self._client.query
                .get(self.collection_name, ["statement", "summary", "politician", "sentiment", "theme", "classification"])
                .with_near_vector({"vector": query_vector.tolist()})
                .with_limit(top_k)
                .with_additional(["distance", "id"])
            )
            
            # Apply filters if present
            # FIX: Check if where_filter is not None before applying
            if filters:
                where_filter = self._build_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute
            response = query.do()
            
            return self._convert_results(response, search_type="vector")
            
        except Exception as e:
            self.logger.error(f"Weaviate vector search failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}")
    
    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search using v3 GraphQL API.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        
        try:
            query = (
                self._client.query
                .get(self.collection_name, ["statement", "summary", "politician", "sentiment", "theme", "classification"])
                .with_bm25(query=query_text)
                .with_limit(top_k)
                .with_additional(["score", "id"])
            )
            
            # FIX: Check if where_filter is not None before applying
            if filters:
                where_filter = self._build_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            response = query.do()
            return self._convert_results(response, search_type="keyword")
            
        except Exception as e:
            self.logger.error(f"Weaviate keyword search failed: {e}")
            raise RuntimeError(f"Keyword search failed: {e}")
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using v3 GraphQL API.
        """
        try:
            query = (
                self._client.query
                .get(self.collection_name, ["statement", "summary", "politician", "sentiment", "theme", "classification"])
                .with_hybrid(query=query_text, vector=query_vector.tolist(), alpha=alpha)
                .with_limit(top_k)
                .with_additional(["score", "id"])
            )
            
            # FIX: Check if where_filter is not None before applying
            if filters:
                where_filter = self._build_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            response = query.do()
            return self._convert_results(response, search_type="hybrid")
            
        except Exception as e:
            self.logger.error(f"Weaviate hybrid search failed: {e}")
            raise RuntimeError(f"Hybrid search failed: {e}")
    
    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Weaviate v3 'where' filter.
        """
        operands = []
        
        for key, value in filters.items():
            if value is None:
                continue
            
            if isinstance(value, list):
                # OR logic for lists
                path_operands = []
                for v in value:
                    path_operands.append({
                        "path": [key],
                        "operator": "Equal",
                        "valueText": str(v)
                    })
                if path_operands:
                    operands.append({"operator": "Or", "operands": path_operands})
            else:
                # Simple Equal
                operands.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueText": str(value)
                })
        
        if not operands:
            return None
        
        if len(operands) == 1:
            return operands[0]
        
        return {"operator": "And", "operands": operands}
    
    def _convert_results(self, response: Dict[str, Any], search_type: str) -> List[Dict[str, Any]]:
        """
        Convert v3 GraphQL response to standard format.
        """
        results = []
        
        # v3 response structure: {'data': {'Get': {'CollectionName': [...]}}}
        try:
            if 'errors' in response:
                self.logger.error(f"Weaviate query error: {response['errors']}")
                return []
                
            data = response.get('data', {}).get('Get', {}).get(self.collection_name, [])
            
            for obj in data:
                additional = obj.get('_additional', {})
                
                # Normalize score
                score = 0.0
                if search_type == "vector":
                    dist = additional.get('distance', 0)
                    score = 1.0 / (1.0 + dist) if dist is not None else 0.0
                else:
                    score = float(additional.get('score', 0.0))
                
                doc = {
                    "content": obj.get("statement", ""),
                    "score": score,
                    "metadata": {
                        "summary": obj.get("summary", ""),
                        "politician": obj.get("politician", ""),
                        "sentiment": obj.get("sentiment", ""),
                        "theme": obj.get("theme", []),
                        "classification": obj.get("classification", "")
                    },
                    "id": additional.get("id", "")
                }
                results.append(doc)
                
        except Exception as e:
            self.logger.error(f"Result conversion failed: {e}")
            
        return results
    
    def close(self) -> None:
        """Close (Not really needed for v3, but keeps interface clean)"""
        self._client = None
    
    def get_collection_info(self) -> Dict[str, Any]:
        return {"store_type": "Weaviate v3", "connected": True}