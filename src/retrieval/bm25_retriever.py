from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain.schema import Document
import numpy as np
import logging
from ..cache.cache import CacheManager

class BM25Retriever:
    """
    BM25-based document retrieval system with caching capabilities.
    
    This class implements the BM25 Okapi algorithm for document retrieval,
    providing efficient search functionality with score normalization and
    caching support. It integrates with LangChain's Document format and
    includes performance optimizations.
    
    Features:
        - BM25 Okapi ranking algorithm
        - Document index caching
        - Score normalization
        - Configurable top-k retrieval
        - Integration with LangChain Documents
    
    Attributes:
        config (dict): Configuration settings
        cache_dir (str): Directory for index caching
        bm25_index (BM25Okapi): BM25 search index
        chunks (List[Document]): Stored document chunks
        cache_manager (CacheManager): Cache management system
        logger (logging.Logger): Logger instance
    
    Note:
        - Index must be created before retrieval
        - Implements BM25 Okapi variant for better ranking
        - Uses caching to improve performance
    """
    
    def __init__(self, config: dict):
        """
        Initialize the BM25Retriever with configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - paths:
                    - cache_dir: Directory for storing cached indices
                
        Note:
            - Initializes empty index and document store
            - Sets up caching and logging
            - Index must be created before retrieval
        """
        self.config = config
        self.cache_dir = config['paths']['cache_dir']
        self.logger = logging.getLogger(__name__)
        self.bm25_index = None
        self.chunks = None
        self.cache_manager = CacheManager(self.config['paths']['cache_dir'])

    def create_index(self, chunks: List[Document]) -> None:
        """
        Create or load BM25 index from document chunks.
        
        Creates a new BM25 index from the provided document chunks or loads
        a cached version if available. The index is used for subsequent
        retrieval operations.
        
        Args:
            chunks (List[Document]): Document chunks to index
            
        Note:
            - Attempts to load from cache first
            - Creates new index if cache miss
            - Caches new indices for future use
            - Tokenizes documents for indexing
            
        Example:
            >>> retriever = BM25Retriever(config)
            >>> retriever.create_index(document_chunks)
        """
        self.chunks = chunks
        
        # Generate cache key from first 100 chunks
        cache_key = "".join([chunk.page_content for chunk in chunks[:100]])
        
        # Try loading from cache
        cached_content = self.cache_manager.get_cached(cache_key, "bm25")
        if cached_content:
            self.bm25_index = cached_content
            self.logger.info("Loaded BM25 index from cache")
            return
        
        # Create new index if not in cache
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
        # Cache the new index
        self.cache_manager.cache_content(cache_key, self.bm25_index, "bm25")
        self.logger.info("Created and cached new BM25 index")

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[float]]:
        """
        Retrieve relevant documents for a given query using BM25 ranking.
        
        Searches the index for documents matching the query and returns
        the top-k most relevant documents with their normalized relevance scores.
        
        Args:
            query (str): Search query text
            top_k (int, optional): Number of documents to retrieve. 
                Defaults to 5.
                
        Returns:
            Tuple[List[Document], List[float]]: Tuple containing:
                - List of retrieved documents
                - List of normalized relevance scores
                
        Raises:
            ValueError: If index hasn't been created yet
            
        Note:
            - Scores are normalized to [0,1] range
            - Documents are returned in descending score order
            - Implements efficient numpy-based ranking
            
        Example:
            >>> docs, scores = retriever.retrieve("search query", top_k=3)
        """
        if not self.bm25_index or not self.chunks:
            raise ValueError("Index not created. Call create_index first.")
            
        # Tokenize query for BM25
        tokenized_query = query.split()
        
        # Calculate relevance scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k documents using numpy
        top_indices = np.argsort(-scores)[:top_k]
        retrieved_docs = [self.chunks[i] for i in top_indices]
        retrieved_scores = [scores[i] for i in top_indices]
        
        # Normalize scores to [0,1] range
        max_score = max(retrieved_scores) if retrieved_scores else 1.0
        normalized_scores = [score/max_score for score in retrieved_scores]
        
        return retrieved_docs, normalized_scores