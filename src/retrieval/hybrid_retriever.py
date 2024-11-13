from langchain.schema import Document
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever
import logging
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RetrievalResult:
    """
    Data class to hold document retrieval results.

    Attributes:
        document (Document): The retrieved document
        score (float): Relevance score (0 to 1, higher is better)
        source (str): Source of the retrieval result:
            - 'bm25': Retrieved using BM25 algorithm only
            - 'vector': Retrieved using vector similarity only
            - 'hybrid': Retrieved and scored using both methods
    """
    document: Document
    score: float
    source: str 

class HybridRetriever:
    """
    Hybrid document retrieval system combining BM25 and vector-based approaches.
    
    This system combines the strengths of both BM25 (keyword-based) and 
    vector (semantic) retrieval methods to provide more robust document retrieval.
    It supports:
    - Pure BM25 retrieval for keyword-focused search
    - Pure vector retrieval for semantic search
    - Hybrid retrieval combining both approaches with configurable weights
    
    The system handles deduplication when documents are retrieved by both methods
    and combines their scores using the configured weights.
    """

    def __init__(self, config: dict):
        """
        Initialize the hybrid retrieval system.
        
        Args:
            config (dict): Configuration dictionary containing:
                - retrieval.bm25_weight (float): Weight for BM25 scores (0 to 1)
                    Vector weight will be (1 - bm25_weight)
                - Additional configuration for BM25 and Vector retrievers
        
        Note:
            Both retrievers must be initialized separately using the initialize()
            method before performing retrieval operations.
        """
        self.config = config
        self.bm25_weight = config['retrieval']['bm25_weight']
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual retrieval systems
        self.bm25_retriever = BM25Retriever(config)
        self.vector_retriever = VectorRetriever(config)

    def initialize(self, chunks: List[Document]) -> None:
        """
        Initialize both BM25 and vector indexes with document chunks.
        
        This method must be called before performing any retrieval operations.
        It sets up both the BM25 index and the vector store in parallel.
        
        Args:
            chunks (List[Document]): List of document chunks to index.
                Each document should have a page_content attribute.
        
        Note:
            This operation can be time-consuming for large document collections
            as it involves computing embeddings for the vector store.
        """
        self.logger.info("Initializing BM25 index...")
        self.bm25_retriever.create_index(chunks)
        
        self.logger.info("Initializing vector store...")
        self.vector_retriever.create_vectorstore(chunks)
        
        self.logger.info("Hybrid retriever initialization complete")

    def retrieve(self,
                query: str,
                top_k: int = 5,
                use_bm25: bool = True,
                use_vector: bool = True) -> List[RetrievalResult]:
        """
        Retrieve documents using a combination of BM25 and vector-based approaches.
        
        This method:
        1. Retrieves documents using enabled methods (BM25 and/or vector)
        2. Combines and deduplicates results
        3. Calculates final scores using configured weights
        4. Returns top_k documents sorted by score
        
        Args:
            query (str): Search query text
            top_k (int, optional): Number of documents to retrieve. Defaults to 5.
            use_bm25 (bool, optional): Whether to use BM25 retrieval. Defaults to True.
            use_vector (bool, optional): Whether to use vector retrieval. Defaults to True.
            
        Returns:
            List[RetrievalResult]: Top k retrieved documents, sorted by combined score.
                Each result includes the document, score, and retrieval source.
        
        Note:
            The method retrieves top_k*5 documents from each enabled retriever
            to ensure good candidates for the final top_k after score combination.
        """
        combined_results = {}
        
        # Retrieve and score documents using BM25 if enabled
        if use_bm25:
            bm25_docs, bm25_scores = self.bm25_retriever.retrieve(query, top_k*10)
            for doc, score in zip(bm25_docs, bm25_scores):
                doc_content = doc.page_content
                if doc_content not in combined_results:
                    combined_results[doc_content] = RetrievalResult(
                        document=doc,
                        score=score * self.bm25_weight,
                        source='bm25'
                    )
                else:
                    combined_results[doc_content].score += score * self.bm25_weight
            

        # Retrieve and score documents using vector similarity if enabled
        if use_vector:
            vector_docs, vector_scores = self.vector_retriever.retrieve(query, top_k*10)
            vector_weight = 1 - self.bm25_weight
            for doc, score in zip(vector_docs, vector_scores):
                doc_content = doc.page_content
                if doc_content not in combined_results:
                    combined_results[doc_content] = RetrievalResult(
                        document=doc,
                        score=score * vector_weight,
                        source='vector'
                    )
                else:
                    # Document was found by both methods - combine scores and mark as hybrid
                    combined_results[doc_content].score += score * vector_weight
                    combined_results[doc_content].source = 'hybrid'
        
        # Sort by score and return top_k results
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )



        return sorted_results[:top_k]

    def retrieve_with_method(self, 
                           query: str,
                           method: str = "hybrid",
                           top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve documents using a specific retrieval method.
        
        This is a convenience method that wraps retrieve() to provide
        a simpler interface for selecting the retrieval approach.
        
        Args:
            query (str): Search query text
            method (str, optional): Retrieval method to use. Defaults to "hybrid".
                Valid values:
                - "bm25": Use only BM25 retrieval
                - "vector": Use only vector retrieval
                - "hybrid": Use both methods with configured weights
            top_k (int, optional): Number of documents to retrieve. Defaults to 5.
            
        Returns:
            List[RetrievalResult]: Top k retrieved documents with scores
            
        Raises:
            ValueError: If an unknown retrieval method is specified
        """
        if method == "bm25":
            return self.retrieve(query, top_k, use_bm25=True, use_vector=False)
        elif method == "vector":
            return self.retrieve(query, top_k, use_bm25=False, use_vector=True)
        elif method == "hybrid":
            return self.retrieve(query, top_k, use_bm25=True, use_vector=True)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")