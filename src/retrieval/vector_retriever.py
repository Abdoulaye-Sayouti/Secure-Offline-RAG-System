from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
import logging
from multiprocessing import Pool, cpu_count
from ..cache.cache import CacheManager
from langchain_huggingface import HuggingFaceEmbeddings
import torch

logger = logging.getLogger(__name__)

def embed_and_cache(embeddings_model, texts, cache_manager):
    """
    Embed texts and cache the results for future use.
    
    Args:
        embeddings_model: Model to generate embeddings
        texts (List[str]): List of texts to embed
        cache_manager: Cache manager instance
        
    Returns:
        List[List[float]]: Document embeddings
        
    Note:
        Uses first 100 characters of concatenated texts as cache key
        to ensure consistent cache hits for same content.
    """
    # Generate cache key from first 100 chars of concatenated texts
    cache_key = "".join([chunk for chunk in texts[:100]])
    cached_content = cache_manager.get_cached(cache_key, "embeddings_parents")
    
    if cached_content:
        embeddings = cached_content['embeddings']
    else:
        embeddings = embeddings_model.embed_documents(texts)
        # Cache embeddings with texts and empty sources
        cache_manager.cache_content(cache_key, {
                    'texts': texts,
                    'sources': [],
                    'embeddings': embeddings
                }, "embeddings_parents")

    return embeddings


class VectorRetriever:
    """
    Advanced vector-based document retrieval system with support for direct FAISS
    and hierarchical document retrieval.
    
    This class provides two main retrieval approaches:
    1. Direct FAISS vector store for simple similarity search
    2. ParentDocumentRetriever for hierarchical document retrieval
    
    Features:
        - GPU-accelerated embeddings with HuggingFace models
        - Efficient caching system for embeddings
        - Support for both flat and hierarchical document retrieval
        - Automatic memory management for GPU operations
        - Score normalization for consistent ranking
    
    Attributes:
        config (dict): Configuration settings
        vectorstore: FAISS vector store instance
        parent_retriever: ParentDocumentRetriever instance
        model_embeddings_hf: HuggingFace embeddings model
        cache_manager: Cache management system
        texts (List[str]): Stored document texts
        sources (List[str]): Document source references
    
    Example:
        ```python
        retriever = VectorRetriever(config)
        retriever.create_vectorstore(documents)
        results, scores = retriever.retrieve("search query", top_k=5)
        ```
    """
    
    def __init__(self, config: dict):
        """
        Initialize VectorRetriever with specified configuration.
        
        Args:
            config (dict): Configuration containing:
                - paths.cache_dir: Directory for caching
                - model.embedding_model: Ollama model name
                - model.embedding_model_hf: HuggingFace model name
                - processing.batch_size_embeddings: Batch size
                - processing.use_parent_document_retriever: Use hierarchical retrieval
                - processing.parent_chunk_size: Size of parent chunks
                - processing.child_chunk_size: Size of child chunks
        """
        self.config = config
        self.cache_dir = config['paths']['cache_dir']
        self.embedding_model_name_hf = config['model']['embedding_model_hf']
        self.batch_size = config['processing']['batch_size_embeddings']
        self.use_parent_retriever = config['processing'].get('use_parent_document_retriever', False)
        self.parent_chunk_size = config['processing'].get('parent_chunk_size', 2000)
        self.child_chunk_size = config['processing'].get('child_chunk_size', 400)
        
        self.logger = logging.getLogger(__name__)
        self.vectorstore = None
        self.parent_retriever = None
        self.num_processes = cpu_count() - 1 or 1
        self.cache_manager = CacheManager(self.cache_dir)
        
        # Initialize HuggingFace embeddings model with GPU support
        model_kwargs = {'device': "cuda", 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': self.batch_size}
        self.model_embeddings_hf = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name_hf,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True
        )
        
       

    def create_vectorstore(self, chunks: List[Document]) -> None:
        """
        Create vector store based on configuration.
        
        Args:
            chunks (List[Document]): Document chunks to process
            
        Note:
            Creates either ParentDocumentRetriever or direct FAISS store
            based on configuration settings.
        """
        if self.use_parent_retriever:
            self._create_parent_retriever(chunks)
        else:
            self._create_direct_vectorstore(chunks)

    def _create_parent_retriever(self, chunks: List[Document]) -> None:
        """
        Create hierarchical document retriever with parent-child relationships.
        
        Args:
            chunks (List[Document]): Documents to process
            
        Note:
            Uses MarkdownTextSplitter for both parent and child documents
            to maintain markdown formatting.
        """
        # Initialize splitters for hierarchical document structure
        parent_splitter = MarkdownTextSplitter(chunk_size=self.parent_chunk_size)
        child_splitter = MarkdownTextSplitter(chunk_size=self.child_chunk_size)

        # Process documents and store references
        child_docs = child_splitter.split_documents(chunks)
        self.texts = [doc.page_content for doc in child_docs]
        self.sources = [doc.metadata.get("source", "") for doc in child_docs]
        
        # Initialize FAISS with dummy document
        texts = ["Initialization document"]
        faiss = FAISS.from_texts(texts, self.model_embeddings_hf)
        
        # Set up document storage and retriever
        store = InMemoryStore()
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=faiss,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        # Add documents to retriever
        self.parent_retriever.add_documents(chunks)
        self.logger.info("Parent Document Retriever created successfully")

    def _create_direct_vectorstore(self, chunks: List[Document]) -> None:
        """
        Create direct FAISS vector store with caching support.
        
        Args:
            chunks (List[Document]): Documents to process
            
        Note:
            Implements caching for embeddings to improve performance
            on subsequent runs with same content.
        """
        # Generate cache key from first 100 chunks
        cache_key = "".join([chunk.page_content for chunk in chunks[:100]])
        cached_content = self.cache_manager.get_cached(cache_key, "embeddings_direct")
        
        if cached_content:
            # Use cached embeddings if available
            self.texts = cached_content['texts']
            embeddings = cached_content['embeddings']
            self.sources = cached_content['sources']
        else:
            # Generate and cache new embeddings
            self.texts = [chunk.page_content for chunk in chunks]
            self.sources = [chunk.metadata["source"] for chunk in chunks]
            embeddings = self.model_embeddings_hf.embed_documents(self.texts)
            self.cache_manager.cache_content(cache_key, {
                    'texts': self.texts,
                    'sources': self.sources,
                    'embeddings': embeddings
                }, "embeddings_direct")
            
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create FAISS store
        text_embedding_pairs = zip(self.texts, embeddings)
        self.vectorstore = FAISS.from_embeddings(text_embedding_pairs, self.model_embeddings_hf)
        self.logger.info("Vector store created successfully")

    def get_retrieved_docs_indexes(self, retrieved_docs):
        """
        Map retrieved documents back to their original indexes.
        
        Args:
            retrieved_docs (List[Document]): Retrieved documents
            
        Returns:
            List[int]: Original indexes of retrieved documents
        """
        indexes = []
        for doc in retrieved_docs:
            for i, orig_text in enumerate(self.texts):
                if doc.page_content == orig_text:
                    indexes.append(i)
                    break
        return indexes

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[float]]:
        """
        Retrieve most relevant documents for the given query.
        
        Args:
            query (str): Search query
            top_k (int): Number of documents to retrieve (default: 5)
            
        Returns:
            Tuple[List[Document], List[float]]: Retrieved documents and 
                their normalized similarity scores
                
        Raises:
            ValueError: If vector store or retriever not initialized
            
        Note:
            For ParentDocumentRetriever, returns constant scores of 1.0
            as it doesn't provide similarity scores.
        """
        if self.use_parent_retriever:
            if not self.parent_retriever:
                raise ValueError("Parent Document Retriever not created. Call create_vectorstore first.")
            
            docs = self.parent_retriever.invoke(f"search_query: {query}", top_k=top_k)
            scores = [1.0] * len(docs)
            return docs, scores
        else:
            if not self.vectorstore:
                raise ValueError("Vectorstore not created. Call create_vectorstore first.")

            # Get documents and scores from FAISS
            results = self.vectorstore.similarity_search_with_score(f"search_query: {query}", k=top_k)
            docs, scores = zip(*results)
            
            # Map documents to original sources
            idx_list = self.get_retrieved_docs_indexes(docs)
            updated_docs = [Document(page_content=d.page_content, metadata={"source": self.sources[idx_list[k]]}) 
                          for k, d in enumerate(docs)]
            
            # Normalize scores to [0,1] range
            max_distance = max(scores)
            normalized_scores = [1 - (dist/max_distance) for dist in scores]

            return updated_docs, normalized_scores