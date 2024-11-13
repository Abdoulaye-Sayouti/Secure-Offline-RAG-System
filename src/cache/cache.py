import pickle
from pathlib import Path
from typing import  Optional, List
import hashlib
from langchain.docstore.document import Document
import logging

class CacheManager:
    """
    A class to manage document caching operations with improved handling of complex structures.
    
    This class provides functionality to cache and retrieve Document objects, using a file-based
    caching system. It implements MD5 hashing for cache keys and handles serialization/deserialization
    of Document objects using pickle.
    
    Attributes:
        cache_dir (Path): Directory path where cache files will be stored
        logger (Logger): Logger instance for error tracking and debugging
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the CacheManager with a specified cache directory.

        Args:
            cache_dir (str): Path to the directory where cache files will be stored.
                           Directory will be created if it doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_cache_path(self, key: str, prefix: str) -> Path:
        """
        Generate a unique cache file path for a given key and prefix.

        Args:
            key (str): The cache key to generate a path for
            prefix (str): Prefix to be added to the cache filename

        Returns:
            Path: Complete path to the cache file

        Note:
            The file path is generated using MD5 hash of the key to ensure
            filename compatibility and uniqueness
        """
        return self.cache_dir / f"{prefix}_{hashlib.md5(key.encode()).hexdigest()}.pkl"

    def get_cached(self, key: str, prefix: str) -> Optional[List[Document]]:
        """
        Retrieve cached content if it exists.

        Args:
            key (str): The key to lookup in the cache
            prefix (str): Prefix used when the content was cached

        Returns:
            Optional[List[Document]]: List of cached Document objects if found,
                                    None if not found or on error

        Raises:
            No exceptions are raised, errors are logged instead
        """
        cache_path = self.get_cache_path(key, prefix)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
                return None
        return None

    def cache_content(self, key: str, documents: List[Document], prefix: str):
        """
        Cache content for future retrieval.

        Args:
            key (str): The key under which to cache the content
            documents (List[Document]): List of Document objects to cache
            prefix (str): Prefix to be used in the cache filename

        Note:
            - If caching fails, the error is logged but the program continues execution
            - Uses pickle for serialization to handle complex Document objects
        """
        cache_path = self.get_cache_path(key, prefix)
        try:
            # Cache the result
            with open(cache_path, 'wb') as f:
                pickle.dump(documents, f)
        except Exception as e:
            self.logger.error(f"Error caching content: {str(e)}")
            # If caching fails, we'll just log the error and continue
            # This prevents the overall process from failing due to cache issues
            pass
