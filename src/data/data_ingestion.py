from typing import  List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import logging
from urllib.parse import urlparse
from tqdm import tqdm
from langchain.docstore.document import Document

from .loaders import FileLoader, EnhancedWebLoader
from ..cache.cache import CacheManager


class DataIngestion:
    """
    Comprehensive data ingestion system supporting multiple source types and formats.
    
    This class provides a unified interface for loading, processing, and managing
    data from various sources including local files, web content, and specialized
    platforms (Wikipedia, GitBook, etc.). It includes caching, validation, and
    result management capabilities.
    
    Features:
        - Multi-source document loading (files, web, special sources)
        - Automatic caching of processed documents
        - Configurable document processing
        - Data validation and integrity checks
        - Comprehensive error handling and logging
        - Progress tracking for long operations
    
    Attributes:
        config (dict): Configuration dictionary controlling ingestion behavior
        cache_manager (CacheManager): Manages document caching
        file_loader (FileLoader): Handles local file loading
        logger (logging.Logger): Logger for tracking operations
    """
    
    def __init__(self, config: dict):
        """
        Initialize the data ingestion system.
        
        Args:
            config (dict): Configuration dictionary containing:
                - paths: Directory configurations
                - processing: Chunking parameters
                - files: File paths for data
                - data_validation: Validation rules
                - results_validation: Result validation rules
                - saving: Output saving parameters
        """
        self.config = config
        self.cache_manager = CacheManager(self.config['paths']['cache_dir'])
        self.file_loader = FileLoader(config)
        self.logger = logging.getLogger(__name__)
    
    def get_document_paths(self) -> List[str]:
        """
        Retrieve document paths from configuration.
        
        Returns:
            List[str]: List of document paths to process
            
        Note:
            Logs the number of documents found for processing
        """
        doc_paths = self.config['files']['document_paths']
        self.logger.info(f"Found {len(doc_paths)} documents for processing")
        return doc_paths
    
    def load_documents(self, traverse_toc: bool = True) -> List[Document]:
        """
        Load and process documents from all configured sources.
        
        Args:
            traverse_toc (bool): Whether to follow table of contents links
                in web sources. Defaults to True.
            
        Returns:
            List[Document]: Processed LangChain documents ready for use
            
        Note:
            - Attempts to load from cache first
            - Processes and caches new documents
            - Handles multiple source types automatically
            - Shows progress bar during loading
        """
        sources = self.get_document_paths()
        all_documents = {}
        
        for source in tqdm(sources, desc="Loading documents"):
            source_str = str(source)
            
            # Try cache first
            cached_content = self.cache_manager.get_cached(source_str, "docs")
            if cached_content is not None:
                all_documents[source_str] = cached_content
                continue
            
            try:
                if isinstance(source, (str, Path)) and Path(source_str).exists():
                    docs = self._process_local_file(source)
                elif urlparse(source_str).scheme in ['http', 'https']:
                    docs = self._process_web_content(source_str, traverse_toc)
                
                
                if docs:
                    self.cache_manager.cache_content(source_str, docs, "docs")
                    all_documents[source_str] = docs
                    self.logger.info(f"Successfully processed and cached documents from {source_str}")
                
            except Exception as e:
                self.logger.error(f"Error processing {source_str}: {str(e)}")
                raise
        
        return all_documents
    
    def _process_local_file(self, source: Union[str, Path]) -> List[Document]:
        """
        Process a local file using appropriate loader.
        
        Args:
            source (Union[str, Path]): Path to local file
            
        Returns:
            List[Document]: Processed documents from file
        """
        raw_docs = self.file_loader.load(source)
        return raw_docs

    def _process_web_content(self, url: str, traverse_toc: bool) -> List[Document]:
        """
        Process web content with enhanced scraping capabilities.
        
        Args:
            url (str): Web URL to process
            traverse_toc (bool): Whether to follow TOC links
            
        Returns:
            List[Document]: Processed documents from web content
            
        Note:
            Uses configuration for TOC selector and chunking parameters
        """
        loader = EnhancedWebLoader(
            config = self.config,
            base_url=url,
            traverse_toc=traverse_toc,
            toc_selector=self.config.get('web_scraping', {}).get('toc_selector', 'nav a'),
        )
        return loader.load()



    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from CSV files.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
            
        Raises:
            FileNotFoundError: If CSV files not found
            ValueError: If validation fails
            
        Note:
            Performs validation on loaded dataframes
        """
        self.logger.info("Loading CSV data...")
        
        try:
            train_df = pd.read_csv(self.config['files']['train_data'])
            test_df = pd.read_csv(self.config['files']['test_data'])
            
            self.logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
            self._validate_dataframes(train_df, test_df)
            
            return train_df, test_df
            
        except FileNotFoundError as e:
            self.logger.error(f"Error loading CSV files: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {str(e)}")
            raise
    
    def _validate_dataframes(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Validate the structure and content of loaded dataframes.
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame): Test dataframe
            
        Raises:
            ValueError: If validation rules are not met
            
        Note:
            Checks for required columns from configuration
        """
        required_columns = self.config.get('data_validation', {}).get('required_columns', [])
        for df, name in [(train_df, 'train'), (test_df, 'test')]:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in {name} data: {missing_columns}")
    
    def load_existing_results(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load existing result files if available.
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
                Existing train and test results
                
        Note:
            Returns None for missing result files
        """
        train_result = None
        test_result = None
        
        try:
            train_path = Path(self.config['files']['train_output'])
            test_path = Path(self.config['files']['test_output'])
            
            if train_path.exists():
                train_result = pd.read_csv(train_path)
                self.logger.info(f"Loaded {len(train_result)} existing train results")
                
            if test_path.exists():
                test_result = pd.read_csv(test_path)
                self.logger.info(f"Loaded {len(test_result)} existing test results")
                
        except Exception as e:
            self.logger.warning(f"Error loading existing results: {str(e)}")
            
        return train_result, test_result
    
    def save_results(self, results: pd.DataFrame, is_test: bool = False):
        """
        Save results to CSV file with validation.
        
        Args:
            results (pd.DataFrame): Results to save
            is_test (bool): Whether these are test results. Defaults to False.
            
        Raises:
            ValueError: If results validation fails
            
        Note:
            - Creates output directory if needed
            - Applies configured saving options
            - Validates results before saving
        """
        output_path = Path(self.config['files']['test_output'] if is_test else self.config['files']['train_output'])
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_results(results)
            
            save_kwargs = self.config.get('saving', {}).get('csv_options', {})
            results.to_csv(output_path, index=False, **save_kwargs)
            
            self.logger.info(f"Saved results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _validate_results(self, results: pd.DataFrame):
        """
        Validate results before saving.
        
        Args:
            results (pd.DataFrame): Results to validate
            
        Raises:
            ValueError: If validation rules are not met
            
        Note:
            - Checks for required columns
            - Validates against configured rules
            - Ensures non-empty results
            - Checks for nulls and uniqueness if configured
        """
        required_columns = self.config.get('results_validation', {}).get('required_columns', [])
        missing_columns = set(required_columns) - set(results.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in results: {missing_columns}")
        
        if results.empty:
            raise ValueError("Results DataFrame is empty")
        
        validation_rules = self.config.get('results_validation', {}).get('rules', {})
        for column, rules in validation_rules.items():
            if column not in results.columns:
                continue
                
            if rules.get('no_nulls', False) and results[column].isnull().any():
                raise ValueError(f"Column {column} contains null values")
            
            if 'unique' in rules and not results[column].is_unique:
                raise ValueError(f"Column {column} contains duplicate values")