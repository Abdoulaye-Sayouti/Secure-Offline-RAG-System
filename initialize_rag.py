import logging
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.models.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.response.response_generator import ResponseGenerator
from src.utils.helpers import setup_logging, load_config

@dataclass
class RAGComponents:
    retriever: HybridRetriever
    reranker: Reranker
    response_generator: ResponseGenerator
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_results: pd.DataFrame
    test_results: pd.DataFrame
    original_chunks: List
    init_config: Dict
    process_config: Dict

class RAGInitializer:
    def __init__(self, init_config_path: str, process_config_path: str):
        """Initialize the RAG system components and data.
        
        Args:
            init_config_path (str): Path to initialization configuration file
            process_config_path (str): Path to processing configuration file
        """
        self.init_config = load_config(init_config_path)
        self.process_config = load_config(process_config_path)
        setup_logging(self.init_config)
        
    def initialize(self) -> RAGComponents:
        """Initialize all components and prepare data.
        
        Returns:
            RAGComponents: Dataclass containing all initialized components and data
        """
        try:
            # Initialize components
            logging.info("Initializing components...")
            data_ingestion = DataIngestion(self.init_config)
            data_preprocessor = DataPreprocessor(self.init_config)
            
            # Initialize retrieval components with both configs
            combined_config = {
                **self.init_config,
                'retrieval': self.process_config['retrieval']
            }
            retriever = HybridRetriever(combined_config)
            reranker = Reranker(combined_config)
            response_generator = ResponseGenerator(combined_config)
            
            # Load and prepare data
            logging.info("Loading and preprocessing data...")
            train_df, test_df = data_ingestion.load_data()
            docs = data_ingestion.load_documents()
            train_results, test_results = data_ingestion.load_existing_results()
            
            # Process documents
            logging.info("Processing documents...")
            original_chunks = data_preprocessor.process_documents(docs)
            
            return RAGComponents(
                retriever=retriever,
                reranker=reranker,
                response_generator=response_generator,
                train_df=train_df,
                test_df=test_df,
                train_results=train_results,
                test_results=test_results,
                original_chunks=original_chunks,
                init_config=self.init_config,
                process_config=self.process_config
            )
            
        except Exception as e:
            logging.error(f"Error in RAG initialization: {str(e)}")
            raise

if __name__ == "__main__":
    # This can be used for testing the initialization separately
    initializer = RAGInitializer(
        "config/init_config.yaml",
        "config/process_config.yaml"
    )
    components = initializer.initialize()
    logging.info("Initialization completed successfully")