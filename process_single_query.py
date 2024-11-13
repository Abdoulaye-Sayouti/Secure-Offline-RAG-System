"""
Query processing module for the RAG (Retrieval-Augmented Generation) system.

This module handles the processing of queries through the RAG pipeline, including:
- Query expansion (optional)
- Document retrieval
- Reranking (optional)
- Response generation

The module implements a complete query processing pipeline with configurable
components and error handling. It supports both direct queries and expanded
queries, with optional reranking of retrieved documents.

Features:
    - Configurable query expansion
    - Flexible document retrieval
    - Optional reranking of results
    - Score-based response generation
    - Comprehensive error handling
    - Command-line interface

Example:
    ```bash
    python process_queries.py --query "What is RAG?" --send_nb_chunks_to_llm 3
    ```
"""

import argparse
import logging
from typing import Dict

from initialize_rag import RAGInitializer

def process_query(query: str,
                 retriever,
                 reranker,
                 response_generator,
                 process_config: Dict,
                 send_nb_chunks_to_llm=1) -> Dict:
    """
    Process a single query through the complete RAG pipeline.
    
    This function orchestrates the query processing workflow:
    1. Optional query expansion
    2. Document retrieval
    3. Optional result reranking
    4. Response generation
    
    Args:
        query (str): The user's query
        retriever: Document retrieval component
        reranker: Result reranking component
        response_generator: Response generation component
        process_config (Dict): Processing configuration
        send_nb_chunks_to_llm (int): Number of chunks to send to LLM
        
    Returns:
        Dict: Processing results containing:
            - Query: Original query
            - Response: Generated response
            - Score: Best retrieval/reranking score
            
    Note:
        The function handles errors gracefully, returning an error message
        in the response if any step fails.
    """
    try:
        # Expand query if configured
        if process_config['retrieval']['use_query_expansion']:
            expanded_query = response_generator.expand_query(query)
            logging.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
            
        # Retrieve relevant documents using expanded or original query
        if process_config['retrieval']['use_bm25']:
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method="hybrid",
                top_k=process_config['retrieval']['top_k']
            )
        else:
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method="vector",
                top_k=process_config['retrieval']['top_k']
            )
        logging.info(f"Retrieved {len(retrieved_results)} documents")
        
        # Apply reranking if configured
        if process_config['retrieval']['use_reranking']:
            reranked_results = reranker.rerank(
                query,
                [r.document for r in retrieved_results],
                top_k=send_nb_chunks_to_llm
            )
            relevant_docs = [r.document for r in reranked_results]
            best_score = reranked_results[0].score if reranked_results else 0.0
            logging.info(f"Reranked results. Best score: {best_score}")
        else:
            relevant_docs = [r.document for r in retrieved_results]
            best_score = retrieved_results[0].score if retrieved_results else 0.0
            logging.info(f"Using retrieval scores. Best score: {best_score}")
        
        # Generate final response using selected documents
        response_data = response_generator.generate_answer(
            query,
            relevant_docs,
            metadata={'retrieval_score': best_score}
        )
        
        return {
            'Query': query,
            'Response': response_data['response'],
            'Score': best_score
        }
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {
            'Query': query,
            'Response': "An error occurred processing your query.",
            'Score': 0.0
        }

def main(init_config_path: str, process_config_path: str, query: str):
    """
    Main execution function for query processing.
    
    This function:
    1. Initializes the RAG system
    2. Sets up the retriever with appropriate chunks
    3. Processes the query
    4. Displays the response
    
    Args:
        init_config_path (str): Path to initialization config
        process_config_path (str): Path to processing config
        query (str): Query to process
        
    Raises:
        Exception: If initialization or processing fails
    """
    try:
        # Initialize system components
        initializer = RAGInitializer(init_config_path, process_config_path)
        components = initializer.initialize()
        logging.info("Initializing retriever...")


        components.retriever.initialize(components.original_chunks)
    
        # Process query and get response
        result = process_query(
            query=query,
            retriever=components.retriever,
            reranker=components.reranker,
            response_generator=components.response_generator,
            process_config=components.process_config,
            send_nb_chunks_to_llm=components.process_config['retrieval']['send_nb_chunks_to_llm']
        )
        
        # Display response (removing potential trailing markers)
        print("\n\nResponse:", result["Response"][:-4])
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(
        description='RAG Query Processing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_queries.py --query "What is RAG?"
  python process_queries.py --query "Explain neural networks" --send_nb_chunks_to_llm 3
  python process_queries.py --init-config custom_init.yaml --query "How does RAG work?"
        """
    )
    
    # Define command-line arguments
    parser.add_argument(
        '--init-config', 
        type=str, 
        default='config/init_config.yaml',
        help='Path to initialization configuration file'
    )
    parser.add_argument(
        '--process-config', 
        type=str, 
        default='config/process_config.yaml',
        help='Path to processing configuration file'
    )
    parser.add_argument(
        '--query', 
        type=str, 
        required=True,
        help='Query to process through the RAG system'
    )
    
    # Parse arguments and execute main function
    args = parser.parse_args()
    main(args.init_config, args.process_config, args.query)