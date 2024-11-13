"""
Main entry point for the RAG (Retrieval-Augmented Generation) system.

This script initializes and configures the RAG system using configuration files
for both initialization and processing parameters. It handles command-line arguments,
system setup, and error handling.

The script:
1. Parses command-line arguments for configuration paths
2. Initializes the RAG system with provided configurations
3. Sets up the retriever component with document chunks
4. Implements error handling and logging

Usage:
    python main.py --init-config path/to/init_config.yaml --process-config path/to/process_config.yaml

Configuration Files:
    - init_config.yaml: System initialization parameters
        - Model configurations
        - Path settings
        - Initial setup parameters
        
    - process_config.yaml: Processing parameters
        - Chunk sizes
        - Retrieval settings
        - Processing options

Example:
    ```bash
    python main.py --init-config config/custom_init.yaml --process-config config/custom_process.yaml
    ```
"""

import argparse
import logging
from initialize_rag import RAGInitializer

def main(init_config_path: str, process_config_path: str) -> None:
    """
    Initialize and set up the RAG system with provided configurations.
    
    This function:
    1. Creates a RAG initializer instance
    2. Initializes system components
    3. Sets up the retriever with document chunks
    
    Args:
        init_config_path (str): Path to initialization configuration file
        process_config_path (str): Path to processing configuration file
        
    Raises:
        Exception: If any error occurs during initialization or setup
        
    Note:
        Both configuration files must be valid YAML files containing
        necessary parameters for system initialization and processing.
    """
    try:
        # Create and initialize RAG system with configurations
        initializer = RAGInitializer(init_config_path, process_config_path)
        
        # Initialize all system components
        components = initializer.initialize()
        
        # Set up retriever with document chunks
        logging.info("Initializing retriever...")
        components.retriever.initialize(components.original_chunks)
        
    except Exception as e:
        # Log error details and re-raise
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='RAG (Retrieval-Augmented Generation) System Initialization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py  # Use default configuration files
            python main.py --init-config custom_init.yaml --process-config custom_process.yaml
                    """
                )
    
    # Define command-line arguments
    parser.add_argument(
        '--init-config', 
        type=str, 
        default='config/init_config.yaml',
        help='Path to initialization configuration file (default: config/init_config.yaml)'
    )
    parser.add_argument(
        '--process-config', 
        type=str, 
        default='config/process_config.yaml',
        help='Path to processing configuration file (default: config/process_config.yaml)'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Execute main function with provided configurations
    main(args.init_config, args.process_config)