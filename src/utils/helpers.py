import logging
import yaml
from typing import Dict, Any
import csv
import os
from datetime import datetime

def setup_logging(config: Dict) -> None:
    """
    Set up a comprehensive logging configuration for the RAG system.
    
    This function:
    1. Creates a timestamped log file in the specified directory
    2. Configures both file and console logging
    3. Sets appropriate logging levels for different components
    4. Optionally disables progress bars for batch processing
    
    Args:
        config (Dict): Configuration dictionary containing:
            - paths.log_dir (str): Directory for log files
            - logging.level (str, optional): Logging level (default: 'INFO')
            - logging.show_progress (bool, optional): Whether to show progress bars
    
    Example config:
        {
            'paths': {'log_dir': './logs'},
            'logging': {
                'level': 'DEBUG',
                'show_progress': False
            }
        }
    
    Note:
        Creates the log directory if it doesn't exist.
        Automatically silences some verbose external loggers.
    """
    # Create log directory if it doesn't exist
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'rag_{timestamp}.log')
    
    # Set logging level from config or default to INFO
    log_level = config.get('logging', {}).get('level', 'INFO')
    
    # Configure root logger with both file and console output
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise by setting higher threshold for HTTP-related logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # Optionally disable progress bars if specified in config
    if not config.get('logging', {}).get('show_progress', True):
        from tqdm import tqdm
        tqdm.write = lambda *args, **kwargs: None

def load_config(config_path: str) -> Dict:
    """
    Load and parse YAML configuration file.
    
    Args:
        config_path (str): Path to YAML configuration file
    
    Returns:
        Dict: Parsed configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    
    Example:
        >>> config = load_config('config/rag_config.yml')
        >>> print(config['paths']['data_dir'])
        './data'
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_results(results: Dict[str, Any],
                output_path: str,
                is_test: bool = False) -> None:
    """
    Save RAG system results to a CSV file.
    
    Appends results to an existing CSV file or creates a new one.
    Automatically creates the output directory if it doesn't exist.
    
    Args:
        results (Dict[str, Any]): Results dictionary containing:
            - Query: The input query
            - Response: The system's response
            - Score: Confidence or relevance score
            - trustii_id (optional): Test case ID for test results
        output_path (str): Path where CSV file should be saved
        is_test (bool, optional): Whether these are test results. 
            If True, includes additional fields. Defaults to False.
    
    Note:
        The function will:
        1. Create the output directory if it doesn't exist
        2. Create a new CSV file with headers if it doesn't exist
        3. Append new results to existing file if it exists
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define fields based on result type
    fieldnames = ["trustii_id", "Query", "Response"] if is_test else ["Query", "Response"]
    
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only for new files
        if f.tell() == 0:
            writer.writeheader()
        
        writer.writerow(results)

def create_directory_structure(base_path: str) -> None:
    """
    Create standardized directory structure for RAG project.
    
    Creates a comprehensive directory structure for organizing:
    - Configuration files
    - Raw and processed data
    - Cache files
    - Logs
    - Source code by component
    
    Args:
        base_path (str): Root directory for the project
    
    Directory Structure:
        base_path/
        ├── config/           # Configuration files
        ├── data/            
        │   ├── raw/         # Original, unprocessed data
        │   └── processed/   # Processed, ready-to-use data
        ├── cache/           # Cached embeddings and computations
        ├── logs/            # Log files
        └── src/             # Source code
            ├── data/        # Data processing code
            │   └── loaders/ # Data loading utilities
            ├── retrieval/   # Document retrieval components
            ├── models/      # Model definitions and handlers
            ├── response/    # Response generation logic
            └── utils/       # Utility functions
    
    Note:
        Creates directories only if they don't exist.
        Existing directories are left unchanged.
    """
    directories = [
        'config',
        'data/raw',
        'data/processed',
        'cache',
        'logs',
        'src/data',
        'src/data/loaders',
        'src/retrieval',
        'src/models',
        'src/response',
        'src/utils'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)