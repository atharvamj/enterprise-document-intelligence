"""
Utility functions for the Enterprise Document Intelligence system.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    log_file = log_config.get('log_file', './outputs/logs/app.log')
    
    # Create log directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger.add(
        log_file,
        rotation=log_config.get('rotation', '10 MB'),
        retention=log_config.get('retention', '30 days'),
        format=log_config.get('format'),
        level=log_config.get('level', 'INFO')
    )
    logger.info("Logging configured successfully")


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['paths']['data_dir'],
        config['paths']['output_dir'],
        config['paths']['models_dir'],
        os.path.dirname(config['faiss']['index_path']),
        config['evaluation']['output_dir'],
        os.path.dirname(config['logging']['log_file'])
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("All required directories created")


def format_sources(sources: list, include_page_numbers: bool = True) -> str:
    """
    Format source citations for display.
    
    Args:
        sources: List of source dictionaries with metadata
        include_page_numbers: Whether to include page numbers
        
    Returns:
        Formatted source string
    """
    if not sources:
        return "No sources found."
    
    formatted = []
    for idx, source in enumerate(sources, 1):
        filename = source.get('filename', 'Unknown')
        page = source.get('page_number', 'N/A')
        
        if include_page_numbers and page != 'N/A':
            formatted.append(f"{idx}. {filename} (Page {page})")
        else:
            formatted.append(f"{idx}. {filename}")
    
    return "\n".join(formatted)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def calculate_token_count(text: str, avg_chars_per_token: int = 4) -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Input text
        avg_chars_per_token: Average characters per token
        
    Returns:
        Estimated token count
    """
    return len(text) // avg_chars_per_token
