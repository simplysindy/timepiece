"""
Unified I/O utilities for the entire watch data pipeline.

This module provides comprehensive file I/O helpers for both data preparation 
and scraping systems, eliminating code duplication.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_file(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read a CSV file with error handling.

    Parameters:
    ----------
    file_path : str or Path
        Path to the CSV file
    **kwargs
        Additional arguments passed to pd.read_csv

    Returns:
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises:
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.debug(f"Successfully read CSV: {file_path} ({len(df)} rows)")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV {file_path}: {str(e)}")


def write_csv_file(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Write a DataFrame to CSV with error handling.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to write
    file_path : str or Path
        Output file path
    **kwargs
        Additional arguments passed to df.to_csv
    """
    file_path = Path(file_path)

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(file_path, **kwargs)
        logger.debug(f"Successfully wrote CSV: {file_path} ({len(df)} rows)")
    except Exception as e:
        raise ValueError(f"Failed to write CSV {file_path}: {str(e)}")


def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a JSON file with error handling.

    Parameters:
    ----------
    file_path : str or Path
        Path to the JSON file

    Returns:
    -------
    Dict[str, Any]
        Loaded JSON data

    Raises:
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.debug(f"Successfully read JSON: {file_path}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to read JSON {file_path}: {str(e)}")


def write_json_file(
    data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2
) -> None:
    """
    Write data to JSON file with error handling.

    Parameters:
    ----------
    data : Dict[str, Any]
        Data to write
    file_path : str or Path
        Output file path
    indent : int
        JSON indentation level
    """
    file_path = Path(file_path)

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        logger.debug(f"Successfully wrote JSON: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to write JSON {file_path}: {str(e)}")


def list_files(
    directory: Union[str, Path], pattern: str = "*", recursive: bool = False
) -> List[Path]:
    """
    List files in a directory matching a pattern.

    Parameters:
    ----------
    directory : str or Path
        Directory to search
    pattern : str
        File pattern (e.g., "*.csv")
    recursive : bool
        Whether to search recursively

    Returns:
    -------
    List[Path]
        List of matching file paths
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if not directory.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return []

    try:
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter to only return files (not directories)
        files = [f for f in files if f.is_file()]

        logger.debug(f"Found {len(files)} files matching '{pattern}' in {directory}")
        return sorted(files)

    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {str(e)}")
        return []


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters:
    ----------
    directory : str or Path
        Directory path

    Returns:
    -------
    Path
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a file.

    Parameters:
    ----------
    file_path : str or Path
        Path to the file

    Returns:
    -------
    Dict[str, Any]
        File information including size, modification time, etc.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {"exists": False}

    stat = file_path.stat()

    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_time": stat.st_mtime,
        "is_file": file_path.is_file(),
        "is_directory": file_path.is_dir(),
        "name": file_path.name,
        "stem": file_path.stem,
        "suffix": file_path.suffix,
        "parent": str(file_path.parent),
    }


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> bool:
    """
    Validate a file path.

    Parameters:
    ----------
    file_path : str or Path
        Path to validate
    must_exist : bool
        Whether the file must exist

    Returns:
    -------
    bool
        True if path is valid
    """
    try:
        file_path = Path(file_path)

        if must_exist and not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False

        # Check if parent directory exists or can be created
        if not file_path.parent.exists():
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    f"Cannot create parent directory for {file_path}: {str(e)}"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Invalid file path {file_path}: {str(e)}")
        return False


# Scraping-specific I/O functions


def read_jsonl_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read a JSONL (JSON Lines) file with error handling.
    
    JSONL format contains one JSON object per line, commonly used for
    streaming data or batch processing.
    
    Parameters:
    ----------
    file_path : str or Path
        Path to the JSONL file
        
    Returns:
    -------
    List[Dict[str, Any]]
        List of JSON objects loaded from the file
        
    Raises:
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file cannot be parsed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue
        
        logger.debug(f"Successfully read JSONL: {file_path} ({len(data)} objects)")
        return data
    except Exception as e:
        raise ValueError(f"Failed to read JSONL {file_path}: {str(e)}")


def write_jsonl_file(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Write data to JSONL (JSON Lines) file with error handling.
    
    Parameters:
    ----------
    data : List[Dict[str, Any]]
        List of JSON-serializable objects to write
    file_path : str or Path
        Output file path
    """
    file_path = Path(file_path)
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for obj in data:
                json.dump(obj, f, ensure_ascii=False)
                f.write('\n')
        
        logger.debug(f"Successfully wrote JSONL: {file_path} ({len(data)} objects)")
    except Exception as e:
        raise ValueError(f"Failed to write JSONL {file_path}: {str(e)}")


def read_mixed_json_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read a JSON file that could be either regular JSON or JSONL format.
    
    This function automatically detects the format and reads accordingly,
    useful for legacy compatibility.
    
    Parameters:
    ----------
    file_path : str or Path
        Path to the JSON file
        
    Returns:
    -------
    List[Dict[str, Any]]
        List of JSON objects
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return []
    
    try:
        # Try JSONL format first (one JSON object per line)
        if file_path.suffix == '.jsonl' or file_path.name.endswith('.jsonl'):
            return read_jsonl_file(file_path)
        
        # Try regular JSON format
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Ensure we return a list
        if isinstance(data, list):
            logger.debug(f"Successfully read JSON array: {file_path} ({len(data)} objects)")
            return data
        else:
            logger.debug(f"Successfully read JSON object: {file_path} (converted to list)")
            return [data]
            
    except json.JSONDecodeError:
        # If regular JSON fails, try JSONL format
        try:
            return read_jsonl_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read {file_path} as both JSON and JSONL: {e}")
            return []
    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}")
        return []


def load_existing_csv_data(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load existing CSV data with proper error handling and date conversion.
    
    Specifically designed for loading scraped watch data with date columns.
    
    Parameters:
    ----------
    file_path : str or Path
        Path to the CSV file
        
    Returns:
    -------
    Optional[pd.DataFrame]
        Loaded DataFrame with date column converted, or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.debug(f"CSV file does not exist: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        if len(df) > 0 and "date" in df.columns:
            # Convert date column to datetime for proper comparison
            df["date"] = pd.to_datetime(df["date"])
            logger.debug(f"Successfully loaded existing CSV: {file_path} ({len(df)} rows)")
            return df
        else:
            logger.warning(f"CSV file has no data or missing date column: {file_path}")
            return None
            
    except Exception as e:
        logger.warning(f"Error loading existing CSV data from {file_path}: {e}")
        return None


def safe_write_csv_with_backup(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """
    Write CSV file with backup and error recovery.
    
    Creates a backup of existing file before writing, and restores on failure.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to write
    file_path : str or Path
        Output file path
    **kwargs
        Additional arguments passed to df.to_csv
        
    Returns:
    -------
    bool
        True if write succeeded, False otherwise
    """
    file_path = Path(file_path)
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    
    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup if original file exists
    backup_created = False
    if file_path.exists():
        try:
            backup_path.write_bytes(file_path.read_bytes())
            backup_created = True
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")
    
    # Try to write the new file
    try:
        df.to_csv(file_path, **kwargs)
        logger.debug(f"Successfully wrote CSV: {file_path} ({len(df)} rows)")
        
        # Clean up backup if write succeeded
        if backup_created and backup_path.exists():
            backup_path.unlink()
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to write CSV {file_path}: {e}")
        
        # Restore backup if write failed
        if backup_created and backup_path.exists():
            try:
                file_path.write_bytes(backup_path.read_bytes())
                backup_path.unlink()
                logger.info(f"Restored backup for {file_path}")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")
        
        return False


def make_filename_safe(text: str, max_length: int = 200) -> str:
    """
    Make a text string safe for use as a filename.
    
    Removes or replaces characters that are invalid in filenames across
    different operating systems.
    
    Parameters:
    ----------
    text : str
        Input text to make filename-safe
    max_length : int
        Maximum length for the resulting filename
        
    Returns:
    -------
    str
        Filename-safe string
    """
    if not text:
        return "unnamed"
    
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    safe_text = text
    
    for char in invalid_chars:
        safe_text = safe_text.replace(char, '_')
    
    # Replace multiple spaces with single underscore
    safe_text = ' '.join(safe_text.split())
    safe_text = safe_text.replace(' ', '_')
    
    # Remove leading/trailing periods and spaces
    safe_text = safe_text.strip('. ')
    
    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length].rstrip('_')
    
    # Ensure we have something
    if not safe_text:
        safe_text = "unnamed"
    
    return safe_text


def ensure_output_directory(base_dir: str, *subdirs: str) -> Path:
    """
    Ensure output directory structure exists, creating nested directories as needed.
    
    Parameters:
    ----------
    base_dir : str
        Base directory path
    *subdirs : str
        Additional subdirectory names to create
        
    Returns:
    -------
    Path
        Path to the final directory
    """
    path = Path(base_dir)
    for subdir in subdirs:
        path = path / subdir
    
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    
    return path
