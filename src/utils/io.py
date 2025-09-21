"""
Unified I/O utilities for the entire watch data pipeline.

This module provides comprehensive file I/O helpers for both data preparation 
and scraping systems, eliminating code duplication.
"""

import json
import logging
import re
import shutil
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


def read_csv_safely(
    file_path: Union[str, Path], **kwargs
) -> Optional[pd.DataFrame]:
    """Read a CSV file and return None instead of raising on failure."""
    file_path = Path(file_path)

    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.debug(f"Safely read CSV: {file_path} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        logger.warning(f"CSV file not found: {file_path}")
        return None
    except Exception as exc:
        logger.error(f"Failed to read CSV {file_path}: {exc}")
        return None


def read_json_file(file_path: Union[str, Path]) -> Any:
    """
    Read a JSON file with error handling.

    Parameters:
    ----------
    file_path : str or Path
        Path to the JSON file

    Returns:
    -------
    Any
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
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully read JSON: {file_path}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to read JSON {file_path}: {str(e)}")


def write_json_file(
    data: Any, file_path: Union[str, Path], indent: int = 2
) -> None:
    """
    Write data to JSON file with error handling.

    Parameters:
    ----------
    data : Any
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
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
        logger.debug(f"Successfully wrote JSON: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to write JSON {file_path}: {str(e)}")


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save JSON data to disk while suppressing exceptions."""
    try:
        write_json_file(data, file_path, indent=indent)
        return True
    except Exception as exc:
        logger.error(f"Failed to save JSON to {file_path}: {exc}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Any]:
    """Load JSON data, returning None instead of raising on error."""
    try:
        return read_json_file(file_path)
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {file_path}")
        return None
    except ValueError as exc:
        logger.error(f"Failed to load JSON from {file_path}: {exc}")
        return None


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
        List of JSON objects loaded from the file. Returns an empty list if
        the file is missing or cannot be parsed.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"JSONL file not found: {file_path}")
        return []

    data: List[Dict[str, Any]] = []

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Invalid JSON on line %s in %s: %s",
                        line_num,
                        file_path,
                        exc,
                    )

        logger.debug(f"Read %d objects from JSONL %s", len(data), file_path)
    except Exception as exc:
        logger.error(f"Failed to read JSONL {file_path}: {exc}")
        return []

    return data


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
        logger.warning(f"File not found: {file_path}")
        return []

    # Try to interpret as JSONL first
    jsonl_data = read_jsonl_file(file_path)
    if jsonl_data:
        logger.debug(
            "Read %d records from JSONL-compatible file %s",
            len(jsonl_data),
            file_path,
        )
        return jsonl_data

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, list):
            logger.debug(
                "Read %d records from JSON array %s",
                len(data),
                file_path,
            )
            return data

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    logger.debug(
                        "Read %d records from JSON dict key '%s' in %s",
                        len(value),
                        key,
                        file_path,
                    )
                    return value

            logger.debug(
                "JSON object in %s converted to single-item list",
                file_path,
            )
            return [data]

        logger.warning(
            "Unexpected JSON structure in %s (type: %s)",
            file_path,
            type(data).__name__,
        )
        return []

    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON from {file_path}: {exc}")
        return []
    except Exception as exc:
        logger.error(f"Failed to read JSON file {file_path}: {exc}")
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
    except Exception as exc:
        logger.error(f"Failed to load CSV {file_path}: {exc}")
        return None

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.debug(f"Loaded {len(df)} rows from {file_path}")
    return df


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
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    
    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup if original file exists
    backup_created = False
    if file_path.exists():
        try:
            shutil.copy2(file_path, backup_path)
            backup_created = True
            logger.debug(f"Created backup: {backup_path}")
        except Exception as exc:
            logger.warning(f"Failed to create backup for {file_path}: {exc}")
    
    # Try to write the new file
    try:
        df.to_csv(file_path, **kwargs)
        logger.debug(f"Successfully wrote CSV: {file_path} ({len(df)} rows)")
        
        # Clean up backup if write succeeded
        if backup_created and backup_path.exists():
            backup_path.unlink()
            
        return True
        
    except Exception as exc:
        logger.error(f"Failed to write CSV {file_path}: {exc}")

        # Restore backup if write failed
        if backup_created and backup_path.exists():
            try:
                shutil.move(backup_path, file_path)
                logger.info(f"Restored backup for {file_path}")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")

        return False


def make_filename_safe(text: str, max_length: int = 200) -> str:
    """Normalize text so it is safe to use as a filename."""
    if not text:
        return "unnamed"

    # Replace filesystem-invalid characters with underscores
    safe = re.sub(r'[<>:"/\\|?*]', "_", text)

    # Strip control characters that might not render on disk
    safe = "".join(ch for ch in safe if ord(ch) >= 32)

    # Collapse whitespace and convert to underscores
    safe = re.sub(r"\s+", " ", safe).strip()
    safe = safe.replace(" ", "_")

    # Trim leading/trailing dots or underscores that can be problematic
    safe = safe.strip("._ ")

    if not safe:
        return "unnamed"

    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._")

    return safe or "unnamed"


def ensure_output_directory(*path_parts: Union[str, Path]) -> Path:
    """Ensure an output directory exists, creating nested directories as needed."""
    if not path_parts:
        raise ValueError("ensure_output_directory requires at least one path component")

    path = Path(path_parts[0])
    for part in path_parts[1:]:
        path /= part

    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def create_individual_output_structure(base_dir: Union[str, Path], processed_subdir: str = "processed", summary_subdir: str = "summary") -> Dict[str, Path]:
    """Create output directory structure for individual watch processing.
    
    Args:
        base_dir: Base output directory
        processed_subdir: Subdirectory name for processed watch files
        summary_subdir: Subdirectory name for summary files
        
    Returns:
        Dict with paths to created directories
    """
    base_path = Path(base_dir)
    
    # Create main directories
    processed_dir = base_path / processed_subdir
    summary_dir = base_path / summary_subdir
    
    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output structure:")
    logger.info(f"  Processed: {processed_dir}")
    logger.info(f"  Summary: {summary_dir}")
    
    return {
        "base": base_path,
        "processed": processed_dir,
        "summary": summary_dir
    }


def save_individual_watch_file(df: pd.DataFrame, watch_id: str, output_dir: Union[str, Path], filename_pattern: str = "{watch_id}.csv", overwrite_existing: bool = True) -> bool:
    """Save individual watch data to its own file.
    
    Args:
        df: DataFrame with processed watch data
        watch_id: Unique watch identifier
        output_dir: Directory to save the file
        filename_pattern: Pattern for filename (should contain {watch_id})
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_path = Path(output_dir)
        filename = filename_pattern.format(watch_id=watch_id)
        file_path = output_path / filename
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and overwrite setting
        if file_path.exists() and not overwrite_existing:
            logger.info(f"Skipping existing file (overwrite disabled): {file_path}")
            return True
        
        # Save with backup functionality
        success = safe_write_csv_with_backup(df, file_path, index=False)
        
        if success:
            logger.debug(f"Saved individual watch file: {file_path}")
            return True
        else:
            logger.error(f"Failed to save individual watch file: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving individual watch file for {watch_id}: {str(e)}")
        return False


def save_watch_summary(summaries: List[Dict], output_dir: Union[str, Path], filename: str = "watch_metadata.csv", overwrite_existing: bool = True) -> bool:
    """Save watch summary metadata to CSV file.
    
    Args:
        summaries: List of summary dictionaries for each watch
        output_dir: Directory to save the summary file
        filename: Name of the summary file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not summaries:
            logger.warning("No summaries to save")
            return False
            
        output_path = Path(output_dir)
        file_path = output_path / filename
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert summaries to DataFrame
        summary_df = pd.DataFrame(summaries)
        
        # Check if file exists and overwrite setting
        if file_path.exists() and not overwrite_existing:
            logger.info(f"Skipping existing summary file (overwrite disabled): {file_path}")
            return True
        
        # Save with backup functionality
        success = safe_write_csv_with_backup(summary_df, file_path, index=False)
        
        if success:
            logger.info(f"Saved watch summary: {file_path} ({len(summaries)} watches)")
            return True
        else:
            logger.error(f"Failed to save watch summary: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving watch summary: {str(e)}")
        return False
