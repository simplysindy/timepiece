"""
CSV data validation for scraped watch data.
Consolidated from csv_validator.py with configuration support.
"""

import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class WatchDataValidator:
    """Validates CSV files for sufficient data points and quality."""

    def __init__(self, config: Dict):
        """Initialize validator with configuration."""
        validation_config = config.get('validation', {})
        
        self.data_dir = Path(validation_config.get('data_dir', 'data/watches'))
        self.min_rows = validation_config.get('min_rows', 90)
        self.move_invalid = validation_config.get('move_invalid', False)
        self.log_dir = Path(validation_config.get('log_dir', 'logs'))
        
        # Set up kiv directory for invalid files
        self.kiv_dir = self.data_dir / "kiv"
        
        # Generate timestamp for unique session folder
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"csv_validation_{self.timestamp}"
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create kiv directory if move_invalid is enabled
        if self.move_invalid:
            self.kiv_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def find_csv_files(self) -> List[Path]:
        """Find all CSV files in the data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        return csv_files

    def validate_csv_file(self, csv_file: Path) -> Tuple[bool, int, str, bool]:
        """
        Validate a single CSV file.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Tuple of (is_valid, row_count, error_message, is_date_ascending)
        """
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 2:  # Less than header + 1 data row
                    return False, 0, "Empty or header-only file", True
                
                row_count = len(rows) - 1  # Subtract 1 for header
                
                # Check date ordering if we have enough rows
                is_date_ascending = True
                date_error = ""
                
                if row_count >= 2:
                    try:
                        # Assume first column is date
                        first_date = rows[1][0]  # First data row
                        last_date = rows[-1][0]  # Last data row
                        
                        # Try to parse dates in common formats
                        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
                        first_parsed = None
                        last_parsed = None
                        
                        for fmt in date_formats:
                            try:
                                first_parsed = datetime.strptime(first_date, fmt)
                                last_parsed = datetime.strptime(last_date, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if first_parsed and last_parsed:
                            is_date_ascending = first_parsed <= last_parsed
                            if not is_date_ascending:
                                date_error = " (DESCENDING ORDER)"
                    
                    except Exception:
                        # If date parsing fails, assume it's okay
                        pass
                
                # Combine validation checks
                has_sufficient_rows = row_count >= self.min_rows
                
                if not has_sufficient_rows and not is_date_ascending:
                    error_msg = f"Insufficient data: {row_count} < {self.min_rows}{date_error}"
                elif not has_sufficient_rows:
                    error_msg = f"Insufficient data: {row_count} < {self.min_rows}"
                elif not is_date_ascending:
                    error_msg = f"Date ordering issue{date_error}"
                else:
                    error_msg = ""
                
                is_valid = has_sufficient_rows and is_date_ascending
                
                return is_valid, row_count, error_msg, is_date_ascending
        
        except Exception as e:
            return False, 0, f"Error reading file: {str(e)}", True

    def extract_watch_info(self, filename: str) -> Dict[str, str]:
        """
        Extract watch information from filename.
        Expected format: {Brand}-{Model}-{ID}.csv
        """
        try:
            # Remove .csv extension
            name = filename.replace(".csv", "")
            
            # Split by hyphens to extract components
            parts = name.split("-")
            
            if len(parts) >= 3:
                brand = parts[0]
                # ID is typically the last part
                watch_id = parts[-1]
                # Model is everything in between
                model = "-".join(parts[1:-1])
                
                return {
                    "brand": brand,
                    "model": model,
                    "watch_id": watch_id,
                    "filename": filename,
                }
            else:
                return {
                    "brand": "Unknown",
                    "model": name,
                    "watch_id": "Unknown",
                    "filename": filename,
                }
        
        except Exception:
            return {
                "brand": "Unknown",
                "model": filename,
                "watch_id": "Unknown",
                "filename": filename,
            }

    def validate_all_files(self) -> Dict:
        """
        Validate all CSV files and return comprehensive results.
        
        Returns:
            Dictionary with validation results and statistics
        """
        csv_files = self.find_csv_files()
        
        if not csv_files:
            logger.warning("No CSV files found to validate!")
            return {
                "total_files": 0,
                "valid_files": [],
                "invalid_files": [],
                "error_files": [],
                "date_order_issues": [],
            }
        
        valid_files = []
        invalid_files = []
        error_files = []
        date_order_issues = []
        
        logger.info(
            f"Validating CSV files (minimum {self.min_rows} rows required, ascending date order)..."
        )
        logger.info("=" * 80)
        
        for csv_file in sorted(csv_files):
            is_valid, row_count, error_msg, is_date_ascending = self.validate_csv_file(csv_file)
            watch_info = self.extract_watch_info(csv_file.name)
            
            file_info = {
                "file": csv_file.name,
                "path": str(csv_file),
                "row_count": row_count,
                "brand": watch_info["brand"],
                "model": watch_info["model"],
                "watch_id": watch_info["watch_id"],
                "error": error_msg,
                "is_date_ascending": is_date_ascending,
            }
            
            if error_msg and "Error reading file" in error_msg:
                error_files.append(file_info)
                logger.error(f"[ERROR] {csv_file.name} - {error_msg}")
            elif is_valid:
                valid_files.append(file_info)
                logger.info(
                    f"[OK] VALID: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']}"
                )
            else:
                invalid_files.append(file_info)
                if not is_date_ascending:
                    date_order_issues.append(file_info)
                    logger.warning(
                        f"[DATE] DATE ORDER: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']} - {error_msg}"
                    )
                else:
                    logger.warning(
                        f"[WARN] INSUFFICIENT: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']}"
                    )
        
        return {
            "total_files": len(csv_files),
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "error_files": error_files,
            "date_order_issues": date_order_issues,
        }

    def print_validation_summary(self, results: Dict) -> None:
        """Print comprehensive validation summary."""
        total = results["total_files"]
        valid_count = len(results["valid_files"])
        invalid_count = len(results["invalid_files"])
        error_count = len(results["error_files"])
        date_order_count = len(results["date_order_issues"])
        
        logger.info("=" * 80)
        logger.info("[STATS] CSV VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total CSV files: {total}")
        logger.info(f"[OK] Valid files (>={self.min_rows} rows, ascending dates): {valid_count}")
        logger.info(f"[WARN] Invalid files: {invalid_count}")
        logger.info(f"   [DATE] Date order issues (descending): {date_order_count}")
        logger.info(f"   [STATS] Insufficient data (<{self.min_rows} rows): {invalid_count - date_order_count}")
        logger.info(f"[ERROR] Error files: {error_count}")
        
        if total > 0:
            success_rate = valid_count / total * 100
            logger.info(f"[CHART] Success rate: {success_rate:.1f}%")
        
        logger.info(f"\\n[FOLDER] Session files saved to: {self.session_dir}")

    def move_invalid_file(self, file_path: Path, reason: str) -> bool:
        """
        Move an invalid file to the kiv directory.
        
        Args:
            file_path: Path to the file to move
            reason: Reason for moving the file
            
        Returns:
            True if moved successfully, False otherwise
        """
        if not self.move_invalid:
            return False
        
        try:
            # Create destination path
            dest_path = self.kiv_dir / file_path.name
            
            # Handle filename conflicts by adding timestamp
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = self.kiv_dir / f"{stem}_{timestamp}{suffix}"
            
            # Move the file
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"[FOLDER] MOVED: {file_path.name} -> kiv/ ({reason})")
            
            # Create a metadata file alongside it
            metadata_path = dest_path.with_suffix(".info")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(f"Original file: {file_path.name}\\n")
                f.write(f"Moved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Reason: {reason}\\n")
                f.write(f"Validation session: {self.timestamp}\\n")
            
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to move {file_path.name} to kiv: {e}")
            return False

    def get_invalid_watches_summary(self, results: Dict) -> List[Dict[str, str]]:
        """
        Get summary of watches that need attention.
        
        Returns:
            List of dictionaries with watch details needing rescrape
        """
        invalid_watches = []
        for file_info in results["invalid_files"]:
            watch_detail = {
                "brand": file_info["brand"],
                "model": file_info["model"],
                "watch_id": file_info["watch_id"],
                "filename": file_info["file"],
                "issue": file_info["error"],
                "row_count": file_info["row_count"],
            }
            invalid_watches.append(watch_detail)
        
        return invalid_watches

    def run_validation(self) -> Dict:
        """
        Run complete validation process.
        
        Returns:
            Validation results dictionary
        """
        logger.info("[SEARCH] CSV DATA VALIDATION STARTED")
        logger.info(f"Directory: {self.data_dir}")
        logger.info(f"Minimum rows required: {self.min_rows}")
        logger.info(f"Move invalid files: {self.move_invalid}")
        if self.move_invalid:
            logger.info(f"KIV directory: {self.kiv_dir}")
        
        try:
            results = self.validate_all_files()
            
            # Move invalid files if enabled
            if self.move_invalid:
                moved_count = 0
                for file_info in results["invalid_files"] + results["error_files"]:
                    file_path = Path(file_info["path"])
                    reason = file_info["error"]
                    if self.move_invalid_file(file_path, reason):
                        moved_count += 1
                
                if moved_count > 0:
                    logger.info(f"[OK] Successfully moved {moved_count} files to {self.kiv_dir}")
            
            self.print_validation_summary(results)
            
            return results
        
        except Exception as e:
            logger.error(f"[ERROR] Validation failed: {e}")
            raise