"""
CSV data validation module.
Validates scraped watch data for completeness and quality.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from omegaconf import DictConfig

from .core.models import ValidationResult

logger = logging.getLogger(__name__)


class WatchDataValidator:
    """Validates CSV files for data quality and completeness."""
    
    def __init__(self, config: DictConfig):
        """Initialize validator with configuration."""
        validation_config = config.get("validation", {})
        
        self.data_dir = Path(validation_config.get("data_dir", "data/watches"))
        self.min_rows = validation_config.get("min_rows", 100)
        self.move_invalid = validation_config.get("move_invalid", False)
        self.log_dir = Path(validation_config.get("log_dir", "logs"))
        
        # Session tracking
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"validation_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # KIV directory for invalid files
        if self.move_invalid:
            self.kiv_dir = self.data_dir / "kiv"
            self.kiv_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_file(self, filepath: Path) -> ValidationResult:
        """
        Validate a single CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            ValidationResult object
        """
        try:
            # Read file
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Basic checks
            if len(rows) < 2:
                return ValidationResult(
                    filename=filepath.name,
                    is_valid=False,
                    row_count=0,
                    error_message="Empty or header-only file"
                )
            
            row_count = len(rows) - 1  # Exclude header
            
            # Check date ordering
            date_ascending = self._check_date_order(rows[1:])  # Skip header
            
            # Determine validity
            has_sufficient_rows = row_count >= self.min_rows
            is_valid = has_sufficient_rows and date_ascending
            
            # Build error message
            error_parts = []
            if not has_sufficient_rows:
                error_parts.append(f"Insufficient rows: {row_count} < {self.min_rows}")
            if not date_ascending:
                error_parts.append("Date ordering issue")
            
            error_message = "; ".join(error_parts) if error_parts else None
            
            return ValidationResult(
                filename=filepath.name,
                is_valid=is_valid,
                row_count=row_count,
                error_message=error_message,
                has_date_issues=not date_ascending
            )
            
        except Exception as e:
            return ValidationResult(
                filename=filepath.name,
                is_valid=False,
                row_count=0,
                error_message=f"Error reading file: {str(e)}"
            )
    
    def validate_all(self) -> Dict[str, List[ValidationResult]]:
        """
        Validate all CSV files in the data directory.
        
        Returns:
            Dictionary with validation results categorized
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning("No CSV files found to validate")
            return {
                "valid": [],
                "invalid": [],
                "errors": []
            }
        
        logger.info(f"Validating {len(csv_files)} CSV files...")
        logger.info(f"Minimum rows required: {self.min_rows}")
        logger.info("="*60)
        
        results = {
            "valid": [],
            "invalid": [],
            "errors": []
        }
        
        for csv_file in sorted(csv_files):
            result = self.validate_file(csv_file)
            
            # Log result
            if result.error_message and "Error reading" in result.error_message:
                results["errors"].append(result)
                logger.error(f"❌ ERROR: {result.filename} - {result.error_message}")
            elif result.is_valid:
                results["valid"].append(result)
                logger.info(f"✅ VALID: {result.filename} ({result.row_count} rows)")
            else:
                results["invalid"].append(result)
                if result.has_date_issues:
                    logger.warning(f"⚠️ DATE ISSUE: {result.filename} ({result.row_count} rows)")
                else:
                    logger.warning(f"⚠️ INSUFFICIENT: {result.filename} ({result.row_count} rows)")
            
            # Move invalid files if configured
            if self.move_invalid and not result.is_valid:
                self._move_invalid_file(csv_file, result.error_message)
        
        self._print_summary(results)
        self._save_report(results)
        
        return results
    
    def _check_date_order(self, data_rows: List[List[str]]) -> bool:
        """Check if dates are in ascending order."""
        if len(data_rows) < 2:
            return True
        
        try:
            # Extract dates (assuming first column)
            dates = [row[0] for row in data_rows]
            dates_pd = pd.to_datetime(dates, errors="coerce")
            
            if dates_pd.isnull().any():
                return False
            
            return dates_pd.is_monotonic_increasing
            
        except Exception:
            return True  # Assume OK if can't parse
    
    def _move_invalid_file(self, filepath: Path, reason: Optional[str]) -> None:
        """Move invalid file to KIV directory."""
        try:
            dest_path = self.kiv_dir / filepath.name
            
            # Handle conflicts
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = self.kiv_dir / f"{stem}_{timestamp}{suffix}"
            
            filepath.rename(dest_path)
            
            # Create metadata file
            metadata_path = dest_path.with_suffix(".info")
            metadata_path.write_text(
                f"Original: {filepath.name}\n"
                f"Moved: {datetime.now()}\n"
                f"Reason: {reason or 'No specific reason provided'}\n"
                f"Session: {self.timestamp}\n"
            )
            
            logger.info(f"📁 Moved {filepath.name} to KIV")
            
        except Exception as e:
            logger.error(f"Failed to move {filepath.name}: {e}")
    
    def _print_summary(self, results: Dict[str, List[ValidationResult]]) -> None:
        """Print validation summary."""
        valid_count = len(results["valid"])
        invalid_count = len(results["invalid"])
        error_count = len(results["errors"])
        total = valid_count + invalid_count + error_count
        
        # Count specific issues
        date_issues = sum(1 for r in results["invalid"] if r.has_date_issues)
        insufficient = invalid_count - date_issues
        
        logger.info("\n" + "="*60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {total}")
        logger.info(f"✅ Valid: {valid_count}")
        logger.info(f"⚠️ Invalid: {invalid_count}")
        logger.info(f"  - Date issues: {date_issues}")
        logger.info(f"  - Insufficient rows: {insufficient}")
        logger.info(f"❌ Errors: {error_count}")
        
        if total > 0:
            success_rate = (valid_count / total) * 100
            logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        logger.info(f"\n📁 Session logs: {self.session_dir}")
    
    def _save_report(self, results: Dict[str, List[ValidationResult]]) -> None:
        """Save validation report to file."""
        report_file = self.session_dir / "validation_report.txt"
        
        with open(report_file, "w") as f:
            f.write(f"Validation Report - {self.timestamp}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Data directory: {self.data_dir}\n")
            f.write(f"  Minimum rows: {self.min_rows}\n\n")
            
            # Valid files
            f.write(f"VALID FILES ({len(results['valid'])}):\n")
            for result in results["valid"]:
                f.write(f"  - {result.filename} ({result.row_count} rows)\n")
            
            # Invalid files
            f.write(f"\nINVALID FILES ({len(results['invalid'])}):\n")
            for result in results["invalid"]:
                f.write(f"  - {result.filename}: {result.error_message}\n")
            
            # Error files
            f.write(f"\nERROR FILES ({len(results['errors'])}):\n")
            for result in results["errors"]:
                f.write(f"  - {result.filename}: {result.error_message}\n")
        
        logger.info(f"Report saved to: {report_file}")
    
    def get_invalid_watches(self, results: Dict[str, List[ValidationResult]]) -> List[Dict]:
        """
        Get list of watches that need re-scraping.
        
        Returns:
            List of watch information dictionaries
        """
        invalid_watches = []
        
        for result in results["invalid"] + results["errors"]:
            # Parse filename to extract watch info
            parts = result.filename.replace(".csv", "").split("-")
            
            watch_info = {
                "filename": result.filename,
                "brand": parts[0] if parts else "Unknown",
                "model": "-".join(parts[1:-1]) if len(parts) > 2 else "Unknown",
                "watch_id": parts[-1] if parts else "Unknown",
                "issue": result.error_message,
                "row_count": result.row_count
            }
            invalid_watches.append(watch_info)
        
        return invalid_watches
