"""
Main entry point for watch scraping pipeline.
Integrates discovery, scraping, and validation with Hydra configuration management.
"""

import logging
import os
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from ..utils.io import read_mixed_json_file
from .core.base_scraper import WatchScraper, WatchTarget
from .discovery import WatchDiscovery
from .validation import WatchDataValidator

logger = logging.getLogger(__name__)


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration."""
    logging_config = config.get("logging", {})
    level = getattr(logging, logging_config.get("level", "INFO").upper())
    format_str = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(level=level, format=format_str, datefmt="%Y-%m-%d %H:%M:%S")


def load_or_discover_targets(config: DictConfig) -> List[WatchTarget]:
    """Load existing targets or discover new ones."""
    discovery_config = config.discovery
    output_file = discovery_config.output_file

    if os.path.exists(output_file):
        logger.info(f"Loading existing targets from {output_file}")
        targets_data = read_mixed_json_file(output_file)

        # Convert to WatchTarget objects
        targets = []
        for target in targets_data:
            # Extract watch_id from model_name if it contains ID format
            watch_id = ""
            model_name = target["model_name"]

            if " - " in model_name and model_name.split(" - ")[0].isdigit():
                watch_id = model_name.split(" - ")[0]
                clean_model_name = model_name.split(" - ", 1)[1]
            else:
                # Extract from URL if no ID in model_name
                scraper = WatchScraper(config)
                watch_id = scraper.extract_watch_id_from_url(target["url"])
                clean_model_name = model_name

            # Create filename-safe model name
            scraper = WatchScraper(config)
            safe_model_name = scraper.make_filename_safe(clean_model_name)

            watch_target = WatchTarget(
                brand=target["brand"],
                model_name=safe_model_name,
                url=target["url"],
                watch_id=watch_id,
            )
            targets.append(watch_target)

        logger.info(f"Loaded {len(targets)} existing watch targets")
        return targets

    logger.info("No existing targets found, running discovery...")
    discovery = WatchDiscovery(config)
    watches_data = discovery.run_discovery()

    # Convert to WatchTarget objects
    targets = []
    for watch_data in watches_data:
        watch_target = WatchTarget(
            brand=watch_data["brand"],
            model_name=watch_data["model_name"],
            url=watch_data["url"],
            watch_id=watch_data.get("watch_id", "unknown"),
        )
        targets.append(watch_target)

    return targets


def run_scraping_pipeline(
    config: DictConfig, targets: List[WatchTarget]
) -> Dict[str, bool]:
    """Run the main scraping pipeline."""
    logger.info("🚀 STARTING WATCH SCRAPING PIPELINE")
    logger.info(f"Total targets: {len(targets)}")

    scraper = WatchScraper(config)
    results = scraper.scrape_watches_batch(targets)

    # Print summary
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful

    logger.info("\\n" + "=" * 60)
    logger.info("🎯 SCRAPING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total watches processed: {len(results)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    if len(results) > 0:
        success_rate = successful / len(results) * 100
        logger.info(f"📈 Success rate: {success_rate:.1f}%")

    return results


def run_validation_pipeline(config: DictConfig) -> Dict:
    """Run the validation pipeline."""
    logger.info("🔍 STARTING VALIDATION PIPELINE")

    validator = WatchDataValidator(config)
    results = validator.run_validation()

    # Print summary of invalid watches
    invalid_watches = validator.get_invalid_watches_summary(results)
    if invalid_watches:
        logger.info(f"\\n⚠️  Found {len(invalid_watches)} watches needing attention:")
        for watch in invalid_watches[:5]:  # Show first 5
            logger.info(f"  - {watch['brand']} {watch['model']}: {watch['issue']}")
        if len(invalid_watches) > 5:
            logger.info(f"  ... and {len(invalid_watches) - 5} more")

    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="scraping")
def main(config: DictConfig) -> None:
    """
    Main function for watch scraping pipeline.

    Args:
        config: Hydra configuration object
    """
    # Setup logging
    setup_logging(config)

    logger.info("🎯 WATCH SCRAPING PIPELINE STARTED")
    logger.info("=" * 50)

    # Print pipeline configuration
    pipeline_config = config.pipeline
    logger.info("Pipeline Configuration:")
    logger.info(f"  Discovery: {'ON' if pipeline_config.run_discovery else 'OFF'}")
    logger.info(f"  Scraping: {'ON' if pipeline_config.run_scraping else 'OFF'}")
    logger.info(f"  Validation: {'ON' if pipeline_config.run_validation else 'OFF'}")

    try:
        targets = []

        # Step 1: Discovery (if enabled)
        if pipeline_config.run_discovery:
            logger.info("\\n" + "=" * 50)
            logger.info("STEP 1: DISCOVERY")
            logger.info("=" * 50)

            discovery = WatchDiscovery(config)
            watch_data = discovery.run_discovery()

            # Convert to WatchTarget objects
            for watch in watch_data:
                watch_target = WatchTarget(
                    brand=watch["brand"],
                    model_name=watch["model_name"],
                    url=watch["url"],
                    watch_id=watch.get("watch_id", "unknown"),
                )
                targets.append(watch_target)

        # Step 2: Scraping (if enabled)
        if pipeline_config.run_scraping:
            logger.info("\\n" + "=" * 50)
            logger.info("STEP 2: SCRAPING")
            logger.info("=" * 50)

            # Load targets if we didn't run discovery
            if not targets:
                targets = load_or_discover_targets(config)

            scraping_results = run_scraping_pipeline(config, targets)

            # Log scraping summary
            successful_scrapes = sum(
                1 for success in scraping_results.values() if success
            )
            logger.info(
                f"Scraping completed: {successful_scrapes}/{len(scraping_results)} successful"
            )

        # Step 3: Validation (if enabled)
        if pipeline_config.run_validation:
            logger.info("\\n" + "=" * 50)
            logger.info("STEP 3: VALIDATION")
            logger.info("=" * 50)

            validation_results = run_validation_pipeline(config)

            # Log validation summary
            valid_files = len(validation_results.get("valid_files", []))
            total_files = validation_results.get("total_files", 0)
            if total_files > 0:
                logger.info(
                    f"Validation completed: {valid_files}/{total_files} files valid"
                )

        # Final summary
        logger.info("\\n" + "=" * 60)
        logger.info("🏁 PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info("All enabled pipeline steps completed successfully!")

        # Output directory info
        output_dir = config.scraping.output_dir
        logger.info(f"\\n📁 Output files location: {output_dir}/")
        if os.path.exists(output_dir):
            csv_files = len([f for f in os.listdir(output_dir) if f.endswith(".csv")])
            logger.info(f"📊 Total CSV files: {csv_files}")

    except KeyboardInterrupt:
        logger.warning("\\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
