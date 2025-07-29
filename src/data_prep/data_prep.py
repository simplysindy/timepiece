"""
Main entry point for the simplified watch data preparation pipeline.

This module uses Hydra for configuration management and orchestrates the complete
data preparation process for watch price data.

Usage:
    python -m src.data_prep.data_prep
    python -m src.data_prep.data_prep data.max_files=10
    python -m src.data_prep.data_prep processing.interpolation_method=linear
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .config import DataPrepConfig
from .process import WatchDataProcessor


@hydra.main(version_base=None, config_path="../../conf", config_name="data_prep")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for data preparation pipeline.
    
    Parameters:
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("WATCH DATA PREPARATION PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Convert OmegaConf to our config dataclass
        config = DataPrepConfig(
            data=cfg.data,
            processing=cfg.processing,
            features=cfg.features,
            watch=cfg.watch,
            output=cfg.output
        )
        
        # Log configuration summary
        logger.info(f"Input directory: {config.data.input_dir}")
        logger.info(f"Output directory: {config.data.output_dir}")
        logger.info(f"Max files: {config.data.max_files or 'unlimited'}")
        logger.info(f"Interpolation method: {config.processing.interpolation_method}")
        logger.info(f"Outlier method: {config.processing.outlier_method}")
        logger.info(f"Min data points: {config.processing.min_data_points}")
        logger.info("=" * 60)
        
        # Initialize processor and run pipeline
        processor = WatchDataProcessor(config)
        results = processor.process_all_watches()
        
        # Display results
        if results["success"]:
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"📁 Raw data files: {results['raw_count']}")
            logger.info(f"✅ Processed watches: {results['processed_count']}")
            logger.info(f"🔧 Featured watches: {results['featured_count']}")
            logger.info(f"📊 Combined records: {results['combined_records']}")
            logger.info(f"💾 Output files: {len(results['output_files'])}")
            
            logger.info("\nOutput files created:")
            for file_path in results['output_files']:
                logger.info(f"  📄 {file_path}")
            
            logger.info(f"\nData saved to: {config.data.output_dir}")
            logger.info("✨ Ready for ML model training!")
            
        else:
            logger.error(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return


if __name__ == "__main__":
    main()