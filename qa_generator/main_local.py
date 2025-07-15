#!/usr/bin/env python3
"""
Main script for QA generation with local model support
"""
import asyncio
import argparse
import sys
from pathlib import Path
import logging

from config import QAGeneratorConfig
from qa_generator_local import QAGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Generate Q&A dataset from technical documentation')
    
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--db-path', type=str, default='docs.db', help='Path to SQLite database')
    parser.add_argument('--output-dir', type=str, default='qa_dataset', help='Output directory')
    parser.add_argument('--num-chunks', type=int, help='Number of chunks to process (default: all)')
    parser.add_argument('--questions-per-chunk', type=int, default=3, help='Questions per chunk')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--quality-threshold', type=float, default=0.7, help='Quality threshold (0-1)')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI API instead of local model')
    parser.add_argument('--local-url', type=str, default='http://localhost:1234', help='Local model URL')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model name')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = QAGeneratorConfig.from_file(args.config)
    else:
        config = QAGeneratorConfig(
            use_local_model=not args.use_openai,
            local_model_url=args.local_url,
            db_path=args.db_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            quality_threshold=args.quality_threshold,
            model=args.model,
            questions_per_chunk=args.questions_per_chunk
        )
    
    try:
        # Validate configuration
        config.validate()
        
        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Using local model: {config.use_local_model}")
        if config.use_local_model:
            logger.info(f"  Local model URL: {config.local_model_url}")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  Database: {config.db_path}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Quality threshold: {config.quality_threshold}")
        
        # Create generator
        generator = QAGenerator(config)
        
        # Generate dataset
        logger.info("Starting Q&A generation...")
        dataset = await generator.generate_dataset(
            num_chunks=args.num_chunks,
            questions_per_chunk=args.questions_per_chunk
        )
        
        logger.info("Q&A generation complete!")
        logger.info(f"Total Q&A pairs generated: {len(dataset['qa_pairs'])}")
        logger.info(f"Statistics: {dataset['metadata']['stats']}")
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())