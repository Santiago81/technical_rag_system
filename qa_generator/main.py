#!/usr/bin/env python3
"""
Main entry point for QA dataset generation
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_generator import QAGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate QA dataset from technical manuals')
    
    # Mode arguments
    parser.add_argument('--mode', choices=['batch', 'interactive', 'synthesis'], 
                        default='batch', help='Generation mode')
    
    # Generation parameters
    parser.add_argument('--total-examples', type=int, default=100,
                        help='Total number of examples to generate')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval for intermediate results')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, 
                        help='Output JSONL filename')
    parser.add_argument('--output-dir', type=str, default='qa_dataset',
                        help='Output directory')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing file')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation')
    
    # Database parameters
    parser.add_argument('--db-path', type=str, 
                        default='/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db',
                        help='Path to SQLite database')
    
    # Other parameters
    parser.add_argument('--domain', type=str, default='technical documentation',
                        help='Domain for the QA dataset')
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                        help='Quality threshold for accepting QA pairs')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize generator
    generator = QAGenerator(
        api_key=api_key,
        model=args.model,
        db_path=args.db_path,
        output_dir=args.output_dir,
        domain=args.domain,
        temperature=args.temperature
    )
    
    # Run generation based on mode
    async def run():
        if args.mode == 'batch':
            await generator.run_generation_pipeline(
                total_examples=args.total_examples,
                batch_size=args.batch_size,
                save_interval=args.save_interval,
                filename=args.output_file,
                append=args.append
            )
        elif args.mode == 'synthesis':
            logger.info("Generating multi-chunk synthesis examples")
            examples = await generator.generate_multi_chunk_synthesis(args.total_examples)
            generator.save_dataset(examples, filename=args.output_file)
        else:
            logger.error(f"Mode {args.mode} not implemented yet")
    
    # Run async function
    asyncio.run(run())
    
    logger.info("Generation completed successfully!")

if __name__ == "__main__":
    main()