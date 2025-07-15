#!/usr/bin/env python3
"""
Main script to generate Q&A dataset for fine-tuning
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_generator import QAGenerator, ChunkManager
from qa_generator.config import QAGeneratorConfig

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

async def generate_dataset(config: QAGeneratorConfig, args):
    """Main dataset generation function"""
    
    # Initialize generator
    generator = QAGenerator(
        api_key=config.openai_api_key,
        model=config.model,
        db_path=config.db_path,
        output_dir=config.output_dir,
        domain=config.domain,
        temperature=config.temperature
    )
    
    # Get initial statistics
    stats = generator.chunk_manager.get_statistics()
    logger.info(f"Database statistics: {json.dumps(stats, indent=2)}")
    
    if args.mode == 'batch':
        # Generate in batches
        filename = args.output_file  # Will be None if not specified, letting run_generation_pipeline use default
        await generator.run_generation_pipeline(
            total_examples=args.total_examples,
            batch_size=config.batch_size,
            save_interval=args.save_interval,
            filename=filename,
            append=args.append
        )
        
    elif args.mode == 'chunks':
        # Process specific chunks
        chunk_ids = [int(x) for x in args.chunk_ids.split(',')]
        examples = []
        
        for chunk_id in chunk_ids:
            chunk = generator.chunk_manager.get_chunk(chunk_id)
            if chunk:
                logger.info(f"Processing chunk {chunk_id}")
                # Generate different types of questions
                from qa_generator.prompt_templates import QuestionType, DifficultyLevel
                
                for q_type in QuestionType:
                    for difficulty in DifficultyLevel:
                        qa_pairs = await generator.generate_qa_for_chunk(
                            chunk, q_type, difficulty
                        )
                        # Convert to examples...
                        logger.info(f"Generated {len(qa_pairs)} {q_type.value} questions at {difficulty.value} level")
        
        if examples:
            filename = args.output_file or f"chunks_{args.chunk_ids}.jsonl"
            generator.save_dataset(examples, filename, append=args.append)
    
    elif args.mode == 'synthesis':
        # Generate multi-chunk synthesis questions
        examples = await generator.generate_multi_chunk_synthesis(args.n_synthesis)
        filename = args.output_file or "synthesis_questions.jsonl"
        generator.save_dataset(examples, filename, append=args.append)
    
    elif args.mode == 'test':
        # Test mode - generate a small sample
        examples = await generator.process_batch(batch_size=5)
        filename = args.output_file or "test_sample.jsonl"
        generator.save_dataset(examples, filename, append=args.append)
        
        # Display sample
        if examples:
            logger.info("\nSample generated Q&A:")
            sample = examples[0]
            logger.info(f"System: {sample.messages[0]['content'][:100]}...")
            logger.info(f"User: {sample.messages[1]['content']}")
            logger.info(f"Assistant: {sample.messages[2]['content'][:200]}...")
            logger.info(f"Metadata: {json.dumps(sample.metadata, indent=2)}")

def main():
    parser = argparse.ArgumentParser(description='Generate Q&A dataset for fine-tuning')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (overrides env/config)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--db-path', type=str, default='docs.db', help='Path to SQLite database')
    parser.add_argument('--output-dir', type=str, default='qa_dataset', help='Output directory')
    
    # Generation mode
    parser.add_argument('--mode', type=str, 
                       choices=['batch', 'chunks', 'synthesis', 'test'],
                       default='batch',
                       help='Generation mode')
    
    # Batch mode options
    parser.add_argument('--total-examples', type=int, default=1000,
                       help='Total examples to generate (batch mode)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save dataset every N examples')
    
    # Chunk mode options
    parser.add_argument('--chunk-ids', type=str,
                       help='Comma-separated chunk IDs to process')
    
    # Synthesis mode options
    parser.add_argument('--n-synthesis', type=int, default=50,
                       help='Number of synthesis examples to generate')
    
    # Other options
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                       help='Minimum quality score to accept Q&A')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='LLM temperature')
    parser.add_argument('--domain', type=str, default='technical documentation',
                       help='Domain for system prompt')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing dataset file instead of creating new one')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Specific output filename (defaults to mode-specific names)')
    
    # Utility options
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics only')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear processed chunks cache')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load or create configuration
    if args.config:
        config = QAGeneratorConfig.from_file(args.config)
    else:
        config = QAGeneratorConfig()
    
    # Override with command line arguments
    if args.api_key:
        config.openai_api_key = args.api_key
    if args.model:
        config.model = args.model
    if args.db_path:
        config.db_path = args.db_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.quality_threshold:
        config.quality_threshold = args.quality_threshold
    if args.temperature:
        config.temperature = args.temperature
    if args.domain:
        config.domain = args.domain
    
    try:
        # Validate configuration
        config.validate()
        
        # Handle utility commands
        if args.stats:
            chunk_manager = ChunkManager(config.db_path)
            stats = chunk_manager.get_statistics()
            print(json.dumps(stats, indent=2))
            return
        
        if args.clear_cache:
            chunk_manager = ChunkManager(config.db_path)
            chunk_manager.clear_cache()
            logger.info("Cache cleared")
            return
        
        # Log configuration
        logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
        
        # Run generation
        asyncio.run(generate_dataset(config, args))
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()