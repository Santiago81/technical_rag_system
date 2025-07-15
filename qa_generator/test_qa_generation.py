#!/usr/bin/env python3
"""
Test script for Q&A generation with JSONL output
"""
import asyncio
import json
from pathlib import Path
import logging

from config import QAGeneratorConfig
from qa_generator_local import QAGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_generation():
    """Test Q&A generation with local model"""
    
    # Configuration for testing
    config = QAGeneratorConfig(
        use_local_model=True,
        local_model_url='http://localhost:1234',
        local_model_endpoint='/v1/chat/completions',
        model='local-model',
        db_path='../data/sqlite/manuals.db',  # Path to the actual database
        output_dir='qa_test_output',
        batch_size=2,  # Small batch for testing
        questions_per_chunk=2,  # Few questions for testing
        quality_threshold=0.5,  # Lower threshold for testing
        temperature=0.7,
        delay_between_requests=0.1  # Faster for testing
    )
    
    try:
        # Validate configuration
        config.validate()
        logger.info("Configuration validated successfully")
        
        # Create generator
        generator = QAGenerator(config)
        logger.info("Generator created successfully")
        
        # Test with just a few chunks
        logger.info("Starting test generation...")
        dataset = await generator.generate_dataset(
            num_chunks=3,  # Just 3 chunks for testing
            questions_per_chunk=2
        )
        
        # Save as JSONL
        output_dir = Path(config.output_dir)
        jsonl_file = output_dir / 'qa_dataset_test.jsonl'
        
        logger.info(f"Saving {len(dataset['qa_pairs'])} Q&A pairs to JSONL...")
        with open(jsonl_file, 'w') as f:
            for qa_pair in dataset['qa_pairs']:
                # Format for fine-tuning
                formatted = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful technical documentation assistant."},
                        {"role": "user", "content": qa_pair['question']},
                        {"role": "assistant", "content": qa_pair['answer']}
                    ],
                    "metadata": {
                        "source": qa_pair.get('source_chunk_id', ''),
                        "type": qa_pair.get('type', ''),
                        "difficulty": qa_pair.get('difficulty', ''),
                        "quality_score": qa_pair.get('quality_score', 0.0)
                    }
                }
                f.write(json.dumps(formatted) + '\n')
        
        logger.info(f"Test complete! JSONL saved to: {jsonl_file}")
        logger.info(f"Generated {len(dataset['qa_pairs'])} Q&A pairs")
        
        # Show a sample
        if dataset['qa_pairs']:
            logger.info("\nSample Q&A pair:")
            sample = dataset['qa_pairs'][0]
            logger.info(f"Q: {sample['question']}")
            logger.info(f"A: {sample['answer'][:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_generation())
    exit(0 if success else 1)