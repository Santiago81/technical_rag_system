#!/usr/bin/env python3
"""
Simple test to verify QA generation works with just 3 chunks
"""

import asyncio
import json
import logging
import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up path
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system')
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system/qa_generator')

from chunk_manager import ChunkManager
from qa_generator import QAGenerator
from quality_evaluator import QualityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_simple():
    """Quick test with 3 chunks"""
    
    # Initialize components
    db_path = "/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db"
    chunk_manager = ChunkManager(db_path=db_path)
    qa_generator = QAGenerator(model="gpt-4o-mini", db_path=db_path)
    evaluator = QualityEvaluator()
    
    # Get 3 valid chunks
    all_chunk_ids = chunk_manager.get_all_chunk_ids()[:10]
    valid_chunks = []
    
    for chunk_id in all_chunk_ids:
        chunk_data = chunk_manager.get_chunk_by_id(chunk_id)
        if chunk_data and len(chunk_data['chunk_text'].strip()) > 200:
            valid_chunks.append(chunk_data)
            if len(valid_chunks) >= 3:
                break
    
    if len(valid_chunks) < 3:
        logger.error("Could not find 3 valid chunks")
        return
    
    logger.info(f"Testing with {len(valid_chunks)} chunks")
    
    # Test single chunk generation
    logger.info("\n=== Testing Single Chunk QA ===")
    chunk = valid_chunks[0]
    logger.info(f"Chunk ID: {chunk['id']}, Length: {len(chunk['chunk_text'])} chars")
    logger.info(f"Preview: {chunk['chunk_text'][:100]}...")
    
    qa = await qa_generator.generate_single_qa(
        chunk_text=chunk['chunk_text'],
        reasoning_type="factual",
        difficulty="intermediate",
        context=chunk
    )
    
    if qa:
        logger.info(f"\nGenerated QA:")
        logger.info(f"Q: {qa['question']}")
        logger.info(f"A: {qa['answer'][:200]}...")
        
        # Evaluate
        is_valid, score = await evaluator.evaluate_single(
            qa['question'],
            qa['answer'],
            chunk['chunk_text']
        )
        logger.info(f"Valid: {is_valid}, Score: {score:.2f}")
    else:
        logger.error("Failed to generate QA")
    
    # Test multi-chunk generation
    logger.info("\n=== Testing Multi-Chunk QA ===")
    context = {
        "chunk_count": 3,
        "source_pdfs": list(set(c.get('filename', 'unknown') for c in valid_chunks)),
        "page_range": {"start": 1, "end": 10},
        "topics": ["technical documentation"],
        "relationships": ["sequential"]
    }
    
    multi_qa = await qa_generator.generate_multi_chunk_qa(
        chunks=valid_chunks,
        reasoning_type="synthesis",
        difficulty="advanced",
        context=context
    )
    
    if multi_qa:
        logger.info(f"\nGenerated Multi-Chunk QA:")
        logger.info(f"Q: {multi_qa['question']}")
        logger.info(f"A: {multi_qa['answer'][:200]}...")
        logger.info(f"Source chunks: {multi_qa['source_chunk_ids']}")
    else:
        logger.error("Failed to generate multi-chunk QA")
    
    logger.info("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_simple())