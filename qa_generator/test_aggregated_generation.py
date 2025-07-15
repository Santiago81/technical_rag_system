#!/usr/bin/env python3
"""
Test script for aggregated QA generation functionality
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Set up path
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system')

from qa_generator.chunk_manager import ChunkManager
from qa_generator.qa_generator import QAGenerator
from qa_generator.quality_evaluator import QualityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_aggregated_generation():
    """Test the aggregated QA generation with a small sample"""
    
    # Initialize components
    chunk_manager = ChunkManager()
    qa_generator = QAGenerator(model="gpt-3.5-turbo")
    evaluator = QualityEvaluator()
    
    # Get a small sample of chunks
    chunk_ids = chunk_manager.get_all_chunk_ids()[:10]  # Get first 10 chunks
    chunks = []
    
    for chunk_id in chunk_ids:
        chunk_data = chunk_manager.get_chunk_by_id(chunk_id)
        if chunk_data and len(chunk_data['chunk_text']) > 200:
            chunks.append(chunk_data)
    
    if len(chunks) < 5:
        logger.error("Not enough valid chunks for testing")
        return
    
    logger.info(f"Testing with {len(chunks)} chunks")
    
    # Test 1: Single chunk QA generation
    logger.info("\n=== Testing Single Chunk QA Generation ===")
    single_qa = await qa_generator.generate_single_qa(
        chunk_text=chunks[0]['chunk_text'],
        reasoning_type="factual",
        difficulty="intermediate",
        context=chunks[0]
    )
    
    if single_qa:
        logger.info(f"Question: {single_qa['question']}")
        logger.info(f"Answer: {single_qa['answer'][:200]}...")
        
        # Evaluate quality
        is_valid, score = await evaluator.evaluate_single(
            single_qa['question'],
            single_qa['answer'],
            chunks[0]['chunk_text']
        )
        logger.info(f"Quality valid: {is_valid}, Score: {score:.2f}")
    
    # Test 2: Multi-chunk synthesis
    logger.info("\n=== Testing Multi-Chunk Synthesis ===")
    context = {
        "chunk_count": len(chunks[:3]),
        "source_pdfs": list(set(c.get('filename', 'unknown') for c in chunks[:3])),
        "page_range": {
            "start": min(c.get('start_page', 0) for c in chunks[:3]),
            "end": max(c.get('end_page', 0) for c in chunks[:3])
        },
        "topics": ["control systems", "sensors"],
        "relationships": ["sequential", "same_document"]
    }
    
    multi_qa = await qa_generator.generate_multi_chunk_qa(
        chunks=chunks[:3],
        reasoning_type="synthesis",
        difficulty="advanced",
        context=context
    )
    
    if multi_qa:
        logger.info(f"Multi-chunk Question: {multi_qa['question']}")
        logger.info(f"Multi-chunk Answer: {multi_qa['answer'][:200]}...")
        logger.info(f"Source chunks: {multi_qa['source_chunk_ids']}")
    
    # Test 3: Comparison QA
    logger.info("\n=== Testing Comparison QA ===")
    comparison_qa = await qa_generator.generate_comparison_qa(
        chunks=chunks[:2],
        context=context
    )
    
    if comparison_qa:
        logger.info(f"Comparison Question: {comparison_qa['question']}")
        logger.info(f"Comparison Answer: {comparison_qa['answer'][:200]}...")
    
    # Test 4: Comprehensive Analysis
    logger.info("\n=== Testing Comprehensive Analysis ===")
    analysis_qa = await qa_generator.generate_comprehensive_analysis(
        chunks=chunks[:5],
        context=context
    )
    
    if analysis_qa:
        logger.info(f"Analysis Question: {analysis_qa['question']}")
        logger.info(f"Analysis Answer: {analysis_qa['answer'][:200]}...")
    
    # Save test results
    test_results = {
        "single_qa": single_qa,
        "multi_qa": multi_qa,
        "comparison_qa": comparison_qa,
        "analysis_qa": analysis_qa
    }
    
    output_file = Path("qa_test_output/aggregated_test_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nTest results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(test_aggregated_generation())