#!/usr/bin/env python3
"""
Test aggregated QA generation with 15 chunks to demonstrate the full workflow
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up path
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system')
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system/qa_generator')

from chunk_manager import ChunkManager
from process_all_chunks_v4 import AggregatedProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_15chunks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_with_15_chunks():
    """Test the complete aggregated workflow with 15 chunks"""
    
    start_time = time.time()
    logger.info("Starting test with 15 chunks...")
    
    # Initialize components
    processor = AggregatedProcessor(
        model="gpt-4o-mini",
        batch_size=5,  # Process 5 chunks at a time
        rps=3  # Lower rate limit for testing
    )
    
    # Get first 15 valid chunks
    db_path = "/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db"
    chunk_manager = ChunkManager(db_path=db_path)
    all_chunk_ids = chunk_manager.get_all_chunk_ids()
    
    logger.info(f"Total chunk IDs found: {len(all_chunk_ids)}")
    
    valid_chunks = []
    invalid_count = 0
    
    # Check more chunks to ensure we get 15 valid ones
    for i, chunk_id in enumerate(all_chunk_ids[:50]):
        chunk_data = chunk_manager.get_chunk_by_id(chunk_id)
        if chunk_data:
            if processor._is_valid_content(chunk_data['chunk_text']):
                valid_chunks.append(chunk_data)
                logger.debug(f"Valid chunk {chunk_id}: {len(chunk_data['chunk_text'])} chars")
                if len(valid_chunks) >= 15:
                    break
            else:
                invalid_count += 1
                logger.debug(f"Invalid chunk {chunk_id}: {len(chunk_data['chunk_text'])} chars")
        else:
            logger.warning(f"Could not retrieve chunk {chunk_id}")
    
    logger.info(f"Checked {i+1} chunks: {len(valid_chunks)} valid, {invalid_count} invalid")
    
    if len(valid_chunks) < 15:
        logger.error(f"Only found {len(valid_chunks)} valid chunks, need 15")
        # Try with relaxed criteria
        logger.info("Trying with relaxed content validation...")
        valid_chunks = []
        for chunk_id in all_chunk_ids[:30]:
            chunk_data = chunk_manager.get_chunk_by_id(chunk_id)
            if chunk_data and len(chunk_data['chunk_text'].strip()) > 100:  # Relaxed criteria
                valid_chunks.append(chunk_data)
                if len(valid_chunks) >= 15:
                    break
        
        if len(valid_chunks) < 15:
            logger.error(f"Still only found {len(valid_chunks)} chunks with relaxed criteria")
            return
    
    logger.info(f"Found 15 valid chunks from PDFs: {set(c.get('filename', 'unknown') for c in valid_chunks)}")
    
    # Test output file
    output_file = Path("qa_test_output/test_15chunks_results.jsonl")
    output_file.parent.mkdir(exist_ok=True)
    
    # Statistics tracking
    stats = {
        "total_chunks": 15,
        "single_chunk_qa": 0,
        "multi_chunk_qa": 0,
        "aggregations_performed": 0,
        "examples_by_type": {},
        "processing_time": 0,
        "chunks_by_pdf": {}
    }
    
    # Process chunks in groups of 5 (simulating the batch processing)
    all_examples = []
    chunk_buffer = []
    
    with open(output_file, 'w') as f:
        for i in range(0, 15, 5):
            batch_chunks = valid_chunks[i:i+5]
            logger.info(f"\n=== Processing batch {i//5 + 1}/3 (chunks {i+1}-{i+len(batch_chunks)}) ===")
            
            # Process individual chunks
            for chunk in batch_chunks:
                logger.info(f"Processing chunk {chunk['id']} from {chunk.get('filename', 'unknown')}")
                
                # Generate single-chunk QAs
                question_types = [
                    ("factual", "basic"),
                    ("factual", "intermediate"),
                    ("application", "intermediate"),
                ]
                
                for reasoning_type, difficulty in question_types:
                    try:
                        qa = await processor.qa_generator.generate_single_qa(
                            chunk_text=chunk['chunk_text'],
                            reasoning_type=reasoning_type,
                            difficulty=difficulty,
                            context=chunk
                        )
                        
                        if qa:
                            # Evaluate quality
                            is_valid, score = await processor.evaluator.evaluate_single(
                                qa['question'],
                                qa['answer'],
                                chunk['chunk_text']
                            )
                            
                            if is_valid and score > 0.6:
                                formatted = processor.format_for_finetuning(qa, chunk)
                                if formatted:
                                    f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                                    all_examples.append(formatted)
                                    stats["single_chunk_qa"] += 1
                                    
                                    # Track by type
                                    qa_type = f"{reasoning_type}_{difficulty}"
                                    stats["examples_by_type"][qa_type] = stats["examples_by_type"].get(qa_type, 0) + 1
                        
                        # Rate limiting
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error generating {reasoning_type} QA: {e}")
                
                # Add to buffer for aggregated analysis
                chunk_buffer.append(chunk)
                
                # Track chunks by PDF
                pdf_name = chunk.get('filename', 'unknown')
                stats["chunks_by_pdf"][pdf_name] = stats["chunks_by_pdf"].get(pdf_name, 0) + 1
            
            # Perform aggregated analysis every 5 chunks
            if len(chunk_buffer) >= 5:
                logger.info(f"\n--- Performing aggregated analysis on {len(chunk_buffer)} chunks ---")
                
                # Generate aggregated questions
                aggregated_results = await processor._generate_aggregated_questions(chunk_buffer)
                stats["aggregations_performed"] += 1
                
                for agg_qa in aggregated_results:
                    formatted = processor.format_for_finetuning(agg_qa, {}, is_multi_chunk=True)
                    if formatted:
                        f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                        all_examples.append(formatted)
                        stats["multi_chunk_qa"] += 1
                        
                        # Track aggregated types
                        agg_type = f"multi_chunk_{agg_qa.get('aggregation_type', 'unknown')}"
                        stats["examples_by_type"][agg_type] = stats["examples_by_type"].get(agg_type, 0) + 1
                
                logger.info(f"Generated {len(aggregated_results)} multi-chunk QA pairs")
                
                # Clear buffer
                chunk_buffer = []
                
            # Small delay between batches
            await asyncio.sleep(2)
    
    # Calculate final statistics
    end_time = time.time()
    stats["processing_time"] = end_time - start_time
    stats["total_examples"] = len(all_examples)
    stats["examples_per_chunk"] = stats["total_examples"] / 15
    
    # Save statistics
    stats_file = Path("qa_test_output/test_15chunks_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save sample of generated QAs for review
    samples = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model": "gpt-4o-mini",
            "chunks_processed": 15,
            "aggregate_interval": 5
        },
        "statistics": stats,
        "sample_single_chunk_qa": [],
        "sample_multi_chunk_qa": []
    }
    
    # Extract samples
    for example in all_examples[:10]:
        if example["metadata"].get("is_multi_chunk"):
            samples["sample_multi_chunk_qa"].append({
                "question": example["messages"][1]["content"],
                "answer": example["messages"][2]["content"][:200] + "...",
                "metadata": example["metadata"]
            })
        else:
            samples["sample_single_chunk_qa"].append({
                "question": example["messages"][1]["content"],
                "answer": example["messages"][2]["content"][:200] + "...",
                "metadata": {
                    "chunk_id": example["metadata"]["source_chunk_ids"][0],
                    "pdf": example["metadata"]["source_pdfs"][0],
                    "type": example["metadata"]["reasoning_type"]
                }
            })
    
    samples_file = Path("qa_test_output/test_15chunks_samples.json")
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total chunks processed: {stats['total_chunks']}")
    logger.info(f"Total QA pairs generated: {stats['total_examples']}")
    logger.info(f"  - Single-chunk QA: {stats['single_chunk_qa']}")
    logger.info(f"  - Multi-chunk QA: {stats['multi_chunk_qa']}")
    logger.info(f"Aggregations performed: {stats['aggregations_performed']}")
    logger.info(f"Average QA per chunk: {stats['examples_per_chunk']:.2f}")
    logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
    logger.info(f"\nQA types generated:")
    for qa_type, count in stats["examples_by_type"].items():
        logger.info(f"  - {qa_type}: {count}")
    logger.info(f"\nChunks by PDF:")
    for pdf, count in stats["chunks_by_pdf"].items():
        logger.info(f"  - {pdf}: {count}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - QA Dataset: {output_file}")
    logger.info(f"  - Statistics: {stats_file}")
    logger.info(f"  - Samples: {samples_file}")

if __name__ == "__main__":
    asyncio.run(test_with_15_chunks())