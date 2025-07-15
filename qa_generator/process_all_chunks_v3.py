#!/usr/bin/env python3
"""
Process ALL chunks from the database with improved rate limiting to avoid 429 errors.
Generates QA pairs for each chunk using OpenAI API.
"""

import asyncio
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import aiofiles
from collections import defaultdict

# Set up path
import sys
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system')

from qa_generator.chunk_manager import ChunkManager
from qa_generator.qa_generator import QAGenerator
from qa_generator.quality_evaluator import QualityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_complete_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Enhanced rate limiter with exponential backoff"""
    def __init__(self, requests_per_second: int = 10):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.consecutive_429s = 0
        self.backoff_until = 0
        
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        
        # Check if we're in backoff period
        if current_time < self.backoff_until:
            wait_time = self.backoff_until - current_time
            logger.info(f"In backoff period, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            current_time = time.time()
        
        # Regular rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def register_429(self):
        """Register a 429 error and calculate backoff"""
        self.consecutive_429s += 1
        backoff_seconds = min(60, 2 ** self.consecutive_429s)  # Exponential backoff, max 60s
        self.backoff_until = time.time() + backoff_seconds
        logger.warning(f"Got 429, backing off for {backoff_seconds}s (consecutive: {self.consecutive_429s})")
    
    def reset_429_counter(self):
        """Reset the 429 counter after successful request"""
        if self.consecutive_429s > 0:
            logger.info("Resetting 429 counter after successful request")
            self.consecutive_429s = 0

class ComprehensiveProcessor:
    def __init__(self, model: str = "gpt-3.5-turbo", batch_size: int = 10, rps: int = 5):
        self.chunk_manager = ChunkManager()
        self.qa_generator = QAGenerator(model=model)
        self.evaluator = QualityEvaluator()
        self.batch_size = batch_size
        self.rate_limiter = RateLimiter(requests_per_second=rps)
        self.processed_chunks = self._load_progress()
        
    def _load_progress(self) -> set:
        """Load previously processed chunk IDs"""
        progress_file = Path("processed_chunks_v3.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_progress(self):
        """Save processed chunk IDs"""
        with open("processed_chunks_v3.json", 'w') as f:
            json.dump(list(self.processed_chunks), f)
    
    def _is_valid_content(self, chunk_text: str) -> bool:
        """Enhanced content validation"""
        if len(chunk_text.strip()) < 200:
            return False
            
        # Skip chunks that are mostly numbers or codes
        words = chunk_text.split()
        if not words:
            return False
            
        # Check for minimum word count
        if len(words) < 30:
            return False
            
        # Skip if too many special characters
        special_char_ratio = sum(1 for c in chunk_text if not c.isalnum() and not c.isspace()) / len(chunk_text)
        if special_char_ratio > 0.3:
            return False
            
        return True
    
    async def _process_single_chunk(self, chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single chunk with enhanced error handling"""
        chunk_id = chunk_data['id']
        
        if chunk_id in self.processed_chunks:
            return []
        
        # Validate content
        if not self._is_valid_content(chunk_data['chunk_text']):
            logger.info(f"Skipping chunk {chunk_id} - invalid content")
            self.processed_chunks.add(chunk_id)
            return []
        
        results = []
        # Define question types to generate
        question_types = [
            ("factual", "basic"),
            ("factual", "intermediate"),
            ("synthesis", "intermediate"),
            ("application", "intermediate"),
            ("analysis", "advanced")
        ]
        
        for reasoning_type, difficulty in question_types:
            try:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                # Generate QA pair
                qa_pair = await self.qa_generator.generate_single_qa(
                    chunk_text=chunk_data['chunk_text'],
                    reasoning_type=reasoning_type,
                    difficulty=difficulty,
                    context=chunk_data
                )
                
                if qa_pair:
                    # Evaluate quality
                    is_valid, score = await self.evaluator.evaluate_single(
                        qa_pair['question'],
                        qa_pair['answer'],
                        chunk_data['chunk_text']
                    )
                    
                    if is_valid and score > 0.6:
                        # Format for fine-tuning
                        formatted = self.format_for_finetuning(qa_pair, chunk_data)
                        if formatted:
                            results.append(formatted)
                            self.rate_limiter.reset_429_counter()
                            
            except Exception as e:
                if "429" in str(e):
                    self.rate_limiter.register_429()
                    # Re-raise to handle at batch level
                    raise
                else:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
                    continue
        
        # Mark as processed
        self.processed_chunks.add(chunk_id)
        return results
    
    def format_for_finetuning(self, qa_pair: Dict[str, Any], chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format QA pair for fine-tuning"""
        try:
            # Filter out generic questions
            generic_patterns = [
                "what is shown", "what can you tell", "describe the", 
                "what information", "based on the text", "according to"
            ]
            
            question_lower = qa_pair['question'].lower()
            if any(pattern in question_lower for pattern in generic_patterns):
                return None
            
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un asistente experto en documentación técnica de sistemas industriales y de control. Proporciona respuestas precisas y detalladas basadas en manuales técnicos."
                    },
                    {
                        "role": "user",
                        "content": qa_pair['question']
                    },
                    {
                        "role": "assistant",
                        "content": qa_pair['answer']
                    }
                ],
                "metadata": {
                    "source_chunk_ids": [chunk_data['id']],
                    "source_pdfs": [chunk_data.get('filename', 'unknown')],
                    "page_numbers": [chunk_data.get('start_page', -1)],
                    "difficulty": qa_pair.get('difficulty', 'unknown'),
                    "reasoning_type": qa_pair.get('reasoning_type', 'unknown'),
                    "quality_score": qa_pair.get('quality_score', 0.0),
                    "chunk_preview": chunk_data['chunk_text'][:200] + "..."
                }
            }
        except Exception as e:
            logger.error(f"Error formatting QA pair: {e}")
            return None
    
    async def process_all_chunks(self, output_file: str = "comprehensive_qa_dataset.jsonl", save_interval: int = 50):
        """Process all chunks with improved error handling"""
        # Get all chunk IDs
        all_chunks = self.chunk_manager.get_all_chunk_ids()
        unprocessed = [cid for cid in all_chunks if cid not in self.processed_chunks]
        
        logger.info(f"Total chunks: {len(all_chunks)}, Unprocessed: {len(unprocessed)}")
        
        examples_generated = 0
        chunks_skipped = 0
        
        async with aiofiles.open(f"qa_dataset/{output_file}", mode='a') as f:
            # Process in batches
            for i in range(0, len(unprocessed), self.batch_size):
                batch_start = time.time()
                batch_chunk_ids = unprocessed[i:i + self.batch_size]
                
                # Get chunk data
                batch_chunks = []
                for chunk_id in batch_chunk_ids:
                    chunk_data = self.chunk_manager.get_chunk_by_id(chunk_id)
                    if chunk_data:
                        batch_chunks.append(chunk_data)
                
                # Filter valid chunks
                valid_chunks = [c for c in batch_chunks if self._is_valid_content(c['chunk_text'])]
                invalid_count = len(batch_chunks) - len(valid_chunks)
                chunks_skipped += invalid_count
                
                if not valid_chunks:
                    logger.info(f"Skipping batch {i//self.batch_size + 1} - no valid chunks")
                    continue
                
                logger.info(f"Processing {len(valid_chunks)} valid chunks from batch of {len(batch_chunks)} (skipped {chunks_skipped} total)")
                
                # Process chunks with retry logic
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Process all chunks in batch concurrently
                        tasks = [self._process_single_chunk(chunk) for chunk in valid_chunks]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Handle results
                        batch_examples = 0
                        for result in results:
                            if isinstance(result, Exception):
                                if "429" in str(result):
                                    raise result  # Re-raise 429 to trigger retry
                                logger.error(f"Error in batch: {result}")
                            elif isinstance(result, list):
                                for example in result:
                                    await f.write(json.dumps(example, ensure_ascii=False) + '\n')
                                    batch_examples += 1
                        
                        examples_generated += batch_examples
                        
                        # Save progress
                        if examples_generated % save_interval == 0:
                            self._save_progress()
                            logger.info(f"Progress saved at {examples_generated} examples")
                        
                        # Success, break retry loop
                        break
                        
                    except Exception as e:
                        if "429" in str(e):
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = 30 * retry_count  # Increasing wait time
                                logger.warning(f"Batch hit rate limit, retry {retry_count}/{max_retries} in {wait_time}s")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"Max retries reached for batch, skipping")
                                # Mark chunks as processed to avoid retrying
                                for chunk in valid_chunks:
                                    self.processed_chunks.add(chunk['id'])
                        else:
                            logger.error(f"Unexpected error in batch: {e}")
                            break
                
                # Log progress
                processed_count = i + len(batch_chunk_ids)
                progress_pct = (processed_count / len(all_chunks)) * 100
                batch_time = time.time() - batch_start
                
                logger.info(f"Progress: {processed_count}/{len(all_chunks)} chunks ({progress_pct:.1f}%)")
                logger.info(f"Examples generated: {examples_generated}")
                logger.info(f"Batch processed in {batch_time:.2f}s")
                logger.info(f"Chunks skipped (too short): {chunks_skipped}")
                
                # Small delay between batches
                await asyncio.sleep(2)
        
        # Final save
        self._save_progress()
        logger.info(f"Processing complete! Generated {examples_generated} QA pairs from {len(all_chunks)} chunks")

async def main():
    parser = argparse.ArgumentParser(description="Process all chunks comprehensively")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of chunks per batch")
    parser.add_argument("--rps", type=int, default=5, help="Requests per second limit")
    parser.add_argument("--output-file", default="comprehensive_qa_dataset_v3.jsonl", help="Output file name")
    parser.add_argument("--save-interval", type=int, default=50, help="Save progress every N examples")
    
    args = parser.parse_args()
    
    processor = ComprehensiveProcessor(
        model=args.model,
        batch_size=args.batch_size,
        rps=args.rps
    )
    
    await processor.process_all_chunks(
        output_file=args.output_file,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    asyncio.run(main())