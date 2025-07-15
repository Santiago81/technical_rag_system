#!/usr/bin/env python3
"""
Enhanced processor with aggregated analysis every 5 chunks.
Generates QA pairs using single chunks AND multi-chunk synthesis.
"""

import asyncio
import json
import logging
import argparse
import time
import psutil
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set up path
import sys
sys.path.append('/Users/santiagojorda/Downloads/clode_technical_rag_system')

from chunk_manager import ChunkManager
from qa_generator import QAGenerator
from quality_evaluator import QualityEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_complete_v4.log'),
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

class AggregatedProcessor:
    def __init__(self, model: str = "gpt-3.5-turbo", batch_size: int = 3, rps: int = 2):
        db_path = "/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db"
        self.chunk_manager = ChunkManager(db_path=db_path)
        self.qa_generator = QAGenerator(model=model, db_path=db_path)
        self.evaluator = QualityEvaluator()
        self.batch_size = batch_size
        self.rate_limiter = RateLimiter(requests_per_second=rps)
        self.processed_chunks = self._load_progress()
        self.chunk_buffer = []  # Buffer for aggregated analysis
        self.process = psutil.Process(os.getpid())
        
    def _load_progress(self) -> set:
        """Load previously processed chunk IDs"""
        progress_file = Path("processed_chunks_v5.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_progress(self):
        """Save processed chunk IDs"""
        with open("processed_chunks_v5.json", 'w') as f:
            json.dump(list(self.processed_chunks), f)
    
    def _log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage {context}: {memory_mb:.2f} MB")
    
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
    
    async def _generate_aggregated_questions(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate questions that require information from multiple chunks"""
        if len(chunks) < 2:
            return []
        
        results = []
        
        try:
            # Create aggregated context
            aggregated_context = self._create_aggregated_context(chunks)
            
            # Generate synthesis questions
            try:
                await self.rate_limiter.acquire()
                synthesis_qa = await self.qa_generator.generate_multi_chunk_qa(
                    chunks=chunks,
                    reasoning_type="synthesis",
                    difficulty="advanced",
                    context=aggregated_context
                )
                
                if synthesis_qa:
                    results.append(synthesis_qa)
                    logger.info("Generated synthesis question successfully")
                else:
                    logger.warning("Failed to generate synthesis question")
                    
            except Exception as e:
                logger.error(f"Error in synthesis generation: {e}")
                if "429" in str(e):
                    raise
                
            # Generate comparison questions (only if we have at least 2 different chunks)
            if len(chunks) >= 2:
                try:
                    await self.rate_limiter.acquire()
                    comparison_qa = await self.qa_generator.generate_comparison_qa(
                        chunks=chunks[:2],  # Use only first 2 chunks for comparison
                        context=aggregated_context
                    )
                    
                    if comparison_qa:
                        results.append(comparison_qa)
                        logger.info("Generated comparison question successfully")
                    else:
                        logger.warning("Failed to generate comparison question")
                        
                except Exception as e:
                    logger.error(f"Error in comparison generation: {e}")
                    if "429" in str(e):
                        raise
                
            # Generate comprehensive analysis questions
            try:
                await self.rate_limiter.acquire()
                analysis_qa = await self.qa_generator.generate_comprehensive_analysis(
                    chunks=chunks,
                    context=aggregated_context
                )
                
                if analysis_qa:
                    results.append(analysis_qa)
                    logger.info("Generated analysis question successfully")
                else:
                    logger.warning("Failed to generate analysis question")
                    
            except Exception as e:
                logger.error(f"Error in analysis generation: {e}")
                if "429" in str(e):
                    raise
                
            if results:
                self.rate_limiter.reset_429_counter()
            
        except Exception as e:
            if "429" in str(e):
                self.rate_limiter.register_429()
                raise
            else:
                logger.error(f"Error generating aggregated questions: {e}")
                
        return results
    
    def _create_aggregated_context(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create enriched context from multiple chunks"""
        context = {
            "chunk_count": len(chunks),
            "total_content_length": sum(len(c['chunk_text']) for c in chunks),
            "source_pdfs": list(set(c.get('filename', 'unknown') for c in chunks)),
            "page_range": {
                "start": min(c.get('start_page', 0) for c in chunks),
                "end": max(c.get('end_page', 0) for c in chunks)
            },
            "topics": self._extract_topics(chunks),
            "relationships": self._identify_relationships(chunks)
        }
        return context
    
    def _extract_topics(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from chunks (simplified)"""
        # This is a simplified version - in production, you might use NLP
        topics = []
        topic_keywords = {
            "control": "control systems",
            "sensor": "sensors",
            "actuator": "actuators",
            "motor": "motor control",
            "servo": "servo drives",
            "communication": "communication protocols",
            "safety": "safety systems",
            "configuration": "system configuration",
            "parameter": "parameter settings",
            "diagnostic": "diagnostics",
            "error": "error handling",
            "installation": "installation procedures"
        }
        
        for chunk in chunks:
            text = chunk['chunk_text'].lower()
            for keyword, topic in topic_keywords.items():
                if keyword in text and topic not in topics:
                    topics.append(topic)
        
        return topics[:5]  # Limit to 5 most relevant topics
    
    def _identify_relationships(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Identify relationships between chunks"""
        relationships = []
        
        # Check if chunks are sequential
        chunk_ids = [c['id'] for c in chunks]
        if all(chunk_ids[i] + 1 == chunk_ids[i + 1] for i in range(len(chunk_ids) - 1)):
            relationships.append("sequential")
            
        # Check if from same PDF
        pdfs = set(c.get('filename', '') for c in chunks)
        if len(pdfs) == 1:
            relationships.append("same_document")
        else:
            relationships.append("cross_document")
            
        return relationships
    
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
        
        # Add to buffer for aggregated analysis
        self.chunk_buffer.append(chunk_data)
        
        # Limit buffer size to prevent excessive memory usage
        if len(self.chunk_buffer) > 20:
            logger.warning(f"Chunk buffer size ({len(self.chunk_buffer)}) exceeds limit, clearing oldest chunks")
            self.chunk_buffer = self.chunk_buffer[-10:]  # Keep only last 10 chunks
        
        # Mark as processed
        self.processed_chunks.add(chunk_id)
        return results
    
    def format_for_finetuning(self, qa_pair: Dict[str, Any], chunk_data: Dict[str, Any], 
                              is_multi_chunk: bool = False) -> Optional[Dict[str, Any]]:
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
            
            metadata = {
                "difficulty": qa_pair.get('difficulty', 'unknown'),
                "reasoning_type": qa_pair.get('reasoning_type', 'unknown'),
                "quality_score": qa_pair.get('quality_score', 0.0),
                "is_multi_chunk": is_multi_chunk
            }
            
            if is_multi_chunk:
                metadata.update({
                    "source_chunk_ids": qa_pair.get('source_chunk_ids', []),
                    "source_pdfs": qa_pair.get('source_pdfs', []),
                    "page_numbers": qa_pair.get('page_numbers', []),
                    "aggregation_type": qa_pair.get('aggregation_type', 'synthesis')
                })
            else:
                metadata.update({
                    "source_chunk_ids": [chunk_data['id']],
                    "source_pdfs": [chunk_data.get('filename', 'unknown')],
                    "page_numbers": [chunk_data.get('start_page', -1)],
                    "chunk_preview": chunk_data['chunk_text'][:200] + "..."
                })
            
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
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error formatting QA pair: {e}")
            return None
    
    async def process_all_chunks(self, output_file: str = "comprehensive_qa_dataset.jsonl", 
                                save_interval: int = 50, aggregate_interval: int = 5):
        """Process all chunks with aggregated analysis every N chunks"""
        # Get all chunk IDs
        all_chunks = self.chunk_manager.get_all_chunk_ids()
        unprocessed = [cid for cid in all_chunks if cid not in self.processed_chunks]
        
        logger.info(f"Total chunks: {len(all_chunks)}, Unprocessed: {len(unprocessed)}")
        self._log_memory_usage("at start")
        
        examples_generated = 0
        chunks_skipped = 0
        multi_chunk_examples = 0
        
        # Ensure output directory exists
        output_path = Path(f"qa_dataset/{output_file}")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'a') as f:
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
                
                logger.info(f"Processing {len(valid_chunks)} valid chunks from batch of {len(batch_chunks)}")
                
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
                                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
                                    batch_examples += 1
                        
                        examples_generated += batch_examples
                        
                        # Check if we should do aggregated analysis
                        if len(self.chunk_buffer) >= aggregate_interval:
                            logger.info(f"Performing aggregated analysis on {len(self.chunk_buffer)} chunks")
                            self._log_memory_usage("before aggregated analysis")
                            
                            # Generate multi-chunk questions
                            aggregated_results = await self._generate_aggregated_questions(self.chunk_buffer)
                            
                            for agg_qa in aggregated_results:
                                formatted = self.format_for_finetuning(agg_qa, {}, is_multi_chunk=True)
                                if formatted:
                                    f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                                    multi_chunk_examples += 1
                                    examples_generated += 1
                            
                            logger.info(f"Generated {len(aggregated_results)} multi-chunk QA pairs")
                            
                            # Clear buffer
                            self.chunk_buffer = []
                        
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
                processed_count = len(self.processed_chunks)
                progress_pct = (processed_count / len(all_chunks)) * 100
                batch_time = time.time() - batch_start
                
                logger.info(f"Progress: {processed_count}/{len(all_chunks)} chunks ({progress_pct:.1f}%)")
                logger.info(f"Examples generated: {examples_generated} (Single: {examples_generated - multi_chunk_examples}, Multi: {multi_chunk_examples})")
                logger.info(f"Batch processed in {batch_time:.2f}s")
                logger.info(f"Chunks skipped (too short): {chunks_skipped}")
                self._log_memory_usage(f"after batch {i//self.batch_size + 1}")
                
                # Small delay between batches
                await asyncio.sleep(2)
            
            # Process any remaining chunks in buffer
            if self.chunk_buffer:
                logger.info(f"Processing final {len(self.chunk_buffer)} chunks in buffer")
                aggregated_results = await self._generate_aggregated_questions(self.chunk_buffer)
                
                for agg_qa in aggregated_results:
                    formatted = self.format_for_finetuning(agg_qa, {}, is_multi_chunk=True)
                    if formatted:
                        await f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                        multi_chunk_examples += 1
                        examples_generated += 1
        
        # Final save
        self._save_progress()
        logger.info(f"Processing complete!")
        logger.info(f"Total QA pairs generated: {examples_generated}")
        logger.info(f"Single-chunk QA pairs: {examples_generated - multi_chunk_examples}")
        logger.info(f"Multi-chunk QA pairs: {multi_chunk_examples}")
        logger.info(f"Chunks processed: {len(all_chunks) - chunks_skipped}")
        logger.info(f"Chunks skipped: {chunks_skipped}")

async def main():
    parser = argparse.ArgumentParser(description="Process chunks with aggregated analysis")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of chunks per batch")
    parser.add_argument("--rps", type=int, default=2, help="Requests per second limit")
    parser.add_argument("--output-file", default="comprehensive_qa_dataset_v5.jsonl", help="Output file name")
    parser.add_argument("--save-interval", type=int, default=50, help="Save progress every N examples")
    parser.add_argument("--aggregate-interval", type=int, default=5, help="Perform aggregated analysis every N chunks")
    
    args = parser.parse_args()
    
    processor = AggregatedProcessor(
        model=args.model,
        batch_size=args.batch_size,
        rps=args.rps
    )
    
    await processor.process_all_chunks(
        output_file=args.output_file,
        save_interval=args.save_interval,
        aggregate_interval=args.aggregate_interval
    )

if __name__ == "__main__":
    asyncio.run(main())