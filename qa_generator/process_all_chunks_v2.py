#!/usr/bin/env python3
"""
Process ALL chunks with rate limiting and better content filtering
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime
import aiohttp
from asyncio import Semaphore

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_generator import QAGenerator, QAExample
from prompt_templates import QuestionType, DifficultyLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation_v2.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RateLimitedQAGenerator(QAGenerator):
    """QA Generator with rate limiting and content validation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rate limiting: 20 requests per second max
        self.semaphore = Semaphore(20)
        self.min_chunk_length = 200  # Minimum characters for useful chunk
        
    async def run_all_chunks_pipeline_v2(self,
                                        batch_size: int = 20,
                                        save_interval: int = 100,
                                        filename: str = None,
                                        requests_per_second: int = 15):
        """Process ALL chunks with rate limiting"""
        
        logger.info(f"Starting to process ALL chunks with batch_size={batch_size}, RPS limit={requests_per_second}")
        
        # Get total chunks count
        total_chunks = self.chunk_manager.get_statistics()['total_chunks']
        unprocessed = self.chunk_manager.get_statistics()['unprocessed_chunks']
        
        logger.info(f"Total chunks: {total_chunks}, Unprocessed: {unprocessed}")
        
        all_examples = []
        start_time = time.time()
        chunks_processed = 0
        chunks_skipped = 0
        append_mode = False
        
        while True:
            # Get batch of unprocessed chunks
            chunks = self.chunk_manager.get_chunks_batch(
                limit=batch_size, 
                filter_processed=True
            )
            
            if not chunks:
                logger.info("No more chunks to process")
                break
            
            # Filter out chunks that are too short
            valid_chunks = []
            for chunk in chunks:
                if len(chunk.content.strip()) >= self.min_chunk_length:
                    valid_chunks.append(chunk)
                else:
                    chunks_skipped += 1
                    logger.debug(f"Skipping chunk {chunk.id} - too short ({len(chunk.content)} chars)")
            
            if not valid_chunks:
                # Mark all chunks as processed even if skipped
                chunk_ids = [chunk.id for chunk in chunks]
                self.chunk_manager.mark_as_processed(chunk_ids)
                continue
            
            logger.info(f"Processing {len(valid_chunks)} valid chunks from batch of {len(chunks)} (skipped {chunks_skipped} total)")
            
            # Process chunks with rate limiting
            batch_start = time.time()
            batch_tasks = []
            
            # All 5 question types as requested
            question_configs = [
                (QuestionType.FACTUAL, DifficultyLevel.BASIC),
                (QuestionType.FACTUAL, DifficultyLevel.INTERMEDIATE),
                (QuestionType.SYNTHESIS, DifficultyLevel.INTERMEDIATE),
                (QuestionType.CAUSAL, DifficultyLevel.ADVANCED),
                (QuestionType.ANALYSIS, DifficultyLevel.ADVANCED)
            ]
            
            for chunk in valid_chunks:
                for q_type, difficulty in question_configs:
                    task = self.generate_qa_with_rate_limit(chunk, q_type, difficulty)
                    batch_tasks.append(task)
            
            # Process with controlled concurrency
            results = []
            for i in range(0, len(batch_tasks), requests_per_second):
                task_batch = batch_tasks[i:i + requests_per_second]
                batch_results = await asyncio.gather(*task_batch, return_exceptions=True)
                results.extend(batch_results)
                
                # Wait to respect rate limit
                if i + requests_per_second < len(batch_tasks):
                    await asyncio.sleep(1.0)  # 1 second between batches
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                elif result:
                    all_examples.extend(result)
            
            # Mark chunks as processed
            chunk_ids = [chunk.id for chunk in chunks]
            self.chunk_manager.mark_as_processed(chunk_ids)
            chunks_processed += len(chunks)
            
            # Save periodically
            if len(all_examples) >= save_interval:
                self.save_dataset(
                    all_examples[:save_interval], 
                    filename=filename, 
                    append=append_mode
                )
                all_examples = all_examples[save_interval:]
                append_mode = True
            
            # Progress update
            progress_pct = (chunks_processed / total_chunks) * 100
            batch_time = time.time() - batch_start
            logger.info(f"Progress: {chunks_processed}/{total_chunks} chunks ({progress_pct:.1f}%)")
            logger.info(f"Examples generated: {len(all_examples) + (save_interval if append_mode else 0)}")
            logger.info(f"Batch processed in {batch_time:.2f}s")
            logger.info(f"Chunks skipped (too short): {chunks_skipped}")
            
        # Save remaining examples
        if all_examples:
            self.save_dataset(all_examples, filename=filename, append=append_mode)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total chunks processed: {chunks_processed}")
        logger.info(f"Total chunks skipped: {chunks_skipped}")
        logger.info(f"Final statistics: {self.stats}")
        
        self._generate_quality_report()
    
    async def generate_qa_with_rate_limit(self, chunk, q_type, difficulty):
        """Generate QA with rate limiting"""
        async with self.semaphore:
            try:
                # Add context validation
                if not self._is_valid_technical_content(chunk.content):
                    logger.debug(f"Skipping chunk {chunk.id} - not technical content")
                    return []
                
                # Generate QA
                qa_pairs = await self.generate_qa_for_chunk(
                    chunk, 
                    q_type, 
                    difficulty,
                    context_window=1
                )
                
                examples = []
                for qa in qa_pairs:
                    # Validate the generated QA is relevant
                    if self._is_relevant_qa(qa, chunk):
                        qa['quality_score'] = 0.8
                        qa['question_variations'] = []
                        
                        context = f"Manual: {chunk.source_pdf}\nChunk ID: {chunk.id}, Página: {chunk.page_number}\n\n{chunk.content}"
                        
                        example = QAExample(
                            messages=[
                                {
                                    "role": "system",
                                    "content": self.prompt_templates.format_system_message(
                                        self.domain, 
                                        context
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": qa['question']
                                },
                                {
                                    "role": "assistant",
                                    "content": qa['answer']
                                }
                            ],
                            metadata={
                                'source_chunk_ids': qa['source_chunk_ids'],
                                'source_pdfs': qa['source_pdfs'],
                                'page_numbers': qa['page_numbers'],
                                'difficulty': difficulty.value,
                                'reasoning_type': q_type.value,
                                'requires_chunks': qa.get('requires_chunks', qa['source_chunk_ids']),
                                'quality_score': qa['quality_score'],
                                'question_variations': qa['question_variations']
                            }
                        )
                        
                        examples.append(example)
                        
                        # Update stats
                        self.stats['total_generated'] += 1
                        self.stats['total_accepted'] += 1
                        self.stats['by_type'][q_type.value] = self.stats['by_type'].get(q_type.value, 0) + 1
                        self.stats['by_difficulty'][difficulty.value] = self.stats['by_difficulty'].get(difficulty.value, 0) + 1
                    else:
                        logger.debug(f"Rejected irrelevant QA: {qa.get('question', 'N/A')}")
                        self.stats['total_rejected'] += 1
                
                return examples
                
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Rate limit hit, waiting 2 seconds...")
                    await asyncio.sleep(2)
                logger.error(f"Error generating QA for chunk {chunk.id}: {e}")
                return []
    
    def _is_valid_technical_content(self, content):
        """Check if content is technical and not just headers/fragments"""
        # Skip if too short
        if len(content.strip()) < self.min_chunk_length:
            return False
        
        # Skip if mostly numbers or single words
        words = content.split()
        if len(words) < 10:
            return False
        
        # Skip if it's just a header or title
        lines = content.strip().split('\n')
        if len(lines) <= 2 and all(len(line.split()) < 5 for line in lines):
            return False
        
        return True
    
    def _is_relevant_qa(self, qa, chunk):
        """Check if QA is relevant to the chunk content"""
        question = qa.get('question', '').lower()
        answer = qa.get('answer', '').lower()
        
        # Reject obviously irrelevant questions
        irrelevant_keywords = [
            'capital de francia', 'capital of france',
            'fotosíntesis', 'photosynthesis',
            'cien años de soledad', 'hundred years of solitude',
            'planetas del sistema solar', 'planets of the solar system',
            'ciclo del agua', 'water cycle',
            'shakespeare', 'cervantes', 'gabriel garcía márquez'
        ]
        
        for keyword in irrelevant_keywords:
            if keyword in question or keyword in answer:
                return False
        
        # Check if question contains at least one word from the chunk
        chunk_words = set(chunk.content.lower().split())
        question_words = set(question.split())
        
        # Must have at least 2 words in common or be about technical topics
        technical_terms = {'cc10', 'hardware', 'interface', 'control', 'automation', 
                          'drive', 'steuerung', 'antrieb', 'system', 'module', 'signal',
                          'input', 'output', 'configuration', 'parameter', 'version'}
        
        common_words = chunk_words.intersection(question_words)
        technical_words = technical_terms.intersection(question_words)
        
        return len(common_words) >= 2 or len(technical_words) >= 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process ALL chunks with rate limiting')
    
    # Processing parameters
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Number of chunks to process in each batch')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval for intermediate results')
    parser.add_argument('--rps', type=int, default=15,
                        help='Requests per second limit')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, 
                        default='comprehensive_qa_dataset_v2.jsonl',
                        help='Output JSONL filename')
    parser.add_argument('--output-dir', type=str, default='qa_dataset',
                        help='Output directory')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation')
    
    # Database parameters
    parser.add_argument('--db-path', type=str, 
                        default='/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db',
                        help='Path to SQLite database')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize processor
    processor = RateLimitedQAGenerator(
        api_key=api_key,
        model=args.model,
        db_path=args.db_path,
        output_dir=args.output_dir,
        domain='technical documentation',
        temperature=args.temperature
    )
    
    # Run processing
    async def run():
        await processor.run_all_chunks_pipeline_v2(
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            filename=args.output_file,
            requests_per_second=args.rps
        )
    
    # Run async function
    asyncio.run(run())
    
    logger.info("Processing completed!")

if __name__ == "__main__":
    main()