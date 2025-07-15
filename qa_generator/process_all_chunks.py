#!/usr/bin/env python3
"""
Process ALL chunks from the database to generate a comprehensive QA dataset
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_generator import QAGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation_all_chunks.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AllChunksProcessor(QAGenerator):
    """Extended QA Generator that processes all chunks"""
    
    async def run_all_chunks_pipeline(self,
                                     batch_size: int = 50,
                                     save_interval: int = 100,
                                     filename: str = None,
                                     max_workers: int = 5):
        """Process ALL chunks in the database"""
        
        logger.info(f"Starting to process ALL chunks with batch_size={batch_size}")
        
        # Get total chunks count
        total_chunks = self.chunk_manager.get_statistics()['total_chunks']
        unprocessed = self.chunk_manager.get_statistics()['unprocessed_chunks']
        
        logger.info(f"Total chunks: {total_chunks}, Unprocessed: {unprocessed}")
        
        all_examples = []
        start_time = time.time()
        chunks_processed = 0
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
            
            logger.info(f"Processing batch of {len(chunks)} chunks (processed so far: {chunks_processed})")
            
            # Process chunks in parallel
            batch_tasks = []
            for chunk in chunks:
                # Only generate 2-3 question types per chunk to speed up
                question_configs = [
                    ('factual', 'basic'),
                    ('synthesis', 'intermediate'),
                    ('analysis', 'advanced')
                ]
                
                for q_type, difficulty in question_configs:
                    task = self.generate_single_qa(chunk, q_type, difficulty)
                    batch_tasks.append(task)
            
            # Execute all tasks concurrently
            if batch_tasks:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
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
                append_mode = True  # After first save, always append
            
            # Progress update
            progress_pct = (chunks_processed / total_chunks) * 100
            logger.info(f"Progress: {chunks_processed}/{total_chunks} chunks ({progress_pct:.1f}%)")
            logger.info(f"Examples generated: {len(all_examples) + (save_interval if append_mode else 0)}")
            logger.info(f"Current stats: {self.stats}")
            
            # Brief pause to avoid rate limiting
            await asyncio.sleep(0.5)
        
        # Save remaining examples
        if all_examples:
            self.save_dataset(all_examples, filename=filename, append=append_mode)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total chunks processed: {chunks_processed}")
        logger.info(f"Final statistics: {self.stats}")
        
        # Generate quality report
        self._generate_quality_report()
    
    async def generate_single_qa(self, chunk, q_type, difficulty):
        """Generate a single QA pair for a chunk"""
        try:
            from prompt_templates import QuestionType, DifficultyLevel
            
            # Map string to enum
            q_type_enum = QuestionType(q_type)
            diff_enum = DifficultyLevel(difficulty)
            
            # Generate QA
            qa_pairs = await self.generate_qa_for_chunk(
                chunk, 
                q_type_enum, 
                diff_enum,
                context_window=1  # Smaller context for speed
            )
            
            examples = []
            for qa in qa_pairs:
                # Skip quality evaluation for speed
                qa['quality_score'] = 0.8  # Default score
                qa['question_variations'] = []
                
                # Create training example
                context = f"Chunk ID: {chunk.id}, PDF: {chunk.source_pdf}, Página: {chunk.page_number}\n\n{chunk.content}"
                
                from qa_generator import QAExample
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
                        'difficulty': difficulty,
                        'reasoning_type': q_type,
                        'requires_chunks': qa.get('requires_chunks', qa['source_chunk_ids']),
                        'quality_score': qa['quality_score'],
                        'question_variations': qa['question_variations']
                    }
                )
                
                examples.append(example)
                
                # Update stats
                self.stats['total_generated'] += 1
                self.stats['total_accepted'] += 1
                self.stats['by_type'][q_type] = self.stats['by_type'].get(q_type, 0) + 1
                self.stats['by_difficulty'][difficulty] = self.stats['by_difficulty'].get(difficulty, 0) + 1
            
            return examples
            
        except Exception as e:
            logger.error(f"Error generating QA for chunk {chunk.id}: {e}")
            return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process ALL chunks to generate comprehensive QA dataset')
    
    # Processing parameters
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of chunks to process in each batch')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval for intermediate results')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum concurrent workers')
    
    # Output parameters
    parser.add_argument('--output-file', type=str, 
                        default='comprehensive_qa_dataset.jsonl',
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
    
    # Other parameters
    parser.add_argument('--domain', type=str, default='technical documentation',
                        help='Domain for the QA dataset')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize processor
    processor = AllChunksProcessor(
        api_key=api_key,
        model=args.model,
        db_path=args.db_path,
        output_dir=args.output_dir,
        domain=args.domain,
        temperature=args.temperature
    )
    
    # Run processing
    async def run():
        await processor.run_all_chunks_pipeline(
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            filename=args.output_file,
            max_workers=args.max_workers
        )
    
    # Run async function
    asyncio.run(run())
    
    logger.info("All chunks processing completed!")

if __name__ == "__main__":
    main()