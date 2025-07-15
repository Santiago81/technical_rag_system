#!/usr/bin/env python3
"""
Fix CC103 Hardware manual processing
Re-chunks the manual with better segmentation
"""
import sqlite3
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class CC103ManualFixer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.manual_id = 2  # CC103 Hardware manual
        
    def analyze_chunks(self):
        """Analyze current chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                AVG(chunk_size) as avg_size,
                MIN(chunk_size) as min_size,
                MAX(chunk_size) as max_size,
                COUNT(CASE WHEN chunk_size < 200 THEN 1 END) as small_chunks,
                COUNT(CASE WHEN chunk_size < 100 THEN 1 END) as tiny_chunks
            FROM content_chunks 
            WHERE manual_id = ?
        """, (self.manual_id,))
        
        stats = cursor.fetchone()
        logger.info(f"CC103 Manual Statistics:")
        logger.info(f"Total chunks: {stats[0]}")
        logger.info(f"Average size: {stats[1]:.1f} chars")
        logger.info(f"Min/Max size: {stats[2]}/{stats[3]} chars")
        logger.info(f"Small chunks (<200 chars): {stats[4]} ({stats[4]/stats[0]*100:.1f}%)")
        logger.info(f"Tiny chunks (<100 chars): {stats[5]} ({stats[5]/stats[0]*100:.1f}%)")
        
        conn.close()
        
    def merge_small_chunks(self):
        """Merge adjacent small chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all chunks for this manual ordered by page and index
        cursor.execute("""
            SELECT id, chunk_text, chunk_index, start_page, chunk_size
            FROM content_chunks
            WHERE manual_id = ?
            ORDER BY start_page, chunk_index
        """, (self.manual_id,))
        
        chunks = cursor.fetchall()
        merged_chunks = []
        current_merged = None
        
        for chunk in chunks:
            chunk_id, text, index, page, size = chunk
            
            # Skip very small chunks that are likely headers/footers
            if size < 50 and any(keyword in text.lower() for keyword in 
                ['contents', 'page', 'cc 10', 'interface conditions', 'flexible automation']):
                logger.debug(f"Skipping header/footer chunk {chunk_id}: {text[:30]}...")
                continue
            
            if current_merged is None:
                # Start new merged chunk
                current_merged = {
                    'text': text,
                    'start_page': page,
                    'chunk_ids': [chunk_id]
                }
            elif (len(current_merged['text']) < 400 and 
                  page - current_merged['start_page'] <= 1):
                # Merge if current is small and on same/adjacent page
                current_merged['text'] += '\n\n' + text
                current_merged['chunk_ids'].append(chunk_id)
            else:
                # Save current and start new
                if len(current_merged['text']) >= 200:  # Only save if meaningful size
                    merged_chunks.append(current_merged)
                current_merged = {
                    'text': text,
                    'start_page': page,
                    'chunk_ids': [chunk_id]
                }
        
        # Don't forget the last chunk
        if current_merged and len(current_merged['text']) >= 200:
            merged_chunks.append(current_merged)
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} larger chunks")
        
        # Update database
        # First mark all old chunks as processed
        old_chunk_ids = [c[0] for c in chunks]
        if old_chunk_ids:
            placeholders = ','.join(['?' for _ in old_chunk_ids])
            cursor.execute(f"""
                UPDATE content_chunks 
                SET chunk_text = chunk_text || '\n[MERGED]'
                WHERE id IN ({placeholders})
            """, old_chunk_ids)
        
        conn.commit()
        conn.close()
        
        return merged_chunks
    
    def clean_qa_dataset(self):
        """Remove irrelevant QA pairs from existing datasets"""
        qa_dir = Path('qa_dataset')
        
        irrelevant_patterns = [
            'capital de francia', 'capital of france',
            'fotosíntesis', 'photosynthesis',
            'cien años de soledad', 'hundred years of solitude',
            'planetas del sistema solar', 'planets of the solar system',
            'ciclo del agua', 'water cycle',
            'shakespeare', 'cervantes', 'gabriel garcía márquez'
        ]
        
        for jsonl_file in qa_dir.glob('*.jsonl'):
            logger.info(f"Cleaning {jsonl_file.name}")
            
            cleaned_lines = []
            removed_count = 0
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # Check if this QA is from CC103 manual
                        if 'CC103_Hardware' in str(data.get('metadata', {}).get('source_pdfs', [])):
                            # Check for irrelevant content
                            qa_text = str(data).lower()
                            if any(pattern in qa_text for pattern in irrelevant_patterns):
                                removed_count += 1
                                logger.debug(f"Removing irrelevant QA: {data['messages'][1]['content'][:50]}...")
                                continue
                        
                        cleaned_lines.append(line)
                        
                    except Exception as e:
                        logger.error(f"Error processing line: {e}")
                        cleaned_lines.append(line)
            
            # Write cleaned file
            if removed_count > 0:
                backup_file = jsonl_file.with_suffix('.jsonl.bak')
                jsonl_file.rename(backup_file)
                
                with open(jsonl_file, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                
                logger.info(f"Removed {removed_count} irrelevant QA pairs from {jsonl_file.name}")
                logger.info(f"Original backed up to {backup_file.name}")

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Fix CC103 Hardware manual issues')
    parser.add_argument('--db-path', type=str, 
                        default='/Users/santiagojorda/Downloads/clode_technical_rag_system/data/sqlite/manuals.db',
                        help='Path to SQLite database')
    parser.add_argument('--analyze', action='store_true', help='Analyze chunks')
    parser.add_argument('--merge', action='store_true', help='Merge small chunks')
    parser.add_argument('--clean', action='store_true', help='Clean QA datasets')
    
    args = parser.parse_args()
    
    fixer = CC103ManualFixer(args.db_path)
    
    if args.analyze:
        fixer.analyze_chunks()
    
    if args.merge:
        merged = fixer.merge_small_chunks()
        logger.info(f"Created {len(merged)} improved chunks")
    
    if args.clean:
        fixer.clean_qa_dataset()
    
    if not any([args.analyze, args.merge, args.clean]):
        # Default: analyze and clean
        fixer.analyze_chunks()
        fixer.clean_qa_dataset()

if __name__ == "__main__":
    main()