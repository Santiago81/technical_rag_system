"""
Chunk Manager for SQLite database operations and caching
"""
import sqlite3
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Data structure for a text chunk"""
    id: int
    content: str
    source_pdf: str
    page_number: int
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'content': self.content,
            'source_pdf': self.source_pdf,
            'page_number': self.page_number
        }

class ChunkManager:
    """Manages chunk retrieval and caching from SQLite database"""
    
    def __init__(self, db_path: str = "docs.db", cache_dir: str = "qa_cache"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache for processed chunks
        self.processed_cache_file = self.cache_dir / "processed_chunks.pkl"
        self.processed_chunks = self._load_processed_cache()
        
        # Memory cache for chunks
        self._chunk_cache = {}
        
    def _load_processed_cache(self) -> set:
        """Load processed chunk IDs from cache"""
        if self.processed_cache_file.exists():
            try:
                with open(self.processed_cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return set()
    
    def _save_processed_cache(self):
        """Save processed chunk IDs to cache"""
        try:
            with open(self.processed_cache_file, 'wb') as f:
                pickle.dump(self.processed_chunks, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        """Get a single chunk by ID"""
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                WHERE cc.id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                chunk = Chunk(*row)
                self._chunk_cache[chunk_id] = chunk
                return chunk
                
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
        
        return None
    
    def get_chunks_batch(self, limit: int = 100, offset: int = 0, 
                        filter_processed: bool = True) -> List[Chunk]:
        """Get a batch of chunks"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
            """
            
            if filter_processed:
                processed_ids = list(self.processed_chunks)
                if processed_ids:
                    placeholders = ','.join('?' * len(processed_ids))
                    query += f" WHERE cc.id NOT IN ({placeholders}) "
                    cursor.execute(query + " LIMIT ? OFFSET ?", 
                                 processed_ids + [limit, offset])
                else:
                    cursor.execute(query + " LIMIT ? OFFSET ?", (limit, offset))
            else:
                cursor.execute(query + " LIMIT ? OFFSET ?", (limit, offset))
            
            rows = cursor.fetchall()
            conn.close()
            
            chunks = [Chunk(*row) for row in rows]
            
            # Cache chunks
            for chunk in chunks:
                self._chunk_cache[chunk.id] = chunk
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks batch: {e}")
            return []
    
    def get_context_chunks(self, chunk_id: int, context_window: int = 2) -> Dict[str, List[Chunk]]:
        """Get chunks before and after the target chunk"""
        try:
            target_chunk = self.get_chunk(chunk_id)
            if not target_chunk:
                return {'before': [], 'after': [], 'target': None}
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get chunks before
            cursor.execute("""
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                WHERE m.filename = ? AND cc.start_page < ? 
                ORDER BY cc.start_page DESC 
                LIMIT ?
            """, (target_chunk.source_pdf, target_chunk.page_number, context_window))
            
            before_rows = cursor.fetchall()
            before_chunks = [Chunk(*row) for row in reversed(before_rows)]
            
            # Get chunks after
            cursor.execute("""
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                WHERE m.filename = ? AND cc.start_page > ? 
                ORDER BY cc.start_page ASC 
                LIMIT ?
            """, (target_chunk.source_pdf, target_chunk.page_number, context_window))
            
            after_rows = cursor.fetchall()
            after_chunks = [Chunk(*row) for row in after_rows]
            
            conn.close()
            
            return {
                'before': before_chunks,
                'after': after_chunks,
                'target': target_chunk
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context chunks: {e}")
            return {'before': [], 'after': [], 'target': None}
    
    def get_chunks_by_pdf(self, pdf_name: str, limit: int = 100) -> List[Chunk]:
        """Get all chunks from a specific PDF"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                WHERE m.filename = ? 
                ORDER BY cc.start_page 
                LIMIT ?
            """, (pdf_name, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [Chunk(*row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error retrieving chunks by PDF: {e}")
            return []
    
    def search_chunks(self, query: str, limit: int = 10) -> List[Chunk]:
        """Search chunks by content"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                WHERE cc.chunk_text LIKE ? 
                ORDER BY cc.id 
                LIMIT ?
            """, (f"%{query}%", limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [Chunk(*row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []
    
    def get_random_chunk_pairs(self, n_pairs: int) -> List[Tuple[Chunk, Chunk]]:
        """Get random pairs of chunks for synthesis questions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get random chunks from same PDF
            cursor.execute("""
                SELECT cc1.id, cc1.chunk_text, m.filename, cc1.start_page,
                       cc2.id, cc2.chunk_text, m.filename, cc2.start_page
                FROM content_chunks cc1
                JOIN content_chunks cc2 ON cc1.manual_id = cc2.manual_id
                JOIN manuals m ON cc1.manual_id = m.id
                WHERE cc1.id < cc2.id 
                AND ABS(cc1.start_page - cc2.start_page) <= 5
                ORDER BY RANDOM()
                LIMIT ?
            """, (n_pairs,))
            
            rows = cursor.fetchall()
            conn.close()
            
            pairs = []
            for row in rows:
                chunk1 = Chunk(row[0], row[1], row[2], row[3])
                chunk2 = Chunk(row[4], row[5], row[6], row[7])
                pairs.append((chunk1, chunk2))
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error getting random chunk pairs: {e}")
            return []
    
    def mark_as_processed(self, chunk_ids: List[int]):
        """Mark chunks as processed"""
        self.processed_chunks.update(chunk_ids)
        self._save_processed_cache()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about chunks"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM content_chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT m.filename, COUNT(cc.id) 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                GROUP BY m.filename
            """)
            
            chunks_by_pdf = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_chunks': total_chunks,
                'processed_chunks': len(self.processed_chunks),
                'unprocessed_chunks': total_chunks - len(self.processed_chunks),
                'chunks_by_pdf': chunks_by_pdf
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def get_sample_chunks(self, n_samples: int = 5) -> List[Chunk]:
        """Get sample chunks for testing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cc.id, cc.chunk_text, m.filename, cc.start_page 
                FROM content_chunks cc
                JOIN manuals m ON cc.manual_id = m.id
                ORDER BY RANDOM()
                LIMIT ?
            """, (n_samples,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [Chunk(*row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting sample chunks: {e}")
            return []
    
    def get_all_chunk_ids(self) -> List[int]:
        """Get all chunk IDs from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM content_chunks ORDER BY id")
            
            rows = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting chunk IDs: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, any]]:
        """Get chunk data by ID as a dictionary"""
        chunk = self.get_chunk(chunk_id)
        if chunk:
            return {
                'id': chunk.id,
                'chunk_text': chunk.content,
                'filename': chunk.source_pdf,
                'start_page': chunk.page_number,
                'end_page': chunk.page_number  # Simplified
            }
        return None