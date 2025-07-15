"""
SQLite Manager para Sistema RAG de Manuales Técnicos
Maneja todas las operaciones de base de datos con optimizaciones para RAG
"""
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta
import hashlib
import logging
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)

@dataclass
class ChunkData:
    """Estructura de datos para chunks"""
    manual_id: int
    chunk_text: str
    chunk_index: int = 0
    chunk_text_processed: Optional[str] = None
    chunk_size: Optional[int] = None
    overlap_size: int = 0
    start_page: int = 1
    end_page: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    keywords: List[str] = None
    entities: List[str] = None
    importance_score: float = 1.0
    metadata: Dict = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    content_block_id: Optional[int] = None
    tf_idf_scores: Optional[str] = None
    metadata_json: Optional[str] = None

class SQLiteRAGManager:
    """Manager principal para operaciones SQLite en sistema RAG"""
    
    def __init__(self, db_path: str, schema_path: Optional[str] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread local storage para conexiones
        self._local = threading.local()
        
        # Inicializar base de datos
        self._init_database(schema_path)
        
    @property
    def conn(self) -> sqlite3.Connection:
        """Obtener conexión thread-safe"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn
    
    def _create_connection(self) -> sqlite3.Connection:
        """Crear nueva conexión con configuración óptima"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        
        # Optimizaciones de rendimiento
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O
        
        return conn
    
    def _init_database(self, schema_path: Optional[str] = None):
        """Inicializar base de datos con schema"""
        if schema_path is None:
            schema_path = Path(__file__).parent / "schema.sql"
        else:
            schema_path = Path(schema_path)
        
        if not self.db_path.exists() and schema_path.exists():
            logger.info(f"Creando base de datos desde schema: {schema_path}")
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            conn = self._create_connection()
            try:
                conn.executescript(schema_sql)
                conn.commit()
                logger.info("Base de datos creada exitosamente")
            finally:
                conn.close()
    
    @contextmanager
    def transaction(self):
        """Context manager para transacciones seguras"""
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error en transacción: {e}")
            raise
    
    @contextmanager
    def batch_operation(self):
        """Context manager para operaciones en lote"""
        conn = self.conn
        # Deshabilitar autocommit temporalmente
        old_isolation = conn.isolation_level
        conn.isolation_level = None
        conn.execute("BEGIN")
        
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.isolation_level = old_isolation
    
    # ========== OPERACIONES CON MANUALES ==========
    
    def insert_manual(self, manual_data: Dict) -> int:
        """Insertar nuevo manual"""
        # Calcular hash del archivo si no está presente
        if 'file_hash' not in manual_data and 'file_path' in manual_data:
            manual_data['file_hash'] = self._calculate_file_hash(manual_data['file_path'])
        
        # Convertir metadata a JSON si es necesario
        if 'metadata_json' in manual_data and isinstance(manual_data['metadata_json'], dict):
            manual_data['metadata_json'] = json.dumps(manual_data['metadata_json'])
        
        columns = ', '.join(manual_data.keys())
        placeholders = ', '.join(['?' for _ in manual_data])
        
        query = f"""
            INSERT INTO manuals ({columns})
            VALUES ({placeholders})
        """
        
        with self.transaction() as conn:
            cursor = conn.execute(query, list(manual_data.values()))
            manual_id = cursor.lastrowid
            
            # Log de procesamiento
            self.log_processing(manual_id, 'extraction', 'started')
            
        return manual_id
    
    def get_manual(self, manual_id: int) -> Optional[Dict]:
        """Obtener información de un manual"""
        query = """
            SELECT * FROM manuals WHERE id = ?
        """
        
        cursor = self.conn.execute(query, (manual_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def update_manual_status(self, manual_id: int, status: str, error: Optional[str] = None):
        """Actualizar estado de procesamiento de manual"""
        query = """
            UPDATE manuals 
            SET processing_status = ?,
                processing_duration_ms = CASE 
                    WHEN ? = 'completed' OR ? = 'failed'
                    THEN (julianday(CURRENT_TIMESTAMP) - julianday(processed_date)) * 86400000
                    ELSE processing_duration_ms
                END
            WHERE id = ?
        """
        
        with self.transaction() as conn:
            conn.execute(query, (status, status, status, manual_id))
            
            # Log del cambio
            self.log_processing(
                manual_id, 
                'extraction', 
                status, 
                error_message=error
            )
    
    # ========== OPERACIONES CON CONTENT BLOCKS ==========
    
    def insert_content_blocks(self, blocks: List[Dict]) -> List[int]:
        """Insertar bloques de contenido en lote"""
        if not blocks:
            return []
        
        # Preparar datos
        for block in blocks:
            # Calcular estadísticas si no están presentes
            if 'char_count' not in block:
                block['char_count'] = len(block.get('content', ''))
            if 'word_count' not in block:
                block['word_count'] = len(block.get('content', '').split())
            
            # Convertir JSONs
            for json_field in ['bounding_box', 'style_attributes']:
                if json_field in block and isinstance(block[json_field], dict):
                    block[json_field] = json.dumps(block[json_field])
        
        # Obtener columnas del primer bloque
        columns = list(blocks[0].keys())
        placeholders = ', '.join(['?' for _ in columns])
        
        query = f"""
            INSERT INTO content_blocks ({', '.join(columns)})
            VALUES ({placeholders})
        """
        
        ids = []
        with self.batch_operation() as conn:
            for block in blocks:
                cursor = conn.execute(query, [block.get(col) for col in columns])
                ids.append(cursor.lastrowid)
        
        return ids
    
    # ========== OPERACIONES CON CHUNKS ==========
    
    def insert_chunks_batch(self, chunks: List[ChunkData]) -> List[int]:
        """Insertar chunks en lote con embeddings"""
        if not chunks:
            return []
        
        values = []
        for chunk in chunks:
            # Convertir ChunkData a dict si es necesario
            if isinstance(chunk, ChunkData):
                chunk_dict = asdict(chunk)
            else:
                chunk_dict = chunk
            
            # Procesar embedding
            embedding_bytes = None
            if chunk_dict.get('embedding') is not None:
                if isinstance(chunk_dict['embedding'], np.ndarray):
                    embedding_bytes = chunk_dict['embedding'].astype(np.float32).tobytes()
                else:
                    embedding_bytes = chunk_dict['embedding']
            
            # Procesar campos JSON
            keywords = chunk_dict.get('keywords', [])
            if isinstance(keywords, list):
                keywords = json.dumps(keywords)
            
            entities = chunk_dict.get('entities', [])
            if isinstance(entities, list):
                entities = json.dumps(entities)
            
            metadata = chunk_dict.get('metadata', {})
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            # Calcular tamaño si no está presente
            chunk_size = chunk_dict.get('chunk_size') or len(chunk_dict.get('chunk_text', ''))
            
            # end_page por defecto es start_page
            end_page = chunk_dict.get('end_page') or chunk_dict.get('start_page', 1)
            
            values.append((
                chunk_dict['manual_id'],
                chunk_dict.get('content_block_id'),
                chunk_dict.get('chunk_index', 0),
                chunk_dict['chunk_text'],
                chunk_dict.get('chunk_text_processed'),
                chunk_size,
                chunk_dict.get('overlap_size', 0),
                chunk_dict.get('start_page', 1),
                end_page,
                embedding_bytes,
                chunk_dict.get('embedding_model'),
                chunk_dict.get('embedding_dimension'),
                datetime.now() if embedding_bytes else None,
                chunk_dict.get('context_before'),
                chunk_dict.get('context_after'),
                keywords,
                entities,
                chunk_dict.get('tf_idf_scores'),
                chunk_dict.get('importance_score', 1.0),
                metadata
            ))
        
        query = """
            INSERT INTO content_chunks (
                manual_id, content_block_id, chunk_index,
                chunk_text, chunk_text_processed,
                chunk_size, overlap_size, start_page, end_page,
                embedding, embedding_model, embedding_dimension, embedding_date,
                context_before, context_after,
                keywords, entities, tf_idf_scores, importance_score,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        ids = []
        with self.batch_operation() as conn:
            cursor = conn.cursor()
            for value_tuple in values:
                cursor.execute(query, value_tuple)
                ids.append(cursor.lastrowid)
        
        # Log de vectorización si hay embeddings
        if any(v[9] is not None for v in values):  # v[9] es embedding_bytes
            manual_id = values[0][0]  # Asumiendo mismo manual para el batch
            self.log_processing(manual_id, 'vectorization', 'completed')
        
        return ids
    
    def search_similar_chunks(self, query_embedding: np.ndarray, 
                            manual_id: Optional[int] = None,
                            limit: int = 10,
                            min_score: float = 0.0) -> List[Dict]:
        """Buscar chunks similares usando embeddings almacenados"""
        
        # Verificar cache primero
        query_hash = self._hash_embedding(query_embedding)
        cached = self._get_cached_search(query_hash)
        if cached:
            return cached
        
        # Construir query
        where_clauses = ["c.embedding IS NOT NULL"]
        params = []
        
        if manual_id:
            where_clauses.append("c.manual_id = ?")
            params.append(manual_id)
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
            SELECT 
                c.id,
                c.chunk_text,
                c.chunk_text_processed,
                c.manual_id,
                m.name as manual_name,
                m.document_type,
                c.start_page,
                c.end_page,
                c.importance_score,
                c.embedding,
                c.embedding_model,
                c.keywords,
                c.entities,
                c.metadata_json,
                c.context_before,
                c.context_after
            FROM content_chunks c
            JOIN manuals m ON c.manual_id = m.id
            WHERE {where_clause}
                AND m.is_latest = 1
            ORDER BY c.importance_score DESC
            LIMIT ?
        """
        params.append(limit * 5)  # Pre-filtrar más para scoring
        
        cursor = self.conn.execute(query, params)
        chunks = []
        
        for row in cursor:
            if row['embedding']:
                # Reconstruir embedding desde bytes
                stored_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                
                # Calcular similitud coseno
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                
                # Aplicar filtro de score mínimo
                if similarity < min_score:
                    continue
                
                # Score ponderado por importancia
                final_score = float(similarity * row['importance_score'])
                
                chunks.append({
                    'id': row['id'],
                    'text': row['chunk_text'],
                    'text_processed': row['chunk_text_processed'],
                    'manual_id': row['manual_id'],
                    'manual_name': row['manual_name'],
                    'document_type': row['document_type'],
                    'pages': f"{row['start_page']}-{row['end_page']}",
                    'score': final_score,
                    'similarity': float(similarity),
                    'importance': row['importance_score'],
                    'keywords': json.loads(row['keywords'] or '[]'),
                    'entities': json.loads(row['entities'] or '[]'),
                    'metadata': json.loads(row['metadata_json'] or '{}'),
                    'context': {
                        'before': row['context_before'],
                        'after': row['context_after']
                    }
                })
        
        # Ordenar por score final y limitar
        chunks.sort(key=lambda x: x['score'], reverse=True)
        result = chunks[:limit]
        
        # Cachear resultado
        if result:
            self._cache_search_result(query_hash, result)
        
        return result
    
    def hybrid_search(self, query_text: str, query_embedding: np.ndarray,
                     manual_id: Optional[int] = None,
                     limit: int = 10,
                     alpha: float = 0.7) -> List[Dict]:
        """Búsqueda híbrida: keyword + vector"""
        
        # Búsqueda por keywords usando FTS
        keyword_results = self._search_by_keywords(query_text, manual_id, limit * 2)
        
        # Búsqueda por vectores
        vector_results = self.search_similar_chunks(query_embedding, manual_id, limit * 2)
        
        # Fusionar resultados
        return self._merge_search_results(keyword_results, vector_results, alpha, limit)
    
    def _search_by_keywords(self, query: str, manual_id: Optional[int], limit: int) -> List[Dict]:
        """Búsqueda por keywords usando FTS5"""
        
        where_clause = ""
        params = [query]
        
        if manual_id:
            where_clause = "AND c.manual_id = ?"
            params.append(manual_id)
        
        fts_query = f"""
            SELECT 
                c.id,
                c.chunk_text,
                c.manual_id,
                m.name as manual_name,
                c.start_page,
                c.end_page,
                c.importance_score,
                c.keywords,
                c.metadata_json,
                rank as fts_rank
            FROM chunks_fts f
            JOIN content_chunks c ON f.rowid = c.id
            JOIN manuals m ON c.manual_id = m.id
            WHERE chunks_fts MATCH ?
                {where_clause}
                AND m.is_latest = 1
            ORDER BY rank
            LIMIT ?
        """
        params.append(limit)
        
        cursor = self.conn.execute(fts_query, params)
        
        results = []
        for row in cursor:
            # Normalizar FTS rank a [0, 1]
            normalized_rank = 1.0 / (1.0 + abs(row['fts_rank']))
            
            results.append({
                'id': row['id'],
                'text': row['chunk_text'],
                'manual_id': row['manual_id'],
                'manual_name': row['manual_name'],
                'pages': f"{row['start_page']}-{row['end_page']}",
                'score': float(normalized_rank * row['importance_score']),
                'search_type': 'keyword',
                'keywords': json.loads(row['keywords'] or '[]'),
                'metadata': json.loads(row['metadata_json'] or '{}')
            })
        
        return results
    
    def _merge_search_results(self, keyword_results: List[Dict], 
                            vector_results: List[Dict],
                            alpha: float, limit: int) -> List[Dict]:
        """Fusionar resultados de búsqueda híbrida"""
        
        # Crear diccionario para fusión
        merged = {}
        
        # Agregar resultados de keywords
        for result in keyword_results:
            chunk_id = result['id']
            result['final_score'] = (1 - alpha) * result['score']
            merged[chunk_id] = result
        
        # Fusionar con resultados vectoriales
        for result in vector_results:
            chunk_id = result['id']
            if chunk_id in merged:
                # Combinar scores
                merged[chunk_id]['final_score'] += alpha * result['score']
                merged[chunk_id]['search_type'] = 'hybrid'
                # Agregar información adicional del resultado vectorial
                merged[chunk_id]['similarity'] = result.get('similarity', 0)
            else:
                result['final_score'] = alpha * result['score']
                result['search_type'] = 'vector'
                merged[chunk_id] = result
        
        # Ordenar por score final
        final_results = sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)
        
        return final_results[:limit]
    
    # ========== OPERACIONES CON IMÁGENES ==========
    
    def insert_images_batch(self, images: List[Dict]) -> List[int]:
        """Insertar imágenes en lote"""
        if not images:
            return []
        
        values = []
        for img in images:
            # Procesar embeddings
            image_embedding = None
            if img.get('image_embedding') is not None:
                if isinstance(img['image_embedding'], np.ndarray):
                    image_embedding = img['image_embedding'].astype(np.float32).tobytes()
            
            text_embedding = None
            if img.get('text_embedding') is not None:
                if isinstance(img['text_embedding'], np.ndarray):
                    text_embedding = img['text_embedding'].astype(np.float32).tobytes()
            
            # Procesar JSON
            detected_objects = img.get('detected_objects', {})
            if isinstance(detected_objects, dict):
                detected_objects = json.dumps(detected_objects)
            
            values.append((
                img['manual_id'],
                img['page_number'],
                img['image_index'],
                img['image_type'],
                img['file_path'],
                img['file_format'],
                img.get('file_size'),
                img.get('width'),
                img.get('height'),
                img.get('dpi'),
                img.get('color_space'),
                img.get('ocr_text'),
                img.get('ocr_confidence'),
                img.get('ocr_language'),
                image_embedding,
                text_embedding,
                img.get('description'),
                img.get('auto_caption'),
                detected_objects,
                img.get('is_technical_diagram', False),
                img.get('has_annotations', False),
                img.get('has_text', False),
                img.get('file_hash'),
                img.get('thumbnail_path')
            ))
        
        query = """
            INSERT INTO images (
                manual_id, page_number, image_index, image_type,
                file_path, file_format, file_size, width, height, dpi,
                color_space, ocr_text, ocr_confidence, ocr_language,
                image_embedding, text_embedding,
                description, auto_caption, detected_objects,
                is_technical_diagram, has_annotations, has_text,
                file_hash, thumbnail_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        ids = []
        with self.batch_operation() as conn:
            cursor = conn.cursor()
            for value_tuple in values:
                cursor.execute(query, value_tuple)
                ids.append(cursor.lastrowid)
        
        return ids
    
    # ========== OPERACIONES CON TABLAS ==========
    
    def insert_tables_batch(self, tables: List[Dict]) -> List[int]:
        """Insertar tablas extraídas en lote"""
        if not tables:
            return []
        
        values = []
        for table in tables:
            # Procesar JSONs
            for json_field in ['json_content', 'headers', 'data_types', 'metadata_json']:
                if json_field in table and isinstance(table[json_field], (dict, list)):
                    table[json_field] = json.dumps(table[json_field])
            
            # Procesar embedding
            table_embedding = None
            if table.get('table_embedding') is not None:
                if isinstance(table['table_embedding'], np.ndarray):
                    table_embedding = table['table_embedding'].astype(np.float32).tobytes()
            
            values.append((
                table['manual_id'],
                table['page_number'],
                table['table_index'],
                table.get('extraction_method'),
                table.get('extraction_accuracy'),
                table.get('csv_path'),
                table.get('json_content'),
                table.get('markdown_content'),
                table.get('row_count'),
                table.get('column_count'),
                table.get('headers'),
                table.get('data_types'),
                table.get('table_content'),
                table_embedding,
                table.get('table_type'),
                table.get('has_numeric_data', False),
                table.get('has_headers', True),
                table.get('metadata_json')
            ))
        
        query = """
            INSERT INTO extracted_tables (
                manual_id, page_number, table_index,
                extraction_method, extraction_accuracy,
                csv_path, json_content, markdown_content,
                row_count, column_count, headers, data_types,
                table_content, table_embedding,
                table_type, has_numeric_data, has_headers,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        ids = []
        with self.batch_operation() as conn:
            cursor = conn.cursor()
            for value_tuple in values:
                cursor.execute(query, value_tuple)
                ids.append(cursor.lastrowid)
        
        return ids
    
    # ========== ANÁLISIS Y ESTRUCTURA ==========
    
    def save_document_analysis(self, analysis: Dict) -> int:
        """Guardar análisis del documento"""
        
        # Convertir estrategia a JSON si es necesario
        if 'extraction_strategy' in analysis and isinstance(analysis['extraction_strategy'], dict):
            analysis['extraction_strategy'] = json.dumps(analysis['extraction_strategy'])
        
        columns = ', '.join(analysis.keys())
        placeholders = ', '.join(['?' for _ in analysis])
        
        query = f"""
            INSERT OR REPLACE INTO document_analysis ({columns})
            VALUES ({placeholders})
        """
        
        with self.transaction() as conn:
            cursor = conn.execute(query, list(analysis.values()))
            return cursor.lastrowid
    
    def save_document_structure(self, manual_id: int, structure: List[Dict]) -> List[int]:
        """Guardar estructura del documento (TOC)"""
        if not structure:
            return []
        
        ids = []
        
        def insert_structure_recursive(items: List[Dict], parent_id: Optional[int] = None, level: int = 0):
            for i, item in enumerate(items):
                cursor = self.conn.execute("""
                    INSERT INTO document_structure (
                        manual_id, structure_type, title,
                        start_page, end_page, parent_id,
                        level, order_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    manual_id,
                    item.get('structure_type', 'section'),
                    item['title'],
                    item.get('start_page'),
                    item.get('end_page'),
                    parent_id,
                    level,
                    i
                ))
                
                item_id = cursor.lastrowid
                ids.append(item_id)
                
                # Insertar hijos recursivamente
                if 'children' in item:
                    insert_structure_recursive(item['children'], item_id, level + 1)
        
        with self.transaction():
            insert_structure_recursive(structure)
        
        return ids
    
    # ========== UTILIDADES ==========
    
    def get_context_window(self, chunk_id: int, window_size: int = 2) -> Dict:
        """Obtener chunk con contexto expandido"""
        query = """
            WITH target_chunk AS (
                SELECT manual_id, chunk_index, start_page
                FROM content_chunks
                WHERE id = ?
            )
            SELECT 
                c.id,
                c.chunk_text,
                c.chunk_index,
                c.start_page,
                c.end_page,
                CASE 
                    WHEN c.id = ? THEN 'current'
                    WHEN c.chunk_index < tc.chunk_index THEN 'before'
                    ELSE 'after'
                END as position
            FROM content_chunks c
            JOIN target_chunk tc ON c.manual_id = tc.manual_id
            WHERE c.chunk_index BETWEEN tc.chunk_index - ? AND tc.chunk_index + ?
            ORDER BY c.chunk_index
        """
        
        cursor = self.conn.execute(query, (chunk_id, chunk_id, window_size, window_size))
        
        context = {
            'before': [],
            'current': None,
            'after': []
        }
        
        for row in cursor:
            chunk_data = {
                'id': row['id'],
                'text': row['chunk_text'],
                'pages': f"{row['start_page']}-{row['end_page']}"
            }
            
            if row['position'] == 'current':
                context['current'] = chunk_data
            else:
                context[row['position']].append(chunk_data)
        
        return context
    
    def update_chunk_importance(self, chunk_id: int, feedback: str):
        """Actualizar importancia basada en feedback del usuario"""
        multipliers = {
            'helpful': 1.1,
            'very_helpful': 1.2,
            'not_helpful': 0.9,
            'irrelevant': 0.7,
            'neutral': 1.0
        }
        
        multiplier = multipliers.get(feedback, 1.0)
        
        query = """
            UPDATE content_chunks
            SET importance_score = MIN(MAX(importance_score * ?, 0.1), 10.0),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        
        with self.transaction() as conn:
            conn.execute(query, (multiplier, chunk_id))
    
    def insert_tables_batch(self, tables: List[Dict]) -> List[int]:
        """Insertar tablas en lote"""
        if not tables:
            return []
        
        table_ids = []
        
        with self.transaction() as conn:
            for table_data in tables:
                # Convertir diccionarios/listas a JSON
                if 'headers' in table_data and isinstance(table_data['headers'], list):
                    table_data['headers'] = json.dumps(table_data['headers'])
                if 'data_types' in table_data and isinstance(table_data['data_types'], dict):
                    table_data['data_types'] = json.dumps(table_data['data_types'])
                if 'metadata_json' in table_data and isinstance(table_data['metadata_json'], dict):
                    table_data['metadata_json'] = json.dumps(table_data['metadata_json'])
                
                columns = ', '.join(table_data.keys())
                placeholders = ', '.join(['?' for _ in table_data])
                
                query = f"""
                    INSERT INTO tables ({columns})
                    VALUES ({placeholders})
                """
                
                cursor = conn.execute(query, list(table_data.values()))
                table_ids.append(cursor.lastrowid)
        
        return table_ids
    
    def save_document_analysis(self, analysis_data: Dict) -> int:
        """Guardar análisis de documento"""
        columns = ', '.join(analysis_data.keys())
        placeholders = ', '.join(['?' for _ in analysis_data])
        
        query = f"""
            INSERT INTO document_analysis ({columns})
            VALUES ({placeholders})
        """
        
        with self.transaction() as conn:
            cursor = conn.execute(query, list(analysis_data.values()))
            return cursor.lastrowid
    
    def log_processing(self, manual_id: int, process_type: str, status: str,
                      error_message: Optional[str] = None, details: Optional[Dict] = None):
        """Log de eventos de procesamiento"""
        
        details_json = None
        if details:
            details_json = json.dumps(details)
        
        # Si es completed, actualizar el registro started existente
        if status == 'completed' or status == 'failed':
            update_query = """
                UPDATE processing_logs
                SET status = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = ?,
                    details_json = ?
                WHERE manual_id = ?
                    AND process_type = ?
                    AND status = 'started'
                    AND completed_at IS NULL
            """
            
            with self.transaction() as conn:
                result = conn.execute(update_query, 
                    (status, error_message, details_json, manual_id, process_type))
                
                # Si no se actualizó ningún registro, insertar uno nuevo
                if result.rowcount == 0:
                    conn.execute("""
                        INSERT INTO processing_logs (
                            manual_id, process_type, status,
                            error_message, details_json, completed_at
                        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (manual_id, process_type, status, error_message, details_json))
        else:
            # Insertar nuevo registro
            with self.transaction() as conn:
                conn.execute("""
                    INSERT INTO processing_logs (
                        manual_id, process_type, status,
                        error_message, details_json
                    ) VALUES (?, ?, ?, ?, ?)
                """, (manual_id, process_type, status, error_message, details_json))
    
    # ========== CACHE ==========
    
    def _get_cached_search(self, query_hash: str) -> Optional[List[Dict]]:
        """Obtener resultado cacheado si existe"""
        query = """
            SELECT result_chunk_ids, result_scores
            FROM search_cache
            WHERE query_hash = ?
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        """
        
        cursor = self.conn.execute(query, (query_hash,))
        row = cursor.fetchone()
        
        if row:
            # Actualizar contador y timestamp
            self.conn.execute("""
                UPDATE search_cache
                SET hit_count = hit_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE query_hash = ?
            """, (query_hash,))
            
            # Reconstruir resultados
            chunk_ids = json.loads(row['result_chunk_ids'])
            scores = json.loads(row['result_scores'])
            
            # Obtener chunks
            results = []
            for chunk_id, score in zip(chunk_ids, scores):
                chunk = self._get_chunk_by_id(chunk_id)
                if chunk:
                    chunk['score'] = score
                    results.append(chunk)
            
            return results
        
        return None
    
    def _cache_search_result(self, query_hash: str, results: List[Dict], 
                           ttl_hours: int = 24):
        """Cachear resultado de búsqueda"""
        chunk_ids = [r['id'] for r in results]
        scores = [r['score'] for r in results]
        
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        query = """
            INSERT OR REPLACE INTO search_cache (
                query_hash, query_text, result_chunk_ids,
                result_scores, model_version, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        
        with self.transaction() as conn:
            conn.execute(query, (
                query_hash,
                '',  # No guardamos el texto por privacidad
                json.dumps(chunk_ids),
                json.dumps(scores),
                'v1.0',
                expires_at
            ))
    
    def _get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        """Obtener chunk por ID"""
        query = """
            SELECT 
                c.*,
                m.name as manual_name
            FROM content_chunks c
            JOIN manuals m ON c.manual_id = m.id
            WHERE c.id = ?
        """
        
        cursor = self.conn.execute(query, (chunk_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    # ========== UTILIDADES PRIVADAS ==========
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcular similitud coseno entre dos vectores"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Generar hash único para un embedding"""
        return hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash SHA256 de un archivo"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    # ========== CONSULTAS Y ESTADÍSTICAS ==========
    
    def get_processing_stats(self, manual_id: Optional[int] = None) -> Dict:
        """Obtener estadísticas de procesamiento"""
        where_clause = ""
        params = []
        
        if manual_id:
            where_clause = "WHERE manual_id = ?"
            params.append(manual_id)
        
        query = f"""
            SELECT * FROM v_processing_stats {where_clause}
        """
        
        cursor = self.conn.execute(query, params)
        results = [dict(row) for row in cursor]
        
        if manual_id and results:
            return results[0]
        return results
    
    def get_manual_summary(self, manual_id: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Obtener resumen de manuales"""
        where_clause = ""
        params = []
        
        if manual_id:
            where_clause = "WHERE id = ?"
            params.append(manual_id)
        
        query = f"""
            SELECT * FROM v_manual_summary {where_clause}
        """
        
        cursor = self.conn.execute(query, params)
        results = [dict(row) for row in cursor]
        
        if manual_id and results:
            return results[0]
        return results
    
    def close(self):
        """Cerrar conexión"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None