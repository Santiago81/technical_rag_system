"""
Adaptador para usar SQLite como backend del vector store manteniendo la interfaz existente
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import logging

from database.sqlite_manager import SQLiteRAGManager, ChunkData
from vectorstore.vector_manager import VectorManager

logger = logging.getLogger(__name__)

class SQLiteVectorAdapter:
    """
    Adaptador que permite usar SQLite como backend manteniendo 
    la interfaz del VectorManager existente
    """
    
    def __init__(self, config, db_path: Optional[str] = None):
        self.config = config
        
        # SQLite como backend principal
        if db_path is None:
            db_path = str(config.DATA_DIR / 'sqlite' / 'manuals.db')
        self.db = SQLiteRAGManager(db_path)
        
        # Vector manager para compatibilidad y generación de embeddings
        self.vector_manager = VectorManager(config)
        
        # Modo de operación
        self.use_sqlite_embeddings = True  # Si True, usa embeddings de SQLite
        self.sync_with_chromadb = False   # Si True, mantiene ChromaDB sincronizado
    
    def add_documents(self, texts: List[str], metadatas: List[Dict], 
                     ids: List[str], embeddings: Optional[List[np.ndarray]] = None):
        """Añadir documentos a ChromaDB (no a SQLite, ya que ya están ahí)"""
        
        # Si no hay embeddings, generarlos
        if embeddings is None or all(e is None for e in embeddings):
            embeddings = self.vector_manager.embedding_model.encode(texts)
        
        # Añadir directamente a la colección de ChromaDB
        self.vector_manager.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        )
        
        logger.info(f"Añadidos {len(texts)} documentos a ChromaDB")
        return ids
    
    def search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> Dict:
        """Buscar documentos - interfaz compatible con VectorManager"""
        
        # Generar embedding de la consulta
        query_embedding = self.vector_manager.embedding_model.encode([query])[0]
        
        # Filtrar por manual si se especifica
        manual_id = None
        if filter and 'manual_name' in filter:
            cursor = self.db.conn.execute(
                "SELECT id FROM manuals WHERE name = ? LIMIT 1",
                (filter['manual_name'],)
            )
            row = cursor.fetchone()
            if row:
                manual_id = row[0]
        
        # Búsqueda híbrida en SQLite
        results = self.db.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            manual_id=manual_id,
            limit=k
        )
        
        # Formatear resultados para compatibilidad
        formatted_results = {
            'documents': [[r['text'] for r in results]],
            'metadatas': [[r.get('metadata', {}) for r in results]],
            'distances': [[1 - r['score'] for r in results]],  # Convertir score a distancia
            'ids': [[f"chunk_{r['id']}" for r in results]]
        }
        
        return formatted_results
    
    def delete_collection(self):
        """Eliminar colección - para compatibilidad"""
        logger.warning("delete_collection llamado - no implementado para SQLite")
        # No eliminamos datos de SQLite, pero podríamos marcarlos como inactivos
        pass
    
    def get_collection_stats(self) -> Dict:
        """Obtener estadísticas de la colección"""
        stats = {}
        
        # Contar chunks con embeddings
        cursor = self.db.conn.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT manual_id) as total_manuals,
                AVG(LENGTH(chunk_text)) as avg_chunk_size
            FROM content_chunks
            WHERE embedding IS NOT NULL
        """)
        
        row = cursor.fetchone()
        if row:
            stats.update(dict(row))
        
        # Información adicional
        cursor = self.db.conn.execute("""
            SELECT 
                COUNT(DISTINCT embedding_model) as embedding_models,
                MIN(embedding_date) as oldest_embedding,
                MAX(embedding_date) as newest_embedding
            FROM content_chunks
            WHERE embedding IS NOT NULL
        """)
        
        row = cursor.fetchone()
        if row:
            stats.update(dict(row))
        
        return stats
    
    def update_embeddings(self, model_name: Optional[str] = None, manual_id: Optional[int] = None):
        """Actualizar embeddings con un nuevo modelo"""
        
        if model_name:
            # Cambiar modelo
            from sentence_transformers import SentenceTransformer
            self.vector_manager.embedding_model = SentenceTransformer(model_name)
            new_model = model_name
        else:
            new_model = self.config.EMBEDDING_MODEL
        
        # Obtener chunks que necesitan actualización
        query = """
            SELECT id, chunk_text, chunk_text_processed
            FROM content_chunks
            WHERE embedding IS NULL OR embedding_model != ?
        """
        params = [new_model]
        
        if manual_id is not None:
            query += " AND manual_id = ?"
            params.append(manual_id)
            
        cursor = self.db.conn.execute(query, params)
        
        chunks = [dict(row) for row in cursor]
        
        if not chunks:
            logger.info("No hay chunks que actualizar")
            return
        
        logger.info(f"Actualizando {len(chunks)} embeddings con modelo {new_model}")
        
        # Procesar en lotes
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Generar embeddings
            texts = [c['chunk_text_processed'] or c['chunk_text'] for c in batch]
            embeddings = self.vector_manager.embedding_model.encode(texts)
            
            # Actualizar en base de datos
            for chunk, embedding in zip(batch, embeddings):
                self.db.conn.execute("""
                    UPDATE content_chunks
                    SET embedding = ?,
                        embedding_model = ?,
                        embedding_dimension = ?,
                        embedding_date = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    embedding.astype('float32').tobytes(),
                    new_model,
                    len(embedding),
                    chunk['id']
                ))
            
            self.db.conn.commit()
            logger.debug(f"Actualizados {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    def _get_or_create_generic_manual(self) -> int:
        """Obtener o crear un manual genérico para chunks sin manual específico"""
        
        cursor = self.db.conn.execute(
            "SELECT id FROM manuals WHERE name = 'generic_manual' LIMIT 1"
        )
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        # Crear manual genérico
        return self.db.insert_manual({
            'name': 'generic_manual',
            'filename': 'generic.pdf',
            'document_type': 'other',
            'processing_status': 'completed'
        })
    
    def migrate_from_chromadb(self):
        """Migrar datos existentes de ChromaDB a SQLite"""
        
        logger.info("Migrando datos de ChromaDB a SQLite...")
        
        # Obtener todos los documentos de ChromaDB
        # Esto depende de tu implementación actual
        collection = self.vector_manager.collection
        
        # ChromaDB no tiene un método directo para obtener todos los documentos
        # Podrías necesitar hacer una búsqueda con un límite alto
        results = collection.get(limit=10000)  # Ajusta según tu caso
        
        if not results['ids']:
            logger.info("No hay documentos en ChromaDB para migrar")
            return
        
        # Preparar para inserción
        texts = results['documents']
        metadatas = results['metadatas']
        embeddings = results['embeddings'] if 'embeddings' in results else None
        
        # Insertar en SQLite
        self.add_documents(texts, metadatas, results['ids'], embeddings)
        
        logger.info(f"Migrados {len(texts)} documentos de ChromaDB a SQLite")
    
    def close(self):
        """Cerrar conexiones"""
        self.db.close()