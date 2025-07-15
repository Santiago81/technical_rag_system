#!/usr/bin/env python3
"""
Script para construir la base de datos vectorial ChromaDB desde SQLite
Utiliza el SQLiteVectorAdapter para sincronizar datos entre SQLite y ChromaDB
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import time

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from vectorstore.sqlite_adapter import SQLiteVectorAdapter
from database.sqlite_manager import SQLiteRAGManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDBSQLiteBuilder:
    """Constructor de base de datos vectorial desde SQLite"""
    
    def __init__(self, config: Config):
        self.config = config
        self.adapter = SQLiteVectorAdapter(config)
        
        # Conectar a SQLite para estadísticas
        db_path = str(config.DATA_DIR / 'sqlite' / 'manuals.db')
        self.db = SQLiteRAGManager(db_path)
    
    def build_from_sqlite(self, force_rebuild: bool = False, manual_id: Optional[int] = None):
        """
        Construir o actualizar ChromaDB desde SQLite
        
        Args:
            force_rebuild: Reconstruir completamente la base vectorial
            manual_id: Procesar solo un manual específico (opcional)
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("CONSTRUCCIÓN DE BASE DE DATOS VECTORIAL")
        logger.info("="*60)
        
        # Verificar estado actual
        stats = self._get_current_stats()
        logger.info(f"\nEstado actual:")
        logger.info(f"  - Chunks en SQLite: {stats['total_chunks']}")
        logger.info(f"  - Chunks con embeddings: {stats['chunks_with_embeddings']}")
        logger.info(f"  - Chunks sin embeddings: {stats['chunks_without_embeddings']}")
        
        if force_rebuild:
            logger.info("\n⚠️  Modo --force: Reconstruyendo base vectorial completa")
            self._clear_chromadb()
        
        # Proceso principal
        if manual_id:
            logger.info(f"\nProcesando manual ID: {manual_id}")
            self._process_single_manual(manual_id)
        else:
            logger.info("\nProcesando todos los manuales...")
            self._process_all_manuals()
        
        # Construir índices
        logger.info("\nConstruyendo índices de búsqueda...")
        self._build_indices()
        
        # Mostrar estadísticas finales
        duration = time.time() - start_time
        self._show_final_stats(duration)
    
    def _process_all_manuals(self):
        """Procesar todos los manuales de la base de datos"""
        # Obtener lista de manuales
        manuals = self.db.conn.execute("""
            SELECT id, name, total_chunks 
            FROM manuals 
            ORDER BY id
        """).fetchall()
        
        for manual_id, manual_name, total_chunks in manuals:
            logger.info(f"\n📖 Procesando: {manual_name}")
            logger.info(f"   Total chunks: {total_chunks or 0}")
            
            # Procesar chunks del manual
            processed, skipped = self._process_manual_chunks(manual_id)
            
            logger.info(f"   ✓ Procesados: {processed}")
            if skipped > 0:
                logger.info(f"   → Ya tenían embeddings: {skipped}")
    
    def _process_single_manual(self, manual_id: int):
        """Procesar un manual específico"""
        manual = self.db.get_manual(manual_id)
        if not manual:
            logger.error(f"Manual {manual_id} no encontrado")
            return
        
        logger.info(f"Manual: {manual['name']}")
        processed, skipped = self._process_manual_chunks(manual_id)
        logger.info(f"Chunks procesados: {processed}, omitidos: {skipped}")
    
    def _process_manual_chunks(self, manual_id: int) -> tuple[int, int]:
        """
        Procesar chunks de un manual
        
        Returns:
            (chunks_procesados, chunks_omitidos)
        """
        # Obtener chunks del manual
        chunks = self.db.conn.execute("""
            SELECT 
                c.id,
                c.chunk_text,
                c.chunk_text_processed,
                c.chunk_index,
                c.start_page,
                c.end_page,
                c.embedding,
                c.keywords,
                c.entities,
                c.importance_score,
                m.name as manual_name,
                m.manufacturer,
                m.model,
                m.document_type
            FROM content_chunks c
            JOIN manuals m ON c.manual_id = m.id
            WHERE c.manual_id = ?
            ORDER BY c.chunk_index
        """, (manual_id,)).fetchall()
        
        if not chunks:
            logger.warning(f"  No se encontraron chunks para el manual {manual_id}")
            return 0, 0
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        processed = 0
        skipped = 0
        
        for chunk in chunks:
            chunk_dict = dict(chunk)
            
            # Usar texto procesado si está disponible, sino el original
            text = chunk_dict['chunk_text_processed'] or chunk_dict['chunk_text']
            
            # Preparar documento
            documents.append(text)
            
            # Preparar metadata
            metadata = {
                'chunk_id': chunk_dict['id'],
                'manual_id': manual_id,
                'manual_name': chunk_dict['manual_name'],
                'manufacturer': chunk_dict['manufacturer'] or 'Unknown',
                'model': chunk_dict['model'] or 'Unknown',
                'document_type': chunk_dict['document_type'] or 'technical',
                'chunk_index': chunk_dict['chunk_index'],
                'start_page': chunk_dict['start_page'],
                'end_page': chunk_dict['end_page'],
                'keywords': chunk_dict['keywords'] or '',
                'entities': chunk_dict['entities'] or '',
                'importance_score': chunk_dict['importance_score'] or 1.0
            }
            metadatas.append(metadata)
            
            # ID único para ChromaDB
            ids.append(f"chunk_{chunk_dict['id']}")
            
            # Verificar si ya tiene embedding
            if chunk_dict['embedding']:
                embeddings.append(None)  # El adapter generará si es necesario
                skipped += 1
            else:
                embeddings.append(None)  # Se generará automáticamente
                processed += 1
        
        # Añadir a ChromaDB a través del adapter
        if documents:
            try:
                self.adapter.add_documents(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                
                # Actualizar embeddings en SQLite si se generaron nuevos
                if processed > 0:
                    logger.info(f"   Actualizando embeddings en SQLite...")
                    self.adapter.update_embeddings(manual_id=manual_id)
                    
            except Exception as e:
                logger.error(f"Error procesando chunks del manual {manual_id}: {e}")
                return 0, skipped
        
        return processed, skipped
    
    def _build_indices(self):
        """Construir índices para búsqueda eficiente"""
        try:
            # El sistema de indexación se construye automáticamente en el adapter
            if hasattr(self.adapter.vector_manager, 'indexing_system'):
                self.adapter.vector_manager.indexing_system.build_indices()
                
                # Guardar índices
                indices_path = self.config.VECTOR_DB_DIR / "indices.json"
                self.adapter.vector_manager.indexing_system.save_indices(indices_path)
                logger.info(f"✓ Índices guardados en: {indices_path}")
            else:
                logger.info("→ Sistema de indexación no disponible")
                
        except Exception as e:
            logger.warning(f"No se pudieron construir índices: {e}")
    
    def _clear_chromadb(self):
        """Limpiar la base de datos ChromaDB"""
        try:
            # Eliminar colecciones existentes
            client = self.adapter.vector_manager.client
            
            try:
                client.delete_collection(self.config.CHROMA_COLLECTION_NAME)
                logger.info("✓ Colección de texto eliminada")
            except:
                pass
            
            try:
                client.delete_collection(f"{self.config.CHROMA_COLLECTION_NAME}_images")
                logger.info("✓ Colección de imágenes eliminada")
            except:
                pass
            
            # Reinicializar adapter
            self.adapter = SQLiteVectorAdapter(self.config)
            
        except Exception as e:
            logger.error(f"Error limpiando ChromaDB: {e}")
    
    def _get_current_stats(self) -> dict:
        """Obtener estadísticas actuales"""
        stats = {}
        
        # Total de chunks
        result = self.db.conn.execute("SELECT COUNT(*) FROM content_chunks").fetchone()
        stats['total_chunks'] = result[0] if result else 0
        
        # Chunks con embeddings
        result = self.db.conn.execute(
            "SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        stats['chunks_with_embeddings'] = result[0] if result else 0
        
        # Chunks sin embeddings
        stats['chunks_without_embeddings'] = stats['total_chunks'] - stats['chunks_with_embeddings']
        
        return stats
    
    def _show_final_stats(self, duration: float):
        """Mostrar estadísticas finales"""
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE CONSTRUCCIÓN")
        logger.info("="*60)
        
        # Estadísticas de ChromaDB
        try:
            collection_stats = self.adapter.get_collection_stats()
            logger.info(f"\n📊 Base Vectorial ChromaDB:")
            logger.info(f"   - Documentos en colección principal: {collection_stats.get('text_count', 0)}")
            logger.info(f"   - Referencias de imágenes: {collection_stats.get('image_count', 0)}")
            
            # Manuales indexados
            manuals = self.adapter.vector_manager.get_manual_list()
            logger.info(f"\n📚 Manuales indexados: {len(manuals)}")
            for manual in sorted(manuals):
                logger.info(f"   - {manual}")
            
        except Exception as e:
            logger.warning(f"No se pudieron obtener estadísticas de ChromaDB: {e}")
        
        # Estadísticas de SQLite
        stats = self._get_current_stats()
        logger.info(f"\n💾 Base de Datos SQLite:")
        logger.info(f"   - Total chunks: {stats['total_chunks']}")
        logger.info(f"   - Con embeddings: {stats['chunks_with_embeddings']}")
        logger.info(f"   - Sin embeddings: {stats['chunks_without_embeddings']}")
        
        logger.info(f"\n⏱️  Tiempo total: {duration:.1f} segundos")
        logger.info("="*60 + "\n")
    
    def verify_sync(self):
        """Verificar sincronización entre SQLite y ChromaDB"""
        logger.info("\nVerificando sincronización SQLite ↔ ChromaDB...")
        
        # Chunks en SQLite
        sqlite_chunks = self.db.conn.execute("SELECT COUNT(*) FROM content_chunks").fetchone()[0]
        
        # Documentos en ChromaDB
        try:
            chroma_docs = self.adapter.vector_manager.collection.count()
            
            logger.info(f"Chunks en SQLite: {sqlite_chunks}")
            logger.info(f"Documentos en ChromaDB: {chroma_docs}")
            
            if sqlite_chunks == chroma_docs:
                logger.info("✅ Las bases de datos están sincronizadas")
            else:
                logger.warning(f"⚠️  Diferencia de {abs(sqlite_chunks - chroma_docs)} documentos")
                
        except Exception as e:
            logger.error(f"Error verificando ChromaDB: {e}")
    
    def close(self):
        """Cerrar conexiones"""
        self.db.close()


def main():
    parser = argparse.ArgumentParser(
        description='Construir base de datos vectorial ChromaDB desde SQLite'
    )
    
    parser.add_argument('--force', action='store_true',
                       help='Reconstruir completamente la base vectorial')
    parser.add_argument('--manual-id', type=int,
                       help='Procesar solo un manual específico por ID')
    parser.add_argument('--verify', action='store_true',
                       help='Verificar sincronización entre SQLite y ChromaDB')
    parser.add_argument('--stats', action='store_true',
                       help='Mostrar solo estadísticas actuales')
    
    args = parser.parse_args()
    
    # Configuración
    config = Config()
    
    # Validar que existe la base SQLite
    sqlite_path = config.DATA_DIR / 'sqlite' / 'manuals.db'
    if not sqlite_path.exists():
        logger.error(f"No se encontró la base de datos SQLite en: {sqlite_path}")
        logger.error("Ejecute primero: python scripts/process_manuals_sqlite.py")
        sys.exit(1)
    
    # Crear builder
    builder = VectorDBSQLiteBuilder(config)
    
    try:
        if args.stats:
            # Solo mostrar estadísticas
            stats = builder._get_current_stats()
            builder._show_final_stats(0)
        elif args.verify:
            # Verificar sincronización
            builder.verify_sync()
        else:
            # Construir base vectorial
            builder.build_from_sqlite(
                force_rebuild=args.force,
                manual_id=args.manual_id
            )
            
            # Verificar al final
            if not args.manual_id:
                builder.verify_sync()
    
    finally:
        builder.close()


if __name__ == "__main__":
    main()