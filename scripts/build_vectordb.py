import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from extractors.text_processor import TextProcessor
from vectorstore.vector_manager import VectorManager
from vectorstore.indexing import IndexingSystem
from models.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBBuilder:
    """Constructor de base de datos vectorial desde datos procesados"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_manager = VectorManager(config)
        self.text_processor = TextProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.indexing_system = IndexingSystem(self.vector_manager)
    
    def build_from_processed_data(self, force_rebuild: bool = False):
        """Construir base vectorial desde datos ya procesados"""
        
        # Verificar si ya existe la base de datos
        if self._database_exists() and not force_rebuild:
            logger.info("La base de datos ya existe. Use --force para reconstruir.")
            return
        
        if force_rebuild:
            logger.info("Reconstruyendo base de datos vectorial...")
            self._clear_database()
        
        # Cargar todos los metadatos
        metadata_files = list((self.config.PROCESSED_DIR / 'metadata').glob('*_metadata.json'))
        
        if not metadata_files:
            logger.error("No se encontraron archivos de metadatos. Ejecute process_manuals.py primero.")
            return
        
        logger.info(f"Encontrados {len(metadata_files)} archivos de metadatos")
        
        # Procesar cada manual
        for metadata_file in tqdm(metadata_files, desc="Procesando manuales"):
            self._process_manual_metadata(metadata_file)
        
        # Construir índices
        logger.info("Construyendo índices...")
        self.indexing_system.build_indices()
        self.indexing_system.save_indices(self.config.VECTOR_DB_DIR / "indices.json")
        
        # Mostrar estadísticas
        self._show_statistics()
    
    def _process_manual_metadata(self, metadata_file: Path):
        """Procesar metadatos de un manual"""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        manual_name = metadata['manual_name']
        logger.info(f"Procesando manual: {manual_name}")
        
        all_documents = []
        
        # Procesar páginas de texto
        for page_data in metadata.get('pages', []):
            # Leer archivo de texto
            text_file = self.config.PROCESSED_DIR / 'texts' / f"{manual_name}_page_{page_data['page_number']}.txt"
            
            if text_file.exists():
                text = text_file.read_text(encoding='utf-8')
                
                # Crear chunks
                documents = self.text_processor.process_text(
                    text,
                    page_data['metadata']
                )
                all_documents.extend(documents)
        
        # Procesar tablas
        for table_data in metadata.get('tables', []):
            # El texto de la tabla ya está en los metadatos
            table_text = table_data.get('table_text', '')
            
            if table_text:
                table_docs = self.text_processor.process_table_text(
                    table_text,
                    table_data
                )
                all_documents.extend(table_docs)
        
        # Añadir documentos a la base vectorial
        if all_documents:
            self.vector_manager.add_documents(all_documents)
            logger.info(f"  - Añadidos {len(all_documents)} chunks")
        
        # Procesar referencias de imágenes
        images = metadata.get('images', [])
        if images:
            self.vector_manager.add_image_references(images)
            logger.info(f"  - Añadidas {len(images)} referencias de imágenes")
    
    def update_manual(self, manual_name: str):
        """Actualizar un manual específico en la base de datos"""
        metadata_file = self.config.PROCESSED_DIR / 'metadata' / f"{manual_name}_metadata.json"
        
        if not metadata_file.exists():
            logger.error(f"No se encontraron metadatos para el manual: {manual_name}")
            return
        
        # Eliminar documentos existentes del manual
        self._remove_manual_from_db(manual_name)
        
        # Procesar y añadir de nuevo
        self._process_manual_metadata(metadata_file)
        
        # Reconstruir índices
        self.indexing_system.build_indices()
        self.indexing_system.save_indices(self.config.VECTOR_DB_DIR / "indices.json")
        
        logger.info(f"Manual {manual_name} actualizado exitosamente")
    
    def _database_exists(self) -> bool:
        """Verificar si la base de datos ya existe"""
        return (self.config.VECTOR_DB_DIR / "chroma").exists()
    
    def _clear_database(self):
        """Limpiar base de datos existente"""
        try:
            self.vector_manager.client.delete_collection(self.config.COLLECTION_NAME)
            self.vector_manager.client.delete_collection(f"{self.config.COLLECTION_NAME}_images")
        except:
            pass
        
        # Recrear colecciones
        self.vector_manager = VectorManager(self.config)
    
    def _remove_manual_from_db(self, manual_name: str):
        """Eliminar todos los documentos de un manual de la base de datos"""
        # Obtener IDs de documentos del manual
        results = self.vector_manager.collection.get(
            where={"manual_name": manual_name}
        )
        
        if results['ids']:
            self.vector_manager.collection.delete(ids=results['ids'])
            logger.info(f"Eliminados {len(results['ids'])} documentos del manual {manual_name}")
        
        # Hacer lo mismo para imágenes
        image_results = self.vector_manager.image_collection.get(
            where={"manual_name": manual_name}
        )
        
        if image_results['ids']:
            self.vector_manager.image_collection.delete(ids=image_results['ids'])
            logger.info(f"Eliminadas {len(image_results['ids'])} referencias de imágenes")
    
    def _show_statistics(self):
        """Mostrar estadísticas de la base de datos"""
        # Estadísticas de texto
        text_count = self.vector_manager.collection.count()
        image_count = self.vector_manager.image_collection.count()
        
        logger.info("\n=== Estadísticas de la Base de Datos ===")
        logger.info(f"Total de chunks de texto: {text_count}")
        logger.info(f"Total de referencias de imágenes: {image_count}")
        
        # Estadísticas por manual
        manuals = self.vector_manager.get_manual_list()
        logger.info(f"\nManuales indexados: {len(manuals)}")
        
        for manual in manuals:
            manual_stats = self.vector_manager.create_manual_index(manual)
            logger.info(f"\n{manual}:")
            logger.info(f"  - Documentos: {manual_stats['total_documents']}")
            logger.info(f"  - Páginas: {len(manual_stats['pages'])}")
            logger.info(f"  - Secciones: {len(manual_stats['sections'])}")
            logger.info(f"  - Capítulos: {len(manual_stats['chapters'])}")

def main():
    parser = argparse.ArgumentParser(
        description='Construir base de datos vectorial desde datos procesados'
    )
    parser.add_argument('--force', action='store_true',
                       help='Forzar reconstrucción de la base de datos')
    parser.add_argument('--update-manual', type=str,
                       help='Actualizar un manual específico')
    parser.add_argument('--show-stats', action='store_true',
                       help='Mostrar solo estadísticas')
    
    args = parser.parse_args()
    
    # Configuración
    config = Config()
    builder = VectorDBBuilder(config)
    
    if args.show_stats:
        builder._show_statistics()
    elif args.update_manual:
        builder.update_manual(args.update_manual)
    else:
        builder.build_from_processed_data(force_rebuild=args.force)

if __name__ == "__main__":
    main()