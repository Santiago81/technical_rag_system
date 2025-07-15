#!/usr/bin/env python3
"""
Script para extraer imágenes y tablas de manuales ya procesados
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from database.sqlite_manager import SQLiteRAGManager
from extractors.sqlite_extractors import SQLiteImageExtractor, SQLiteTableExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_images_and_tables(manual_id: int):
    """Extraer imágenes y tablas de un manual específico"""
    
    config = Config()
    db_path = str(config.DATA_DIR / 'sqlite' / 'manuals.db')
    db = SQLiteRAGManager(db_path)
    
    try:
        # Obtener información del manual
        manual = db.get_manual(manual_id)
        if not manual:
            logger.error(f"Manual {manual_id} no encontrado")
            return
        
        pdf_path = Path(manual['file_path'])
        if not pdf_path.exists():
            # Intentar con path relativo
            pdf_path = config.BASE_DIR / manual['file_path']
            if not pdf_path.exists():
                logger.error(f"PDF no encontrado: {manual['file_path']}")
                return
        
        logger.info(f"Procesando manual: {manual['name']}")
        
        # Extraer imágenes
        logger.info("Extrayendo imágenes...")
        image_extractor = SQLiteImageExtractor(db, config.PROCESSED_DIR / 'images')
        image_results = image_extractor.extract_and_store(pdf_path, manual_id)
        logger.info(f"  → Extraídas {image_results['total_images']} imágenes")
        
        # Extraer tablas
        logger.info("Extrayendo tablas...")
        table_extractor = SQLiteTableExtractor(db, config.PROCESSED_DIR / 'tables')
        table_results = table_extractor.extract_and_store(pdf_path, manual_id)
        logger.info(f"  → Extraídas {table_results['total_tables']} tablas")
        
        # Actualizar estadísticas en el manual
        db.conn.execute("""
            UPDATE manuals 
            SET total_images = ?, total_tables = ?
            WHERE id = ?
        """, (image_results['total_images'], table_results['total_tables'], manual_id))
        db.conn.commit()
        
        logger.info("Proceso completado exitosamente")
        
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extract_images_tables.py <manual_id>")
        sys.exit(1)
    
    manual_id = int(sys.argv[1])
    extract_images_and_tables(manual_id)