"""
Script para migrar datos existentes del sistema de archivos a SQLite
"""
import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
from tqdm import tqdm

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from database.sqlite_manager import SQLiteRAGManager
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMigrator:
    """Migrador de datos existentes a SQLite"""
    
    def __init__(self, config: Config, db_path: str):
        self.config = config
        self.db = SQLiteRAGManager(db_path)
        self.processed_dir = config.PROCESSED_DIR
        self.stats = {
            'manuals': 0,
            'content_blocks': 0,
            'chunks': 0,
            'images': 0,
            'tables': 0,
            'errors': []
        }
    
    def migrate_all(self, verify: bool = True):
        """Migrar todos los datos procesados a SQLite"""
        logger.info("Iniciando migración de datos a SQLite...")
        
        # Encontrar todos los manuales procesados
        text_dir = self.processed_dir / 'texts'
        if not text_dir.exists():
            logger.error(f"No se encontró el directorio de textos: {text_dir}")
            return
        
        # Listar manuales
        manual_dirs = [d for d in text_dir.iterdir() if d.is_dir()]
        
        if not manual_dirs:
            logger.warning("No se encontraron manuales para migrar")
            return
        
        logger.info(f"Encontrados {len(manual_dirs)} manuales para migrar")
        
        # Migrar cada manual
        for manual_dir in tqdm(manual_dirs, desc="Migrando manuales"):
            try:
                self._migrate_manual(manual_dir.name)
            except Exception as e:
                logger.error(f"Error migrando {manual_dir.name}: {e}")
                self.stats['errors'].append({
                    'manual': manual_dir.name,
                    'error': str(e)
                })
        
        # Mostrar estadísticas
        self._show_stats()
        
        # Verificar integridad si se solicita
        if verify:
            self._verify_migration()
    
    def _migrate_manual(self, manual_name: str):
        """Migrar un manual específico"""
        logger.debug(f"Migrando manual: {manual_name}")
        
        # Buscar archivos relacionados
        text_dir = self.processed_dir / 'texts' / manual_name
        metadata_dir = self.processed_dir / 'metadata' / manual_name
        images_dir = self.processed_dir / 'images' / manual_name
        tables_dir = self.processed_dir / 'tables' / manual_name
        
        # 1. Crear entrada del manual
        manual_metadata = self._load_manual_metadata(metadata_dir)
        manual_data = {
            'name': manual_name,
            'filename': manual_metadata.get('filename', f"{manual_name}.pdf"),
            'file_path': manual_metadata.get('file_path'),
            'total_pages': manual_metadata.get('total_pages', 0),
            'language': manual_metadata.get('language', 'es'),
            'document_type': manual_metadata.get('document_type', 'technical'),
            'processing_status': 'completed',
            'metadata_json': manual_metadata
        }
        
        manual_id = self.db.insert_manual(manual_data)
        self.stats['manuals'] += 1
        
        # 2. Migrar contenido de texto
        self._migrate_text_content(manual_id, text_dir, metadata_dir)
        
        # 3. Migrar chunks
        self._migrate_chunks(manual_id, metadata_dir)
        
        # 4. Migrar imágenes
        if images_dir.exists():
            self._migrate_images(manual_id, images_dir, metadata_dir)
        
        # 5. Migrar tablas
        if tables_dir.exists():
            self._migrate_tables(manual_id, tables_dir, metadata_dir)
        
        # 6. Migrar análisis si existe
        self._migrate_analysis(manual_id, metadata_dir)
    
    def _load_manual_metadata(self, metadata_dir: Path) -> Dict:
        """Cargar metadata del manual"""
        metadata = {}
        
        # Buscar archivo de metadata principal
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        if metadata_files:
            with open(metadata_files[0], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Buscar información adicional
        info_file = metadata_dir / "processing_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                metadata.update(info)
        
        return metadata
    
    def _migrate_text_content(self, manual_id: int, text_dir: Path, metadata_dir: Path):
        """Migrar contenido de texto"""
        # Buscar archivos de texto por página
        text_files = sorted(text_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No se encontraron archivos de texto para manual {manual_id}")
            return
        
        blocks_to_insert = []
        
        for text_file in text_files:
            # Extraer número de página del nombre
            page_num = self._extract_page_number(text_file.name)
            if page_num is None:
                continue
            
            # Leer contenido
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Buscar metadata de la página si existe
            page_metadata = {}
            page_meta_file = metadata_dir / f"page_{page_num}_metadata.json"
            if page_meta_file.exists():
                with open(page_meta_file, 'r', encoding='utf-8') as f:
                    page_metadata = json.load(f)
            
            # Crear bloques (simplificado - en realidad deberías preservar la estructura original)
            blocks_to_insert.append({
                'manual_id': manual_id,
                'page_number': page_num,
                'block_index': 0,
                'block_type': 'text',
                'content': content,
                'section': page_metadata.get('section'),
                'chapter': page_metadata.get('chapter'),
                'char_count': len(content),
                'word_count': len(content.split())
            })
        
        # Insertar bloques
        if blocks_to_insert:
            block_ids = self.db.insert_content_blocks(blocks_to_insert)
            self.stats['content_blocks'] += len(block_ids)
            logger.debug(f"Migrados {len(block_ids)} bloques de contenido")
    
    def _migrate_chunks(self, manual_id: int, metadata_dir: Path):
        """Migrar chunks existentes"""
        chunks_file = metadata_dir / "chunks_metadata.json"
        
        if not chunks_file.exists():
            logger.warning(f"No se encontró archivo de chunks para manual {manual_id}")
            return
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks_to_insert = []
        
        for i, chunk in enumerate(chunks_data.get('chunks', [])):
            chunks_to_insert.append({
                'manual_id': manual_id,
                'chunk_index': i,
                'chunk_text': chunk['text'],
                'chunk_size': len(chunk['text']),
                'start_page': chunk.get('start_page', 1),
                'end_page': chunk.get('end_page', chunk.get('start_page', 1)),
                'metadata': chunk.get('metadata', {})
            })
        
        # Insertar chunks
        if chunks_to_insert:
            from database.sqlite_manager import ChunkData
            chunk_objects = [ChunkData(**chunk) for chunk in chunks_to_insert]
            chunk_ids = self.db.insert_chunks_batch(chunk_objects)
            self.stats['chunks'] += len(chunk_ids)
            logger.debug(f"Migrados {len(chunk_ids)} chunks")
    
    def _migrate_images(self, manual_id: int, images_dir: Path, metadata_dir: Path):
        """Migrar imágenes"""
        # Buscar metadata de imágenes
        images_metadata_file = metadata_dir / "images_metadata.json"
        
        if not images_metadata_file.exists():
            logger.warning(f"No se encontró metadata de imágenes para manual {manual_id}")
            return
        
        with open(images_metadata_file, 'r', encoding='utf-8') as f:
            images_metadata = json.load(f)
        
        images_to_insert = []
        
        for img_meta in images_metadata.get('images', []):
            # Verificar que la imagen existe
            img_path = images_dir / img_meta['image_filename']
            if not img_path.exists():
                logger.warning(f"Imagen no encontrada: {img_path}")
                continue
            
            # Calcular ruta relativa
            relative_path = img_path.relative_to(self.processed_dir / 'images')
            
            images_to_insert.append({
                'manual_id': manual_id,
                'page_number': img_meta['page_number'],
                'image_index': img_meta.get('image_index', 0),
                'image_type': img_meta.get('content_type', 'raster'),
                'file_path': str(relative_path),
                'file_format': img_path.suffix[1:],
                'file_size': img_path.stat().st_size,
                'width': img_meta.get('width'),
                'height': img_meta.get('height'),
                'ocr_text': img_meta.get('ocr_text'),
                'file_hash': self._calculate_file_hash(str(img_path))
            })
        
        # Insertar imágenes
        if images_to_insert:
            image_ids = self.db.insert_images_batch(images_to_insert)
            self.stats['images'] += len(image_ids)
            logger.debug(f"Migradas {len(image_ids)} imágenes")
    
    def _migrate_tables(self, manual_id: int, tables_dir: Path, metadata_dir: Path):
        """Migrar tablas"""
        # Buscar archivos CSV
        csv_files = list(tables_dir.glob("*.csv"))
        
        if not csv_files:
            logger.debug(f"No se encontraron tablas para manual {manual_id}")
            return
        
        tables_to_insert = []
        
        for csv_file in csv_files:
            # Extraer información del nombre
            page_num = self._extract_page_number(csv_file.name)
            if page_num is None:
                continue
            
            # Calcular ruta relativa
            relative_path = csv_file.relative_to(self.processed_dir / 'tables')
            
            # Leer CSV para obtener información
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                tables_to_insert.append({
                    'manual_id': manual_id,
                    'page_number': page_num,
                    'table_index': 0,
                    'csv_path': str(relative_path),
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'headers': list(df.columns),
                    'table_content': df.head(3).to_string() if len(df) > 0 else ""
                })
            except Exception as e:
                logger.warning(f"Error leyendo tabla {csv_file}: {e}")
        
        # Insertar tablas
        if tables_to_insert:
            table_ids = self.db.insert_tables_batch(tables_to_insert)
            self.stats['tables'] += len(table_ids)
            logger.debug(f"Migradas {len(table_ids)} tablas")
    
    def _migrate_analysis(self, manual_id: int, metadata_dir: Path):
        """Migrar análisis del documento si existe"""
        analysis_file = metadata_dir / "document_analysis.json"
        
        if not analysis_file.exists():
            return
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # Adaptar al formato de la base de datos
        db_analysis = {
            'manual_id': manual_id,
            'document_type': analysis.get('document_type', 'unknown'),
            'extraction_strategy': analysis.get('extraction_strategy', {})
        }
        
        # Agregar estadísticas si están disponibles
        if 'page_analysis' in analysis:
            pa = analysis['page_analysis']
            db_analysis.update({
                'avg_text_density': pa.get('avg_text_density', 0),
                'avg_images_per_page': pa.get('avg_images_per_page', 0),
                'avg_vector_graphics': pa.get('avg_vector_graphics', 0),
                'table_frequency': pa.get('table_frequency', 0)
            })
        
        self.db.save_document_analysis(db_analysis)
    
    def _extract_page_number(self, filename: str) -> Optional[int]:
        """Extraer número de página del nombre de archivo"""
        import re
        
        # Buscar patrones comunes
        patterns = [
            r'page_(\d+)',
            r'p(\d+)_',
            r'_(\d+)\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash MD5 de archivo"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _show_stats(self):
        """Mostrar estadísticas de migración"""
        print("\n" + "="*60)
        print("ESTADÍSTICAS DE MIGRACIÓN")
        print("="*60)
        print(f"Manuales migrados: {self.stats['manuals']}")
        print(f"Bloques de contenido: {self.stats['content_blocks']}")
        print(f"Chunks creados: {self.stats['chunks']}")
        print(f"Imágenes migradas: {self.stats['images']}")
        print(f"Tablas migradas: {self.stats['tables']}")
        print(f"Errores: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\nErrores encontrados:")
            for error in self.stats['errors'][:5]:
                print(f"  - {error['manual']}: {error['error']}")
            
            if len(self.stats['errors']) > 5:
                print(f"  ... y {len(self.stats['errors']) - 5} errores más")
        
        print("="*60)
    
    def _verify_migration(self):
        """Verificar integridad de la migración"""
        logger.info("\nVerificando integridad de la migración...")
        
        # Contar registros en la base de datos
        checks = [
            ("Manuales", "SELECT COUNT(*) FROM manuals"),
            ("Bloques de contenido", "SELECT COUNT(*) FROM content_blocks"),
            ("Chunks", "SELECT COUNT(*) FROM content_chunks"),
            ("Imágenes", "SELECT COUNT(*) FROM images"),
            ("Tablas", "SELECT COUNT(*) FROM extracted_tables")
        ]
        
        print("\nVerificación de base de datos:")
        for name, query in checks:
            count = self.db.conn.execute(query).fetchone()[0]
            print(f"  {name}: {count}")
        
        # Verificar que los archivos referenciados existen
        logger.info("Verificando archivos referenciados...")
        
        # Verificar imágenes
        missing_images = 0
        cursor = self.db.conn.execute("SELECT file_path FROM images")
        for row in cursor:
            img_path = self.processed_dir / 'images' / row['file_path']
            if not img_path.exists():
                missing_images += 1
        
        if missing_images > 0:
            logger.warning(f"Imágenes faltantes: {missing_images}")
        
        # Verificar tablas
        missing_tables = 0
        cursor = self.db.conn.execute("SELECT csv_path FROM extracted_tables WHERE csv_path IS NOT NULL")
        for row in cursor:
            table_path = self.processed_dir / 'tables' / row['csv_path']
            if not table_path.exists():
                missing_tables += 1
        
        if missing_tables > 0:
            logger.warning(f"Tablas CSV faltantes: {missing_tables}")
        
        print("\nVerificación completada.")


def main():
    parser = argparse.ArgumentParser(description='Migrar datos existentes a SQLite')
    parser.add_argument('--source', type=str, help='Directorio fuente de datos procesados')
    parser.add_argument('--target', type=str, help='Ruta de la base de datos SQLite destino')
    parser.add_argument('--verify', action='store_true', help='Verificar integridad después de migrar')
    parser.add_argument('--manual', type=str, help='Migrar solo un manual específico')
    
    args = parser.parse_args()
    
    # Configuración
    config = Config()
    
    # Sobrescribir directorio fuente si se especifica
    if args.source:
        config.PROCESSED_DIR = Path(args.source)
    
    # Determinar base de datos destino
    db_path = args.target or str(config.DATA_DIR / 'sqlite' / 'manuals.db')
    
    # Crear migrador
    migrator = DataMigrator(config, db_path)
    
    # Ejecutar migración
    if args.manual:
        # Migrar manual específico
        try:
            migrator._migrate_manual(args.manual)
            print(f"Manual {args.manual} migrado exitosamente")
        except Exception as e:
            print(f"Error migrando manual: {e}")
    else:
        # Migrar todo
        migrator.migrate_all(verify=args.verify)
    
    # Cerrar conexión
    migrator.db.close()


if __name__ == "__main__":
    main()