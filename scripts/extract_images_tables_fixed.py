#!/usr/bin/env python3
"""
Script mejorado para extraer imágenes y tablas de manuales ya procesados
Maneja problemas de conversión de color y campos faltantes
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from database.sqlite_manager import SQLiteRAGManager
import logging
import fitz  # PyMuPDF
from PIL import Image
import io
import hashlib
import pandas as pd
import tabula

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedImageExtractor:
    """Extractor de imágenes con manejo mejorado de formatos"""
    
    def __init__(self, db, output_dir: Path):
        self.db = db
        self.output_dir = output_dir
        
    def extract_and_store(self, pdf_path: Path, manual_id: int):
        """Extraer y almacenar imágenes"""
        
        # Crear directorio de salida
        manual_dir = self.output_dir / str(manual_id)
        manual_dir.mkdir(parents=True, exist_ok=True)
        
        images_extracted = []
        
        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Manejar diferentes espacios de color
                        if pix.colorspace is None:
                            logger.warning(f"Imagen sin espacio de color en página {page_num+1}")
                            continue
                            
                        # Convertir a RGB si es necesario
                        if pix.n - pix.alpha >= 4:  # CMYK u otro
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Guardar como PNG
                        img_path = manual_dir / f"page_{page_num+1}_img_{img_index+1}.png"
                        pix.save(str(img_path))
                        
                        # Calcular hash
                        with open(img_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        # Preparar datos para inserción (sin campo dpi)
                        image_data = {
                            'manual_id': manual_id,
                            'page_number': page_num + 1,
                            'image_index': img_index + 1,
                            'image_type': 'raster',
                            'file_path': str(img_path.relative_to(self.output_dir)),
                            'file_format': 'png',
                            'file_size': img_path.stat().st_size,
                            'file_hash': file_hash,
                            'width': pix.width,
                            'height': pix.height,
                            'color_space': 'RGB'
                        }
                        
                        images_extracted.append(image_data)
                        pix = None  # Liberar memoria
                        
                    except Exception as e:
                        logger.warning(f"Error procesando imagen {img_index+1} en página {page_num+1}: {e}")
                        continue
        
        # Insertar en base de datos
        if images_extracted:
            for img_data in images_extracted:
                try:
                    # Insertar manualmente sin usar insert_images_batch
                    self.db.conn.execute("""
                        INSERT INTO images (
                            manual_id, page_number, image_index, image_type,
                            file_path, file_format, file_size, file_hash,
                            width, height, color_space
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        img_data['manual_id'], img_data['page_number'], 
                        img_data['image_index'], img_data['image_type'],
                        img_data['file_path'], img_data['file_format'],
                        img_data['file_size'], img_data['file_hash'],
                        img_data['width'], img_data['height'], img_data['color_space']
                    ))
                except Exception as e:
                    logger.warning(f"Error insertando imagen: {e}")
            
            self.db.conn.commit()
        
        logger.info(f"Extraídas {len(images_extracted)} imágenes")
        return {'total_images': len(images_extracted)}

class FixedTableExtractor:
    """Extractor de tablas"""
    
    def __init__(self, db, output_dir: Path):
        self.db = db
        self.output_dir = output_dir
        
    def extract_and_store(self, pdf_path: Path, manual_id: int):
        """Extraer y almacenar tablas"""
        
        # Crear directorio de salida
        manual_dir = self.output_dir / str(manual_id)
        manual_dir.mkdir(parents=True, exist_ok=True)
        
        tables_extracted = []
        
        try:
            # Intentar con tabula
            tables = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True)
            
            for table_index, df in enumerate(tables):
                if df.empty:
                    continue
                
                # Obtener página (aproximada)
                page_num = table_index + 1  # Simplificación
                
                # Guardar como CSV
                csv_path = manual_dir / f"table_{table_index+1}.csv"
                df.to_csv(csv_path, index=False)
                
                # Preparar datos
                table_data = {
                    'manual_id': manual_id,
                    'page_number': page_num,
                    'table_index': table_index + 1,
                    'extraction_method': 'tabula',
                    'extraction_accuracy': 0.8,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'headers': ','.join(str(col) for col in df.columns),
                    'csv_path': str(csv_path.relative_to(self.output_dir)),
                    'table_content': df.to_string()[:1000],  # Primeros 1000 caracteres
                    'has_numeric_data': df.select_dtypes(include=['number']).shape[1] > 0,
                    'has_headers': True
                }
                
                tables_extracted.append(table_data)
                
        except Exception as e:
            logger.warning(f"Error extrayendo tablas con tabula: {e}")
        
        # Insertar en base de datos
        if tables_extracted:
            for table_data in tables_extracted:
                try:
                    self.db.conn.execute("""
                        INSERT INTO tables (
                            manual_id, page_number, table_index, extraction_method,
                            extraction_accuracy, row_count, column_count, headers,
                            csv_path, table_content, has_numeric_data, has_headers
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        table_data['manual_id'], table_data['page_number'],
                        table_data['table_index'], table_data['extraction_method'],
                        table_data['extraction_accuracy'], table_data['row_count'],
                        table_data['column_count'], table_data['headers'],
                        table_data['csv_path'], table_data['table_content'],
                        table_data['has_numeric_data'], table_data['has_headers']
                    ))
                except Exception as e:
                    logger.warning(f"Error insertando tabla: {e}")
            
            self.db.conn.commit()
        
        logger.info(f"Extraídas {len(tables_extracted)} tablas")
        return {'total_tables': len(tables_extracted)}

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
        
        # Limpiar registros existentes
        db.conn.execute("DELETE FROM images WHERE manual_id = ?", (manual_id,))
        db.conn.execute("DELETE FROM tables WHERE manual_id = ?", (manual_id,))
        db.conn.commit()
        
        # Extraer imágenes
        logger.info("Extrayendo imágenes...")
        image_extractor = FixedImageExtractor(db, config.PROCESSED_DIR / 'images')
        image_results = image_extractor.extract_and_store(pdf_path, manual_id)
        logger.info(f"  → Extraídas {image_results['total_images']} imágenes")
        
        # Extraer tablas
        logger.info("Extrayendo tablas...")
        table_extractor = FixedTableExtractor(db, config.PROCESSED_DIR / 'tables')
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
        print("Uso: python extract_images_tables_fixed.py <manual_id>")
        sys.exit(1)
    
    manual_id = int(sys.argv[1])
    extract_images_and_tables(manual_id)