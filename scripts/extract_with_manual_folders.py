#!/usr/bin/env python3
"""
Script mejorado que organiza el contenido extraído en carpetas por nombre de manual
Estructura: data/processed/[nombre_manual]/[images|diagrams|tables]/
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
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_folder_name(name: str) -> str:
    """Sanitizar nombre para uso como carpeta"""
    # Remover caracteres no válidos para nombres de carpeta
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remover espacios múltiples
    name = re.sub(r'\s+', '_', name)
    # Remover puntos al final
    name = name.rstrip('.')
    return name

class OrganizedContentExtractor:
    """Extractor que organiza contenido por carpetas de manual"""
    
    def __init__(self, db, base_output_dir: Path):
        self.db = db
        self.base_output_dir = base_output_dir
        
    def extract_and_store(self, pdf_path: Path, manual_id: int, manual_name: str):
        """Extraer y almacenar TODO el contenido visual organizado por manual"""
        
        # Crear estructura de directorios para este manual
        manual_folder_name = sanitize_folder_name(manual_name)
        manual_dir = self.base_output_dir / manual_folder_name
        
        images_dir = manual_dir / "images"
        diagrams_dir = manual_dir / "diagrams"
        tables_dir = manual_dir / "tables"
        
        # Crear todos los directorios
        for dir_path in [images_dir, diagrams_dir, tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creada estructura de directorios en: {manual_dir}")
        
        all_visuals = []
        
        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                logger.info(f"Procesando página {page_num + 1}")
                
                # 1. Extraer imágenes raster embebidas
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.colorspace is None:
                            continue
                            
                        # Convertir a RGB si es necesario
                        if pix.n - pix.alpha >= 4:  # CMYK u otro
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Guardar imagen en carpeta images
                        img_filename = f"page_{page_num+1}_img_{img_index+1}.png"
                        img_path = images_dir / img_filename
                        pix.save(str(img_path))
                        
                        # Calcular hash
                        with open(img_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        # Ruta relativa desde base_output_dir
                        relative_path = img_path.relative_to(self.base_output_dir)
                        
                        visual_data = {
                            'manual_id': manual_id,
                            'page_number': page_num + 1,
                            'image_index': img_index + 1,
                            'image_type': 'raster',
                            'file_path': str(relative_path),
                            'file_format': 'png',
                            'file_size': img_path.stat().st_size,
                            'file_hash': file_hash,
                            'width': pix.width,
                            'height': pix.height,
                            'color_space': 'RGB'
                        }
                        
                        all_visuals.append(visual_data)
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error procesando imagen raster {img_index+1} en página {page_num+1}: {e}")
                
                # 2. Renderizar la página completa como diagrama si tiene contenido visual significativo
                try:
                    text = page.get_text()
                    has_drawings = len(page.get_drawings()) > 0
                    has_images = len(image_list) > 0
                    text_ratio = len(text.strip()) / (page.rect.width * page.rect.height) if page.rect.width * page.rect.height > 0 else 0
                    
                    # Criterios para renderizar como diagrama
                    should_render = has_drawings or (has_images and text_ratio < 0.1) or text_ratio < 0.05
                    
                    if should_render:
                        # Renderizar página completa a alta resolución
                        zoom = 2.0  # 200% zoom para mejor calidad
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        
                        # Guardar diagrama en carpeta diagrams
                        diagram_filename = f"page_{page_num+1}_diagram.png"
                        diagram_path = diagrams_dir / diagram_filename
                        pix.save(str(diagram_path))
                        
                        # Calcular hash
                        with open(diagram_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        # Ruta relativa desde base_output_dir
                        relative_path = diagram_path.relative_to(self.base_output_dir)
                        
                        visual_data = {
                            'manual_id': manual_id,
                            'page_number': page_num + 1,
                            'image_index': 999,  # Índice especial para diagramas
                            'image_type': 'technical_diagram',
                            'file_path': str(relative_path),
                            'file_format': 'png',
                            'file_size': diagram_path.stat().st_size,
                            'file_hash': file_hash,
                            'width': pix.width,
                            'height': pix.height,
                            'color_space': 'RGB'
                        }
                        
                        all_visuals.append(visual_data)
                        logger.info(f"  → Renderizado diagrama de página {page_num + 1}")
                        pix = None
                        
                except Exception as e:
                    logger.warning(f"Error renderizando diagrama de página {page_num+1}: {e}")
        
        # Insertar en base de datos
        inserted_count = 0
        if all_visuals:
            for visual_data in all_visuals:
                try:
                    # Verificar si ya existe por hash
                    existing = self.db.conn.execute(
                        "SELECT id FROM images WHERE file_hash = ?", 
                        (visual_data['file_hash'],)
                    ).fetchone()
                    
                    if not existing:
                        self.db.conn.execute("""
                            INSERT INTO images (
                                manual_id, page_number, image_index, image_type,
                                file_path, file_format, file_size, file_hash,
                                width, height, color_space
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            visual_data['manual_id'], visual_data['page_number'], 
                            visual_data['image_index'], visual_data['image_type'],
                            visual_data['file_path'], visual_data['file_format'],
                            visual_data['file_size'], visual_data['file_hash'],
                            visual_data['width'], visual_data['height'], visual_data['color_space']
                        ))
                        inserted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error insertando visual: {e}")
            
            self.db.conn.commit()
        
        # Contar por tipo
        by_type = {}
        for v in all_visuals:
            by_type[v['image_type']] = by_type.get(v['image_type'], 0) + 1
        
        logger.info(f"Contenido visual extraído - Imágenes: {by_type.get('raster', 0)}, Diagramas: {by_type.get('technical_diagram', 0)}")
        
        return {
            'total_images': len(all_visuals),
            'inserted': inserted_count,
            'by_type': by_type,
            'tables_dir': tables_dir  # Retornar directorio de tablas para el siguiente paso
        }

class OrganizedTableExtractor:
    """Extractor de tablas que guarda en carpeta del manual"""
    
    def __init__(self, db):
        self.db = db
        
    def extract_and_store(self, pdf_path: Path, manual_id: int, tables_dir: Path):
        """Extraer y almacenar tablas en el directorio especificado"""
        
        tables_extracted = []
        
        try:
            # Intentar con tabula
            tables = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True)
            
            for table_index, df in enumerate(tables):
                if df.empty:
                    continue
                
                # Guardar como CSV en carpeta tables del manual
                csv_filename = f"table_{table_index+1}.csv"
                csv_path = tables_dir / csv_filename
                df.to_csv(csv_path, index=False)
                
                # Ruta relativa desde data/processed
                base_processed_dir = tables_dir.parent.parent  # data/processed
                relative_path = csv_path.relative_to(base_processed_dir)
                
                # Preparar datos
                table_data = {
                    'manual_id': manual_id,
                    'page_number': table_index + 1,  # Aproximación
                    'table_index': table_index + 1,
                    'extraction_method': 'tabula',
                    'extraction_accuracy': 0.8,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'headers': ','.join(str(col) for col in df.columns),
                    'csv_path': str(relative_path),
                    'table_content': df.to_string()[:1000],
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
        
        logger.info(f"Tablas extraídas: {len(tables_extracted)}")
        return {'total_tables': len(tables_extracted)}


def extract_all_content_organized(manual_id: int):
    """Extraer TODO el contenido con la nueva estructura de carpetas"""
    
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
            pdf_path = config.BASE_DIR / manual['file_path']
            if not pdf_path.exists():
                logger.error(f"PDF no encontrado: {manual['file_path']}")
                return
        
        logger.info(f"\nProcesando manual: {manual['name']}")
        logger.info("="*60)
        
        # Limpiar registros existentes de este manual
        logger.info("Limpiando registros anteriores...")
        db.conn.execute("DELETE FROM images WHERE manual_id = ?", (manual_id,))
        db.conn.execute("DELETE FROM tables WHERE manual_id = ?", (manual_id,))
        db.conn.commit()
        
        # Extraer contenido visual con nueva estructura
        logger.info("\n1. Extrayendo contenido visual (imágenes + diagramas)...")
        visual_extractor = OrganizedContentExtractor(db, config.PROCESSED_DIR)
        visual_results = visual_extractor.extract_and_store(pdf_path, manual_id, manual['name'])
        
        # Extraer tablas en la carpeta correcta
        logger.info("\n2. Extrayendo tablas...")
        table_extractor = OrganizedTableExtractor(db)
        table_results = table_extractor.extract_and_store(
            pdf_path, 
            manual_id, 
            visual_results['tables_dir']
        )
        
        # Actualizar estadísticas
        total_visuals = db.conn.execute(
            "SELECT COUNT(*) FROM images WHERE manual_id = ?", 
            (manual_id,)
        ).fetchone()[0]
        
        total_tables = db.conn.execute(
            "SELECT COUNT(*) FROM tables WHERE manual_id = ?", 
            (manual_id,)
        ).fetchone()[0]
        
        db.conn.execute("""
            UPDATE manuals 
            SET total_images = ?, total_tables = ?
            WHERE id = ?
        """, (total_visuals, total_tables, manual_id))
        db.conn.commit()
        
        # Resumen final
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE EXTRACCIÓN")
        logger.info("="*60)
        logger.info(f"Manual: {manual['name']}")
        logger.info(f"Contenido visual total: {total_visuals}")
        if 'by_type' in visual_results:
            for tipo, count in visual_results['by_type'].items():
                logger.info(f"  - {tipo}: {count}")
        logger.info(f"Tablas extraídas: {total_tables}")
        
        manual_folder = sanitize_folder_name(manual['name'])
        logger.info(f"\nArchivos guardados en:")
        logger.info(f"  data/processed/{manual_folder}/")
        logger.info(f"    ├── images/    ({visual_results['by_type'].get('raster', 0)} archivos)")
        logger.info(f"    ├── diagrams/  ({visual_results['by_type'].get('technical_diagram', 0)} archivos)")
        logger.info(f"    └── tables/    ({total_tables} archivos)")
        
    finally:
        db.close()


def migrate_existing_content():
    """Migrar contenido existente a la nueva estructura"""
    config = Config()
    db_path = str(config.DATA_DIR / 'sqlite' / 'manuals.db')
    db = SQLiteRAGManager(db_path)
    
    try:
        # Obtener todos los manuales
        manuals = db.conn.execute("SELECT id, name FROM manuals").fetchall()
        
        for manual_id, manual_name in manuals:
            logger.info(f"\nMigrando contenido del manual: {manual_name}")
            
            # Crear nueva estructura
            manual_folder_name = sanitize_folder_name(manual_name)
            new_manual_dir = config.PROCESSED_DIR / manual_folder_name
            
            new_images_dir = new_manual_dir / "images"
            new_diagrams_dir = new_manual_dir / "diagrams"
            new_tables_dir = new_manual_dir / "tables"
            
            # Crear directorios
            for dir_path in [new_images_dir, new_diagrams_dir, new_tables_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Directorios antiguos
            old_images_dir = config.PROCESSED_DIR / "images" / str(manual_id)
            old_tables_dir = config.PROCESSED_DIR / "tables" / str(manual_id)
            
            # Migrar imágenes y diagramas
            if old_images_dir.exists():
                # Migrar imágenes raster
                if (old_images_dir / "raster").exists():
                    for img_file in (old_images_dir / "raster").glob("*.png"):
                        img_file.rename(new_images_dir / img_file.name)
                
                # Migrar diagramas
                if (old_images_dir / "diagrams").exists():
                    for diag_file in (old_images_dir / "diagrams").glob("*.png"):
                        diag_file.rename(new_diagrams_dir / diag_file.name)
                
                # Migrar imágenes sueltas (estructura anterior)
                for img_file in old_images_dir.glob("*.png"):
                    if img_file.is_file():
                        if "diagram" in img_file.name:
                            img_file.rename(new_diagrams_dir / img_file.name)
                        else:
                            img_file.rename(new_images_dir / img_file.name)
            
            # Migrar tablas
            if old_tables_dir.exists():
                for table_file in old_tables_dir.glob("*.csv"):
                    table_file.rename(new_tables_dir / table_file.name)
            
            # Actualizar rutas en la base de datos
            # Actualizar imágenes
            images = db.conn.execute(
                "SELECT id, file_path, image_type FROM images WHERE manual_id = ?", 
                (manual_id,)
            ).fetchall()
            
            for img_id, old_path, img_type in images:
                filename = Path(old_path).name
                if img_type == 'technical_diagram':
                    new_path = f"{manual_folder_name}/diagrams/{filename}"
                else:
                    new_path = f"{manual_folder_name}/images/{filename}"
                
                db.conn.execute(
                    "UPDATE images SET file_path = ? WHERE id = ?",
                    (new_path, img_id)
                )
            
            # Actualizar tablas
            tables = db.conn.execute(
                "SELECT id, csv_path FROM tables WHERE manual_id = ?", 
                (manual_id,)
            ).fetchall()
            
            for table_id, old_path in tables:
                filename = Path(old_path).name
                new_path = f"{manual_folder_name}/tables/{filename}"
                
                db.conn.execute(
                    "UPDATE tables SET csv_path = ? WHERE id = ?",
                    (new_path, table_id)
                )
            
            db.conn.commit()
            logger.info(f"  ✓ Migración completada para {manual_name}")
            
            # Limpiar directorios antiguos si están vacíos
            for old_dir in [old_images_dir, old_tables_dir]:
                if old_dir.exists() and not any(old_dir.iterdir()):
                    old_dir.rmdir()
        
        logger.info("\n✅ Migración completada exitosamente")
        
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python extract_with_manual_folders.py <manual_id>  # Procesar un manual")
        print("  python extract_with_manual_folders.py --migrate    # Migrar estructura existente")
        sys.exit(1)
    
    if sys.argv[1] == "--migrate":
        migrate_existing_content()
    else:
        manual_id = int(sys.argv[1])
        extract_all_content_organized(manual_id)