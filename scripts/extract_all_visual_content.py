#!/usr/bin/env python3
"""
Script completo para extraer TODO el contenido visual: imágenes raster Y diagramas vectoriales
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

class ComprehensiveImageExtractor:
    """Extractor completo de contenido visual: imágenes Y diagramas"""
    
    def __init__(self, db, output_dir: Path):
        self.db = db
        self.output_dir = output_dir
        
    def extract_and_store(self, pdf_path: Path, manual_id: int):
        """Extraer y almacenar TODO el contenido visual"""
        
        # Crear directorio de salida
        manual_dir = self.output_dir / str(manual_id)
        manual_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios para organizar mejor
        raster_dir = manual_dir / "raster"
        diagrams_dir = manual_dir / "diagrams"
        raster_dir.mkdir(exist_ok=True)
        diagrams_dir.mkdir(exist_ok=True)
        
        all_visuals = []
        
        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                logger.info(f"Procesando página {page_num + 1}")
                
                # 1. Extraer imágenes raster embebidas
                image_list = page.get_images()
                raster_count = 0
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.colorspace is None:
                            continue
                            
                        # Convertir a RGB si es necesario
                        if pix.n - pix.alpha >= 4:  # CMYK u otro
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Guardar imagen raster
                        img_path = raster_dir / f"page_{page_num+1}_img_{img_index+1}.png"
                        pix.save(str(img_path))
                        
                        # Calcular hash
                        with open(img_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        visual_data = {
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
                        
                        all_visuals.append(visual_data)
                        raster_count += 1
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error procesando imagen raster {img_index+1} en página {page_num+1}: {e}")
                
                # 2. Renderizar la página completa como diagrama si tiene contenido visual significativo
                try:
                    # Verificar si la página tiene contenido visual significativo
                    text = page.get_text()
                    has_drawings = len(page.get_drawings()) > 0
                    has_images = len(image_list) > 0
                    text_ratio = len(text.strip()) / (page.rect.width * page.rect.height) if page.rect.width * page.rect.height > 0 else 0
                    
                    # Criterios para renderizar como diagrama:
                    # - Tiene dibujos vectoriales
                    # - Tiene imágenes pero poco texto (posible diagrama técnico)
                    # - Ratio texto/área bajo (página principalmente visual)
                    should_render = has_drawings or (has_images and text_ratio < 0.1) or text_ratio < 0.05
                    
                    if should_render:
                        # Renderizar página completa a alta resolución
                        zoom = 2.0  # 200% zoom para mejor calidad
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        
                        # Guardar diagrama
                        diagram_path = diagrams_dir / f"page_{page_num+1}_diagram.png"
                        pix.save(str(diagram_path))
                        
                        # Calcular hash
                        with open(diagram_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        visual_data = {
                            'manual_id': manual_id,
                            'page_number': page_num + 1,
                            'image_index': 999,  # Índice especial para diagramas de página completa
                            'image_type': 'technical_diagram',
                            'file_path': str(diagram_path.relative_to(self.output_dir)),
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
                
                # 3. Extraer regiones específicas con dibujos vectoriales
                try:
                    drawings = page.get_drawings()
                    if drawings:
                        # Agrupar dibujos cercanos en regiones
                        drawing_rects = []
                        for drawing in drawings:
                            for item in drawing.get("items", []):
                                if "rect" in item:
                                    drawing_rects.append(fitz.Rect(item["rect"]))
                        
                        # Combinar rectángulos cercanos
                        if drawing_rects:
                            combined_rects = self._combine_nearby_rects(drawing_rects)
                            
                            for idx, rect in enumerate(combined_rects):
                                # Expandir un poco el rectángulo para capturar contexto
                                rect.x0 = max(0, rect.x0 - 10)
                                rect.y0 = max(0, rect.y0 - 10)
                                rect.x1 = min(page.rect.width, rect.x1 + 10)
                                rect.y1 = min(page.rect.height, rect.y1 + 10)
                                
                                # Renderizar región
                                zoom = 2.0
                                mat = fitz.Matrix(zoom, zoom)
                                pix = page.get_pixmap(clip=rect, matrix=mat, alpha=False)
                                
                                # Guardar solo si tiene tamaño significativo
                                if pix.width > 50 and pix.height > 50:
                                    region_path = diagrams_dir / f"page_{page_num+1}_region_{idx+1}.png"
                                    pix.save(str(region_path))
                                    
                                    with open(region_path, 'rb') as f:
                                        file_hash = hashlib.md5(f.read()).hexdigest()
                                    
                                    visual_data = {
                                        'manual_id': manual_id,
                                        'page_number': page_num + 1,
                                        'image_index': 900 + idx,  # Índices 900+ para regiones
                                        'image_type': 'vector_region',
                                        'file_path': str(region_path.relative_to(self.output_dir)),
                                        'file_format': 'png',
                                        'file_size': region_path.stat().st_size,
                                        'file_hash': file_hash,
                                        'width': pix.width,
                                        'height': pix.height,
                                        'color_space': 'RGB'
                                    }
                                    
                                    all_visuals.append(visual_data)
                                    pix = None
                                
                except Exception as e:
                    logger.warning(f"Error extrayendo regiones vectoriales de página {page_num+1}: {e}")
        
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
        
        logger.info(f"Total de contenido visual extraído: {len(all_visuals)}")
        logger.info(f"Nuevos registros insertados: {inserted_count}")
        
        # Contar por tipo
        by_type = {}
        for v in all_visuals:
            by_type[v['image_type']] = by_type.get(v['image_type'], 0) + 1
        
        logger.info(f"Desglose por tipo: {by_type}")
        
        return {
            'total_images': len(all_visuals),
            'inserted': inserted_count,
            'by_type': by_type
        }
    
    def _combine_nearby_rects(self, rects, threshold=50):
        """Combinar rectángulos cercanos en regiones más grandes"""
        if not rects:
            return []
        
        combined = []
        used = set()
        
        for i, rect1 in enumerate(rects):
            if i in used:
                continue
                
            # Comenzar con este rectángulo
            combined_rect = fitz.Rect(rect1)
            used.add(i)
            
            # Buscar rectángulos cercanos
            changed = True
            while changed:
                changed = False
                for j, rect2 in enumerate(rects):
                    if j in used:
                        continue
                    
                    # Verificar si están cerca
                    if (abs(combined_rect.x0 - rect2.x1) < threshold or
                        abs(combined_rect.x1 - rect2.x0) < threshold or
                        abs(combined_rect.y0 - rect2.y1) < threshold or
                        abs(combined_rect.y1 - rect2.y0) < threshold):
                        
                        # Expandir el rectángulo combinado
                        combined_rect.x0 = min(combined_rect.x0, rect2.x0)
                        combined_rect.y0 = min(combined_rect.y0, rect2.y0)
                        combined_rect.x1 = max(combined_rect.x1, rect2.x1)
                        combined_rect.y1 = max(combined_rect.y1, rect2.y1)
                        used.add(j)
                        changed = True
            
            combined.append(combined_rect)
        
        return combined


def extract_all_content(manual_id: int):
    """Extraer TODO el contenido visual y tablas de un manual"""
    
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
        
        # Extraer TODO el contenido visual
        logger.info("\n1. Extrayendo contenido visual (imágenes + diagramas)...")
        image_extractor = ComprehensiveImageExtractor(db, config.PROCESSED_DIR / 'images')
        image_results = image_extractor.extract_and_store(pdf_path, manual_id)
        
        # Extraer tablas (usando el extractor anterior)
        logger.info("\n2. Extrayendo tablas...")
        from extract_images_tables_fixed import FixedTableExtractor
        table_extractor = FixedTableExtractor(db, config.PROCESSED_DIR / 'tables')
        table_results = table_extractor.extract_and_store(pdf_path, manual_id)
        
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
        if 'by_type' in image_results:
            for tipo, count in image_results['by_type'].items():
                logger.info(f"  - {tipo}: {count}")
        logger.info(f"Tablas extraídas: {total_tables}")
        logger.info(f"\nArchivos guardados en:")
        logger.info(f"  - Imágenes: data/processed/images/{manual_id}/")
        logger.info(f"  - Tablas: data/processed/tables/{manual_id}/")
        
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extract_all_visual_content.py <manual_id>")
        sys.exit(1)
    
    manual_id = int(sys.argv[1])
    extract_all_content(manual_id)