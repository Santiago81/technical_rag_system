"""
Extractores adaptados para guardar directamente en SQLite
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import hashlib
from datetime import datetime
import json

from database.sqlite_manager import SQLiteRAGManager, ChunkData
from extractors.pdf_extractor import PDFExtractor
from extractors.text_processor import TextProcessor
from extractors.enhanced_image_extractor import EnhancedImageExtractor
from extractors.table_extractor import TableExtractor
from extractors.document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)

class SQLitePDFExtractor:
    """Extractor de PDF que guarda directamente en SQLite"""
    
    def __init__(self, db_manager: SQLiteRAGManager):
        self.db = db_manager
        # No usar PDFExtractor, implementar extracción directamente
        
    def extract_and_store(self, pdf_path: Path, manual_data: Optional[Dict] = None) -> int:
        """Extraer contenido de PDF y almacenar en SQLite"""
        
        # Preparar datos del manual
        if manual_data is None:
            manual_data = {}
        
        manual_data.update({
            'filename': pdf_path.name,
            'file_path': str(pdf_path),
            'file_size_bytes': pdf_path.stat().st_size,
            'name': manual_data.get('name', pdf_path.stem)
        })
        
        # Insertar manual
        manual_id = self.db.insert_manual(manual_data)
        
        try:
            # Extraer contenido usando PyMuPDF directamente
            import fitz
            content = self._extract_pdf_content(pdf_path)
            
            # Actualizar información del manual
            self.db.conn.execute("""
                UPDATE manuals 
                SET total_pages = ?,
                    language = ?
                WHERE id = ?
            """, (content['page_count'], content.get('language', 'es'), manual_id))
            
            # Guardar bloques de contenido
            blocks_to_insert = []
            for page_data in content['pages']:
                for i, block in enumerate(page_data.get('blocks', [])):
                    blocks_to_insert.append({
                        'manual_id': manual_id,
                        'page_number': page_data['page_number'],
                        'block_index': i,
                        'block_type': block.get('type', 'text'),
                        'content': block['text'],
                        'section': page_data.get('section'),
                        'chapter': page_data.get('chapter'),
                        'confidence_score': block.get('confidence', 1.0),
                        'bounding_box': json.dumps(block.get('bbox')) if block.get('bbox') else None,
                        'style_attributes': block.get('style')
                    })
            
            # Insertar bloques
            if blocks_to_insert:
                block_ids = self.db.insert_content_blocks(blocks_to_insert)
                logger.info(f"Insertados {len(block_ids)} bloques de contenido")
            
            # Actualizar estado
            self.db.update_manual_status(manual_id, 'completed')
            
            return manual_id
            
        except Exception as e:
            logger.error(f"Error extrayendo PDF {pdf_path}: {e}")
            self.db.update_manual_status(manual_id, 'failed', str(e))
            raise
    
    def _extract_pdf_content(self, pdf_path: Path) -> Dict:
        """Extraer contenido del PDF usando PyMuPDF"""
        import fitz
        
        content = {
            'full_text': '',
            'pages': []
        }
        
        with fitz.open(str(pdf_path)) as doc:
            content['page_count'] = len(doc)
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                content['full_text'] += text + '\n\n'
                
                # Extraer bloques de texto
                blocks = []
                for block in page.get_text_blocks():
                    blocks.append({
                        'text': block[4],
                        'bbox': block[:4],
                        'type': 'text'
                    })
                
                content['pages'].append({
                    'page_number': page_num + 1,
                    'content': text,
                    'blocks': blocks
                })
        
        return content


class SQLiteTextProcessor:
    """Procesador de texto que guarda chunks en SQLite"""
    
    def __init__(self, db_manager: SQLiteRAGManager, chunk_size: int = 512, chunk_overlap: int = 50):
        self.db = db_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_and_store(self, manual_id: int, generate_embeddings: bool = False) -> List[int]:
        """Procesar texto de un manual y crear chunks"""
        
        # Obtener bloques de contenido
        cursor = self.db.conn.execute("""
            SELECT id, content, page_number, section, chapter
            FROM content_blocks
            WHERE manual_id = ?
            ORDER BY page_number, block_index
        """, (manual_id,))
        
        blocks = [dict(row) for row in cursor]
        
        if not blocks:
            logger.warning(f"No se encontraron bloques para manual {manual_id}")
            return []
        
        # Log de inicio
        self.db.log_processing(manual_id, 'chunking', 'started')
        
        try:
            # Combinar texto por páginas para mejor contexto
            page_texts = {}
            for block in blocks:
                page_num = block['page_number']
                if page_num not in page_texts:
                    page_texts[page_num] = {
                        'text': [],
                        'block_ids': [],
                        'section': block['section'],
                        'chapter': block['chapter']
                    }
                page_texts[page_num]['text'].append(block['content'])
                page_texts[page_num]['block_ids'].append(block['id'])
            
            # Crear chunks
            all_chunks = []
            chunk_index = 0
            
            for page_num, page_data in sorted(page_texts.items()):
                full_text = ' '.join(page_data['text'])
                
                # Crear chunks directamente
                chunks = self._create_chunks(full_text)
                
                for chunk in chunks:
                    # Preparar datos del chunk
                    chunk_data = ChunkData(
                        manual_id=manual_id,
                        chunk_text=chunk['text'],
                        chunk_text_processed=self._preprocess_text(chunk['text']),
                        chunk_index=chunk_index,
                        chunk_size=len(chunk['text']),
                        overlap_size=chunk.get('overlap', 0),
                        start_page=page_num,
                        end_page=page_num,
                        keywords=self._extract_keywords(chunk['text']),
                        entities=self._extract_entities(chunk['text']),
                        importance_score=self._calculate_importance(chunk['text'], page_num),
                        metadata={
                            'section': page_data['section'],
                            'chapter': page_data['chapter'],
                            'source_blocks': page_data['block_ids'][:5]  # Limitar para no sobrecargar
                        }
                    )
                    
                    # Agregar contexto si está disponible
                    if chunk_index > 0 and all_chunks:
                        chunk_data.context_before = all_chunks[-1].chunk_text[-200:]
                    
                    all_chunks.append(chunk_data)
                    chunk_index += 1
            
            # Agregar contexto posterior
            for i in range(len(all_chunks) - 1):
                all_chunks[i].context_after = all_chunks[i + 1].chunk_text[:200]
            
            # Insertar chunks en batch
            chunk_ids = self.db.insert_chunks_batch(all_chunks)
            
            # Log de éxito
            self.db.log_processing(manual_id, 'chunking', 'completed', 
                                 details={'chunks_created': len(chunk_ids)})
            
            logger.info(f"Creados {len(chunk_ids)} chunks para manual {manual_id}")
            
            # Generar embeddings si se solicita
            if generate_embeddings:
                self._generate_embeddings_for_chunks(manual_id, chunk_ids)
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error procesando texto para manual {manual_id}: {e}")
            self.db.log_processing(manual_id, 'chunking', 'failed', str(e))
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesar texto para búsqueda"""
        # Normalizar espacios
        text = ' '.join(text.split())
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales pero mantener estructura
        import re
        text = re.sub(r'[^\w\s\-\.,;:!?]', ' ', text)
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraer keywords del texto"""
        # Implementación simple - en producción usar RAKE, YAKE, etc.
        import re
        from collections import Counter
        
        # Tokenizar
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filtrar stopwords básicas
        stopwords = {'el', 'la', 'de', 'en', 'y', 'a', 'que', 'es', 'para', 'con', 'un', 'una'}
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Top 10 palabras más frecuentes
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(10)]
        
        return keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extraer entidades nombradas"""
        # Implementación simple - en producción usar spaCy, transformers, etc.
        entities = []
        
        # Buscar patrones comunes
        import re
        
        # Números de parte/modelo
        part_numbers = re.findall(r'\b[A-Z0-9]{4,}\b', text)
        entities.extend([{'text': pn, 'type': 'PART_NUMBER'} for pn in part_numbers[:5]])
        
        # Medidas
        measurements = re.findall(r'\d+\.?\d*\s*(?:mm|cm|m|kg|g|l|ml|°C|bar|psi)', text)
        entities.extend([{'text': m, 'type': 'MEASUREMENT'} for m in measurements[:5]])
        
        return entities
    
    def _calculate_importance(self, text: str, page_num: int) -> float:
        """Calcular score de importancia del chunk"""
        score = 1.0
        
        # Boost para primeras páginas (suelen tener info importante)
        if page_num <= 5:
            score *= 1.2
        
        # Boost si contiene palabras clave importantes
        important_terms = ['advertencia', 'peligro', 'importante', 'atención', 
                          'mantenimiento', 'instalación', 'configuración']
        text_lower = text.lower()
        for term in important_terms:
            if term in text_lower:
                score *= 1.1
        
        # Penalizar chunks muy cortos
        if len(text) < 100:
            score *= 0.8
        
        return min(max(score, 0.1), 10.0)
    
    def _create_chunks(self, text: str) -> List[Dict]:
        """Crear chunks de texto con overlapping"""
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            # Si el texto es más pequeño que el tamaño del chunk, devolver como está
            chunks.append({'text': text, 'overlap': 0})
            return chunks
        
        # Crear chunks con overlap
        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Intentar cortar en un espacio para no dividir palabras
            if end < text_length:
                last_space = text[start:end].rfind(' ')
                if last_space > self.chunk_size * 0.8:  # Si encontramos espacio en el último 20%
                    end = start + last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:  # Solo agregar chunks no vacíos
                overlap = self.chunk_overlap if start > 0 else 0
                chunks.append({'text': chunk_text, 'overlap': overlap})
            
            # Mover el inicio considerando el overlap
            start = end - self.chunk_overlap if end < text_length else end
        
        return chunks
    
    def _generate_embeddings_for_chunks(self, manual_id: int, chunk_ids: List[int]):
        """Generar embeddings para chunks (placeholder)"""
        # Aquí conectarías con tu modelo de embeddings
        # Por ahora solo logueamos
        logger.info(f"Embeddings pendientes para {len(chunk_ids)} chunks del manual {manual_id}")


class SQLiteImageExtractor:
    """Extractor de imágenes que guarda en SQLite"""
    
    def __init__(self, db_manager: SQLiteRAGManager, image_output_dir: Path):
        self.db = db_manager
        self.output_dir = image_output_dir
        # No usar EnhancedImageExtractor, implementar extracción directamente
        
    def extract_and_store(self, pdf_path: Path, manual_id: int) -> Dict:
        """Extraer imágenes y guardar referencias en SQLite"""
        
        # Crear directorio para este manual
        manual_dir = self.output_dir / f"manual_{manual_id}"
        manual_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extraer imágenes usando PyMuPDF directamente
            results = self._extract_images_from_pdf(pdf_path, manual_dir)
            
            # Preparar datos para inserción
            images_to_insert = []
            
            # Procesar imágenes raster
            for img in results.get('raster_images', []):
                images_to_insert.append({
                    'manual_id': manual_id,
                    'page_number': img['page_number'],
                    'image_index': img['image_index'],
                    'image_type': img.get('content_type', 'raster'),
                    'file_path': str(Path(img['image_path']).relative_to(self.output_dir)),
                    'file_format': Path(img['image_path']).suffix[1:],  # sin el punto
                    'file_size': Path(img['image_path']).stat().st_size,
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'color_space': img.get('color_space'),
                    'ocr_text': img.get('ocr_text'),
                    'file_hash': self._calculate_file_hash(img['image_path'])
                })
            
            # Procesar diagramas vectoriales
            for diag in results.get('vector_diagrams', []):
                images_to_insert.append({
                    'manual_id': manual_id,
                    'page_number': diag['page_number'],
                    'image_index': 0,  # Diagramas son únicos por página
                    'image_type': 'technical_diagram',
                    'file_path': str(Path(diag['image_path']).relative_to(self.output_dir)),
                    'file_format': 'png',
                    'file_size': Path(diag['image_path']).stat().st_size,
                    'width': diag.get('width'),
                    'height': diag.get('height'),
                    'is_technical_diagram': True,
                    'ocr_text': diag.get('ocr_text'),
                    'file_hash': self._calculate_file_hash(diag['image_path']),
                    'dpi': diag.get('render_dpi', 200)
                })
            
            # Insertar en batch
            if images_to_insert:
                image_ids = self.db.insert_images_batch(images_to_insert)
                logger.info(f"Insertadas {len(image_ids)} imágenes para manual {manual_id}")
            
            return {
                'total_images': len(images_to_insert),
                'statistics': results.get('statistics', {})
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo imágenes para manual {manual_id}: {e}")
            raise
    
    def _extract_images_from_pdf(self, pdf_path: Path, output_dir: Path) -> Dict:
        """Extraer imágenes del PDF usando PyMuPDF"""
        import fitz
        from PIL import Image
        import io
        
        results = {
            'raster_images': [],
            'vector_diagrams': [],
            'statistics': {
                'total_images': 0,
                'raster_count': 0,
                'vector_count': 0
            }
        }
        
        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc):
                # Extraer imágenes raster
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Obtener imagen
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convertir a PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY o RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix.tobytes("png")
                        
                        # Guardar imagen
                        img_path = output_dir / f"page_{page_num+1}_img_{img_index+1}.png"
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        
                        results['raster_images'].append({
                            'page_number': page_num + 1,
                            'image_index': img_index + 1,
                            'image_path': str(img_path),
                            'width': pix.width,
                            'height': pix.height,
                            'content_type': 'raster'
                        })
                        
                        results['statistics']['raster_count'] += 1
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error extrayendo imagen {img_index} de página {page_num}: {e}")
                
                # TODO: Extraer diagramas vectoriales si es necesario
        
        results['statistics']['total_images'] = results['statistics']['raster_count'] + results['statistics']['vector_count']
        return results
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash MD5 de archivo"""
        import hashlib
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class SQLiteTableExtractor:
    """Extractor de tablas que guarda en SQLite"""
    
    def __init__(self, db_manager: SQLiteRAGManager, table_output_dir: Path):
        self.db = db_manager
        self.output_dir = table_output_dir
        # No usar TableExtractor, implementar extracción directamente
        
    def extract_and_store(self, pdf_path: Path, manual_id: int) -> Dict:
        """Extraer tablas y guardar en SQLite"""
        
        # Crear directorio para este manual
        manual_dir = self.output_dir / f"manual_{manual_id}"
        manual_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extraer tablas usando tabula-py directamente
            tables = self._extract_tables_from_pdf(pdf_path, manual_dir)
            
            # Preparar datos para inserción
            tables_to_insert = []
            
            for table in tables:
                # Leer contenido del CSV para análisis
                csv_path = Path(table['csv_path'])
                if csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # Detectar tipos de datos
                    data_types = {}
                    has_numeric = False
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        data_types[col] = dtype
                        if dtype.startswith(('int', 'float')):
                            has_numeric = True
                    
                    # Crear representación textual
                    table_text = f"Tabla con {len(df)} filas y {len(df.columns)} columnas. "
                    table_text += f"Columnas: {', '.join(df.columns)}. "
                    if len(df) > 0:
                        table_text += f"Muestra: {df.head(3).to_string()}"
                    
                    tables_to_insert.append({
                        'manual_id': manual_id,
                        'page_number': table['page_number'],
                        'table_index': table['table_index'],
                        'extraction_method': table.get('extraction_method', 'camelot'),
                        'extraction_accuracy': table.get('accuracy', 0.8),
                        'csv_path': str(csv_path.relative_to(self.output_dir)),
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'headers': list(df.columns),
                        'data_types': data_types,
                        'table_content': table_text[:1000],  # Limitar tamaño
                        'has_numeric_data': has_numeric,
                        'has_headers': True,
                        'metadata_json': {
                            'extraction_timestamp': datetime.now().isoformat()
                        }
                    })
            
            # Insertar en batch
            if tables_to_insert:
                table_ids = self.db.insert_tables_batch(tables_to_insert)
                logger.info(f"Insertadas {len(table_ids)} tablas para manual {manual_id}")
            
            return {
                'total_tables': len(tables_to_insert),
                'tables': tables_to_insert
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo tablas para manual {manual_id}: {e}")
            raise
    
    def _extract_tables_from_pdf(self, pdf_path: Path, output_dir: Path) -> List[Dict]:
        """Extraer tablas del PDF usando tabula-py"""
        try:
            import tabula
            import pandas as pd
            
            tables = []
            
            # Intentar extraer tablas con tabula
            dfs = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True, 
                                  pandas_options={'header': None})
            
            for idx, df in enumerate(dfs):
                if df.empty:
                    continue
                
                # Guardar como CSV
                csv_path = output_dir / f"table_{idx+1}.csv"
                df.to_csv(csv_path, index=False)
                
                tables.append({
                    'table_index': idx + 1,
                    'page_number': 1,  # tabula no siempre da el número de página exacto
                    'csv_path': str(csv_path),
                    'extraction_method': 'tabula',
                    'accuracy': 0.8
                })
            
            logger.info(f"Extraídas {len(tables)} tablas con tabula-py")
            return tables
            
        except Exception as e:
            logger.warning(f"Error extrayendo tablas con tabula: {e}")
            return []


class SQLiteDocumentAnalyzer:
    """Analizador de documentos que guarda en SQLite"""
    
    def __init__(self, db_manager: SQLiteRAGManager):
        self.db = db_manager
        # No usar DocumentAnalyzer, implementar análisis directamente
        
    def analyze_and_store(self, pdf_path: Path, manual_id: int) -> Dict:
        """Analizar documento y guardar análisis"""
        
        try:
            # Realizar análisis directamente
            analysis = self._analyze_document(pdf_path)
            
            # Preparar datos para SQLite
            db_analysis = {
                'manual_id': manual_id,
                'document_type': analysis['document_type'],
                'text_pages': analysis['content_distribution']['text_pages'],
                'image_pages': analysis['content_distribution']['image_pages'],
                'mixed_pages': analysis['content_distribution']['mixed_pages'],
                'empty_pages': analysis['content_distribution']['empty_pages'],
                'avg_text_per_page': analysis['page_analysis']['avg_text_per_page'],
                'image_frequency': analysis['page_analysis']['image_frequency'],
                'table_frequency': analysis['page_analysis']['table_frequency'],
                'recommended_chunk_size': analysis['extraction_strategy']['chunk_size'],
                'recommended_overlap': analysis['extraction_strategy']['overlap'],
                'use_ocr': 1 if analysis['extraction_strategy']['use_ocr'] else 0
            }
            
            # Guardar análisis
            analysis_id = self.db.save_document_analysis(db_analysis)
            
            logger.info(f"Análisis guardado para manual {manual_id}: tipo {analysis['document_type']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando manual {manual_id}: {e}")
            raise
    
    def _analyze_document(self, pdf_path: Path) -> Dict:
        """Analizar características del documento PDF"""
        import fitz
        
        analysis = {
            'document_type': 'technical',
            'content_distribution': {
                'text_pages': 0,
                'image_pages': 0,
                'mixed_pages': 0,
                'empty_pages': 0
            },
            'page_analysis': {
                'total_pages': 0,
                'avg_text_per_page': 0,
                'image_frequency': 0,
                'table_frequency': 0
            },
            'extraction_strategy': {
                'chunk_size': 512,
                'overlap': 50,
                'use_ocr': False
            }
        }
        
        try:
            with fitz.open(str(pdf_path)) as doc:
                analysis['page_analysis']['total_pages'] = len(doc)
                
                text_lengths = []
                image_counts = []
                
                for page in doc:
                    # Analizar texto
                    text = page.get_text()
                    text_length = len(text.strip())
                    text_lengths.append(text_length)
                    
                    # Analizar imágenes
                    images = page.get_images()
                    image_counts.append(len(images))
                    
                    # Clasificar página
                    if text_length < 100 and len(images) == 0:
                        analysis['content_distribution']['empty_pages'] += 1
                    elif text_length < 100 and len(images) > 0:
                        analysis['content_distribution']['image_pages'] += 1
                    elif text_length >= 100 and len(images) > 0:
                        analysis['content_distribution']['mixed_pages'] += 1
                    else:
                        analysis['content_distribution']['text_pages'] += 1
                
                # Calcular promedios
                if text_lengths:
                    analysis['page_analysis']['avg_text_per_page'] = sum(text_lengths) / len(text_lengths)
                
                if image_counts:
                    total_images = sum(image_counts)
                    pages_with_images = sum(1 for c in image_counts if c > 0)
                    analysis['page_analysis']['image_frequency'] = pages_with_images / len(doc)
                
                # Determinar tipo de documento
                if analysis['page_analysis']['image_frequency'] > 0.7:
                    analysis['document_type'] = 'technical_diagrams'
                elif analysis['page_analysis']['avg_text_per_page'] < 200:
                    analysis['document_type'] = 'scanned'
                    analysis['extraction_strategy']['use_ocr'] = True
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error en análisis de documento: {e}")
            return analysis