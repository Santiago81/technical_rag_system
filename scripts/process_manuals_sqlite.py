"""
Script principal de procesamiento de manuales con SQLite como backend
"""
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import json
import re

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from database.sqlite_manager import SQLiteRAGManager
from extractors.sqlite_extractors import (
    SQLitePDFExtractor,
    SQLiteTextProcessor,
    SQLiteImageExtractor,
    SQLiteTableExtractor,
    SQLiteDocumentAnalyzer
)
from vectorstore.vector_manager import VectorManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extrae metadatos del nombre del archivo PDF.
    
    Ejemplos de patrones reconocidos:
    - AX5000_SystemManual_V2_5.pdf -> manufacturer: "Beckhoff", model: "AX5000"
    - Lexium32M_UserGuide072022.pdf -> manufacturer: "Schneider Electric", model: "Lexium32M"
    - 072152-101_CC103_Hardware_en.pdf -> manufacturer: "Unknown", model: "CC103"
    """
    metadata = {
        'manufacturer': None,
        'model': None,
        'version': None
    }
    
    # Mapeo de prefijos conocidos a fabricantes
    manufacturer_patterns = {
        'AX': 'Beckhoff',
        'EL': 'Beckhoff',
        'EK': 'Beckhoff',
        'CX': 'Beckhoff',
        'Lexium': 'Schneider Electric',
        'Altivar': 'Schneider Electric',
        'Modicon': 'Schneider Electric',
        'CC': 'Unknown',  # Para CC103
    }
    
    # Remover la extensión .pdf
    name_without_ext = filename.replace('.pdf', '').replace('.PDF', '')
    
    # Buscar patrones de modelo
    # Patrón 1: Modelo al inicio seguido de underscore (ej: AX5000_SystemManual)
    model_match = re.match(r'^([A-Z]+[0-9]+[A-Z]*)_', name_without_ext)
    if model_match:
        metadata['model'] = model_match.group(1)
    else:
        # Patrón 2: Modelo con letras y números juntos (ej: Lexium32M)
        model_match = re.match(r'^([A-Za-z]+[0-9]+[A-Za-z]*)', name_without_ext)
        if model_match:
            metadata['model'] = model_match.group(1)
        else:
            # Patrón 3: Buscar después de underscore (ej: 072152-101_CC103_Hardware)
            model_match = re.search(r'_([A-Z]+[0-9]+)_', name_without_ext)
            if model_match:
                metadata['model'] = model_match.group(1)
    
    # Determinar fabricante basado en el modelo encontrado
    if metadata['model']:
        for prefix, manufacturer in manufacturer_patterns.items():
            if metadata['model'].startswith(prefix):
                metadata['manufacturer'] = manufacturer
                break
        
        # Si no se encontró fabricante, usar "Unknown"
        if not metadata['manufacturer']:
            metadata['manufacturer'] = 'Unknown'
    
    # Buscar versión (ej: V2_5, v1.0, etc.)
    version_match = re.search(r'[Vv](\d+[._]\d+)', name_without_ext)
    if version_match:
        metadata['version'] = version_match.group(1).replace('_', '.')
    
    return metadata

class SQLiteManualProcessor:
    """Procesador principal de manuales con backend SQLite"""
    
    def __init__(self, config: Config, db_path: Optional[str] = None):
        self.config = config
        
        # Inicializar base de datos
        if db_path is None:
            db_path = str(config.DATA_DIR / 'sqlite' / 'manuals.db')
        
        self.db = SQLiteRAGManager(db_path, str(Path(__file__).parent.parent / 'database' / 'schema_legacy.sql'))
        
        # Inicializar extractores (sin paths específicos, se configuran por manual)
        self.pdf_extractor = SQLitePDFExtractor(self.db)
        self.text_processor = SQLiteTextProcessor(self.db, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.document_analyzer = SQLiteDocumentAnalyzer(self.db)
        
        # Vector manager para embeddings
        self.vector_manager = None  # Se inicializa si se requiere
        
    def process_manual(self, pdf_path: Path, options: dict) -> dict:
        """Procesar un manual completo"""
        
        start_time = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesando: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        results = {
            'pdf_path': str(pdf_path),
            'pdf_name': pdf_path.name,
            'start_time': start_time.isoformat(),
            'steps': {}
        }
        
        try:
            # 1. Extracción de contenido PDF (crear manual primero)
            logger.info("\n[1/6] Extrayendo contenido del PDF...")
            
            # Extraer metadatos del nombre del archivo si no se proporcionaron
            extracted_metadata = extract_metadata_from_filename(pdf_path.name)
            
            manual_data = {
                'name': pdf_path.stem,
                'manufacturer': options.get('manufacturer') or extracted_metadata.get('manufacturer'),
                'model': options.get('model') or extracted_metadata.get('model'),
                'document_type': options.get('document_type', 'technical'),
                'version': extracted_metadata.get('version')  # Añadir versión si se encontró
            }
            
            # Log de metadatos extraídos
            if extracted_metadata.get('manufacturer') or extracted_metadata.get('model'):
                logger.info(f"  → Metadatos extraídos: Fabricante={manual_data['manufacturer']}, Modelo={manual_data['model']}")
            
            manual_id = self.pdf_extractor.extract_and_store(pdf_path, manual_data)
            results['manual_id'] = manual_id
            results['steps']['pdf_extraction'] = {'status': 'completed', 'manual_id': manual_id}
            logger.info(f"  → Manual registrado con ID: {manual_id}")
            
            # 2. Análisis del documento
            if not options.get('skip_analysis', False):
                logger.info("\n[2/6] Analizando documento...")
                analysis = self.document_analyzer.analyze_and_store(pdf_path, manual_id)
                results['steps']['analysis'] = {
                    'status': 'completed',
                    'document_type': analysis['document_type']
                }
                logger.info(f"  → Tipo detectado: {analysis['document_type']}")
            
            # 3. Crear chunks de texto
            if not options.get('skip_chunks', False):
                logger.info("\n[3/6] Creando chunks de texto...")
                chunk_ids = self.text_processor.process_and_store(manual_id)
                results['steps']['chunking'] = {
                    'status': 'completed',
                    'chunks_created': len(chunk_ids)
                }
                logger.info(f"  → Creados {len(chunk_ids)} chunks")
            
            # 4. Extraer imágenes y diagramas
            if not options.get('skip_images', False):
                logger.info("\n[4/6] Extrayendo imágenes y diagramas...")
                
                # Obtener nombre del manual para crear estructura de carpetas
                manual_info = self.db.get_manual(manual_id)
                manual_name = manual_info['name']
                
                # Crear estructura de carpetas para este manual
                manual_folder_name = sanitize_folder_name(manual_name)
                manual_dir = self.config.PROCESSED_DIR / manual_folder_name
                
                # Importar y usar el extractor organizado
                sys.path.append(str(Path(__file__).parent))
                from extract_with_manual_folders import OrganizedContentExtractor
                organized_image_extractor = OrganizedContentExtractor(self.db, self.config.PROCESSED_DIR)
                image_results = organized_image_extractor.extract_and_store(pdf_path, manual_id, manual_name)
                
                results['steps']['images'] = {
                    'status': 'completed',
                    'total_images': image_results['total_images'],
                    'by_type': image_results.get('by_type', {})
                }
                logger.info(f"  → Extraídas {image_results['total_images']} imágenes")
                if image_results.get('by_type'):
                    for img_type, count in image_results['by_type'].items():
                        logger.info(f"    - {img_type}: {count}")
            
            # 5. Extraer tablas
            if not options.get('skip_tables', False):
                logger.info("\n[5/6] Extrayendo tablas...")
                
                # Obtener info del manual si no la tenemos
                if 'manual_name' not in locals():
                    manual_info = self.db.get_manual(manual_id)
                    manual_name = manual_info['name']
                    manual_folder_name = sanitize_folder_name(manual_name)
                
                # Directorio de tablas para este manual
                tables_dir = self.config.PROCESSED_DIR / manual_folder_name / 'tables'
                tables_dir.mkdir(parents=True, exist_ok=True)
                
                # Importar y usar el extractor organizado
                from extract_with_manual_folders import OrganizedTableExtractor
                organized_table_extractor = OrganizedTableExtractor(self.db)
                table_results = organized_table_extractor.extract_and_store(pdf_path, manual_id, tables_dir)
                
                results['steps']['tables'] = {
                    'status': 'completed',
                    'total_tables': table_results['total_tables']
                }
                logger.info(f"  → Extraídas {table_results['total_tables']} tablas")
            
            # 6. Generar embeddings y actualizar vector store
            if options.get('generate_embeddings', False):
                logger.info("\n[6/6] Generando embeddings...")
                embedding_results = self._generate_embeddings(manual_id)
                results['steps']['embeddings'] = {
                    'status': 'completed',
                    'chunks_embedded': embedding_results['chunks_embedded']
                }
                logger.info(f"  → Generados {embedding_results['chunks_embedded']} embeddings")
            
            # Actualizar estado final
            self.db.update_manual_status(manual_id, 'completed')
            
            # Calcular tiempo total
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['status'] = 'success'
            
            # Mostrar resumen
            self._show_processing_summary(results)
            
            # Mostrar estructura de archivos creada
            self._show_final_structure(manual_id)
            
            # Guardar log de procesamiento
            self._save_processing_log(manual_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"\nError procesando manual: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
            # Actualizar estado si tenemos manual_id
            if 'manual_id' in results:
                self.db.update_manual_status(results['manual_id'], 'failed', str(e))
            
            return results
    
    def _generate_embeddings(self, manual_id: int) -> dict:
        """Generar embeddings para chunks"""
        
        # Inicializar vector manager si no existe
        if self.vector_manager is None:
            self.vector_manager = VectorManager(self.config)
        
        # Obtener chunks sin embeddings
        cursor = self.db.conn.execute("""
            SELECT id, chunk_text, chunk_text_processed, start_page, end_page
            FROM content_chunks
            WHERE manual_id = ? AND embedding IS NULL
        """, (manual_id,))
        
        chunks = [dict(row) for row in cursor]
        
        if not chunks:
            return {'chunks_embedded': 0}
        
        # Generar embeddings en lotes
        batch_size = 32
        total_embedded = 0
        
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
                    self.config.EMBEDDING_MODEL,
                    len(embedding),
                    chunk['id']
                ))
            
            self.db.conn.commit()
            total_embedded += len(batch)
            
            logger.debug(f"  Procesados {total_embedded}/{len(chunks)} chunks")
        
        # También añadir al vector store para compatibilidad
        self._add_to_vector_store(manual_id)
        
        return {'chunks_embedded': total_embedded}
    
    def _add_to_vector_store(self, manual_id: int):
        """Añadir chunks al vector store para mantener compatibilidad"""
        
        # Esta función mantiene la compatibilidad con el sistema existente
        # En el futuro, podrías eliminarla y usar solo SQLite
        
        cursor = self.db.conn.execute("""
            SELECT 
                c.id,
                c.chunk_text,
                c.embedding,
                c.start_page,
                c.end_page,
                m.name as manual_name
            FROM content_chunks c
            JOIN manuals m ON c.manual_id = m.id
            WHERE c.manual_id = ? AND c.embedding IS NOT NULL
        """, (manual_id,))
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for row in cursor:
            documents.append(row['chunk_text'])
            # Reconstruir embedding desde bytes
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            embeddings.append(embedding)
            
            metadatas.append({
                'manual_name': row['manual_name'],
                'chunk_id': row['id'],
                'pages': f"{row['start_page']}-{row['end_page']}"
            })
            
            ids.append(f"chunk_{row['id']}")
        
        if documents:
            self.vector_manager.add_documents(documents, metadatas, ids, embeddings)
    
    def batch_process(self, pdf_files: List[Path], options: dict) -> List[dict]:
        """Procesar múltiples manuales"""
        
        results = []
        total = len(pdf_files)
        
        logger.info(f"\nProcesando {total} manuales...")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{total}] Procesando {pdf_path.name}")
            
            try:
                result = self.process_manual(pdf_path, options)
                results.append(result)
            except Exception as e:
                logger.error(f"Error procesando {pdf_path.name}: {e}")
                results.append({
                    'pdf_path': str(pdf_path),
                    'pdf_name': pdf_path.name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Mostrar resumen final
        self._show_batch_summary(results)
        
        return results
    
    def _show_processing_summary(self, results: dict):
        """Mostrar resumen del procesamiento"""
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE PROCESAMIENTO")
        print(f"{'='*60}")
        print(f"Manual: {results['pdf_name']}")
        print(f"Estado: {'✓ Exitoso' if results['status'] == 'success' else '✗ Fallido'}")
        
        if results['status'] == 'success':
            print(f"Duración: {results['duration_seconds']:.1f} segundos")
            print(f"\nComponentes procesados:")
            
            for step, data in results['steps'].items():
                if data['status'] == 'completed':
                    if step == 'analysis':
                        print(f"  - Análisis: {data['document_type']}")
                    elif step == 'chunking':
                        print(f"  - Chunks: {data['chunks_created']}")
                    elif step == 'images':
                        print(f"  - Imágenes: {data['total_images']}")
                    elif step == 'tables':
                        print(f"  - Tablas: {data['total_tables']}")
                    elif step == 'embeddings':
                        print(f"  - Embeddings: {data['chunks_embedded']}")
        else:
            print(f"Error: {results.get('error', 'Desconocido')}")
        
        print(f"{'='*60}\n")
    
    def _show_batch_summary(self, results: List[dict]):
        """Mostrar resumen de procesamiento por lotes"""
        
        total = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = total - successful
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE PROCESAMIENTO POR LOTES")
        print(f"{'='*60}")
        print(f"Total de manuales: {total}")
        print(f"Exitosos: {successful} ({'%.1f' % (successful/total*100)}%)")
        print(f"Fallidos: {failed}")
        
        if failed > 0:
            print(f"\nManuales con errores:")
            for result in results:
                if result['status'] == 'failed':
                    print(f"  - {result['pdf_name']}: {result.get('error', 'Error desconocido')}")
        
        # Estadísticas agregadas
        total_chunks = sum(r.get('steps', {}).get('chunking', {}).get('chunks_created', 0) 
                          for r in results if r['status'] == 'success')
        total_images = sum(r.get('steps', {}).get('images', {}).get('total_images', 0) 
                          for r in results if r['status'] == 'success')
        total_tables = sum(r.get('steps', {}).get('tables', {}).get('total_tables', 0) 
                          for r in results if r['status'] == 'success')
        
        print(f"\nTotales procesados:")
        print(f"  - Chunks: {total_chunks}")
        print(f"  - Imágenes: {total_images}")
        print(f"  - Tablas: {total_tables}")
        
        print(f"{'='*60}\n")
    
    def _save_processing_log(self, manual_id: int, results: dict):
        """Guardar log detallado del procesamiento"""
        
        log_dir = self.config.DATA_DIR / 'logs' / 'processing'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"manual_{manual_id}_{timestamp}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def reprocess_manual(self, manual_id: int, steps: List[str]):
        """Reprocesar pasos específicos de un manual"""
        
        # Obtener información del manual
        manual = self.db.get_manual(manual_id)
        if not manual:
            raise ValueError(f"Manual {manual_id} no encontrado")
        
        pdf_path = Path(manual['file_path'])
        if not pdf_path.exists():
            pdf_path = self.config.BASE_DIR / manual['file_path']
            if not pdf_path.exists():
                raise FileNotFoundError(f"Archivo PDF no encontrado: {manual['file_path']}")
        
        logger.info(f"\nReprocesando manual {manual['name']} - Pasos: {steps}")
        logger.info("="*60)
        
        results = {
            'manual_id': manual_id,
            'pdf_name': manual['name'],
            'steps': {}
        }
        
        try:
            # Limpiar datos existentes si es necesario
            if 'chunks' in steps:
                self.db.conn.execute("DELETE FROM content_chunks WHERE manual_id = ?", (manual_id,))
                self.db.conn.commit()
            if 'images' in steps:
                self.db.conn.execute("DELETE FROM images WHERE manual_id = ?", (manual_id,))
                self.db.conn.commit()
            if 'tables' in steps:
                self.db.conn.execute("DELETE FROM tables WHERE manual_id = ?", (manual_id,))
                self.db.conn.commit()
            
            # Procesar solo los pasos solicitados
            if 'images' in steps:
                logger.info("\nExtrayendo imágenes y diagramas...")
                manual_name = manual['name']
                manual_folder_name = sanitize_folder_name(manual_name)
                
                # Importar y usar el extractor organizado
                sys.path.append(str(Path(__file__).parent))
                from extract_with_manual_folders import OrganizedContentExtractor
                organized_image_extractor = OrganizedContentExtractor(self.db, self.config.PROCESSED_DIR)
                image_results = organized_image_extractor.extract_and_store(pdf_path, manual_id, manual_name)
                
                results['steps']['images'] = {
                    'status': 'completed',
                    'total_images': image_results['total_images'],
                    'by_type': image_results.get('by_type', {})
                }
                logger.info(f"  → Extraídas {image_results['total_images']} imágenes")
                if image_results.get('by_type'):
                    for img_type, count in image_results['by_type'].items():
                        logger.info(f"    - {img_type}: {count}")
            
            if 'tables' in steps:
                logger.info("\nExtrayendo tablas...")
                if 'manual_name' not in locals():
                    manual_name = manual['name']
                    manual_folder_name = sanitize_folder_name(manual_name)
                
                tables_dir = self.config.PROCESSED_DIR / manual_folder_name / 'tables'
                tables_dir.mkdir(parents=True, exist_ok=True)
                
                from extract_with_manual_folders import OrganizedTableExtractor
                organized_table_extractor = OrganizedTableExtractor(self.db)
                table_results = organized_table_extractor.extract_and_store(pdf_path, manual_id, tables_dir)
                
                results['steps']['tables'] = {
                    'status': 'completed',
                    'total_tables': table_results['total_tables']
                }
                logger.info(f"  → Extraídas {table_results['total_tables']} tablas")
            
            if 'chunks' in steps:
                logger.info("\nCreando chunks de texto...")
                chunk_ids = self.text_processor.process_and_store(manual_id)
                results['steps']['chunking'] = {
                    'status': 'completed',
                    'chunks_created': len(chunk_ids)
                }
                logger.info(f"  → Creados {len(chunk_ids)} chunks")
            
            if 'embeddings' in steps:
                logger.info("\nGenerando embeddings...")
                embedding_results = self._generate_embeddings(manual_id)
                results['steps']['embeddings'] = {
                    'status': 'completed',
                    'chunks_embedded': embedding_results['chunks_embedded']
                }
                logger.info(f"  → Generados {embedding_results['chunks_embedded']} embeddings")
            
            # Actualizar totales en el manual
            if 'images' in steps or 'tables' in steps:
                total_images = self.db.conn.execute(
                    "SELECT COUNT(*) FROM images WHERE manual_id = ?",
                    (manual_id,)
                ).fetchone()[0]
                
                total_tables = self.db.conn.execute(
                    "SELECT COUNT(*) FROM tables WHERE manual_id = ?",
                    (manual_id,)
                ).fetchone()[0]
                
                self.db.conn.execute("""
                    UPDATE manuals 
                    SET total_images = ?, total_tables = ?
                    WHERE id = ?
                """, (total_images, total_tables, manual_id))
                self.db.conn.commit()
            
            # Mostrar estructura final
            self._show_final_structure(manual_id)
            
            logger.info("\n✅ Reprocesamiento completado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"Error en reprocesamiento: {e}")
            raise
    
    def close(self):
        """Cerrar conexiones"""
        self.db.close()
        if self.vector_manager:
            # El vector manager no tiene método close, pero podríamos añadirlo si es necesario
            pass
    
    def _show_final_structure(self, manual_id: int):
        """Mostrar la estructura de archivos creada"""
        manual_info = self.db.get_manual(manual_id)
        manual_name = manual_info['name']
        manual_folder = sanitize_folder_name(manual_name)
        
        logger.info(f"\nEstructura de archivos creada:")
        logger.info(f"  data/processed/{manual_folder}/")
        logger.info(f"    ├── images/     (imágenes raster)")
        logger.info(f"    ├── diagrams/   (diagramas renderizados)")
        logger.info(f"    └── tables/     (tablas CSV)")


def main():
    parser = argparse.ArgumentParser(
        description='Procesar manuales PDF con almacenamiento en SQLite'
    )
    
    # Entrada
    parser.add_argument('input', nargs='?', help='Archivo PDF o directorio')
    parser.add_argument('--pdf-dir', type=str, help='Directorio con PDFs')
    parser.add_argument('--single-pdf', type=str, help='Procesar un solo PDF')
    
    # Opciones de procesamiento
    parser.add_argument('--skip-analysis', action='store_true', help='Saltar análisis del documento')
    parser.add_argument('--skip-chunks', action='store_true', help='Saltar creación de chunks')
    parser.add_argument('--skip-images', action='store_true', help='Saltar extracción de imágenes')
    parser.add_argument('--skip-tables', action='store_true', help='Saltar extracción de tablas')
    parser.add_argument('--embeddings', action='store_true', help='Generar embeddings')
    
    # Metadatos
    parser.add_argument('--manufacturer', type=str, help='Fabricante del equipo')
    parser.add_argument('--model', type=str, help='Modelo del equipo')
    parser.add_argument('--doc-type', type=str, choices=['technical', 'user', 'maintenance'], 
                       default='technical', help='Tipo de documento')
    
    # Base de datos
    parser.add_argument('--db-path', type=str, help='Ruta a la base de datos SQLite')
    
    # Reprocesamiento
    parser.add_argument('--reprocess', type=int, help='ID del manual a reprocesar')
    parser.add_argument('--steps', nargs='+', 
                       choices=['analysis', 'chunks', 'images', 'tables', 'embeddings'],
                       help='Pasos a reprocesar')
    
    args = parser.parse_args()
    
    # Configuración
    config = Config()
    
    # Crear procesador
    processor = SQLiteManualProcessor(config, args.db_path)
    
    try:
        # Modo reprocesamiento
        if args.reprocess:
            if not args.steps:
                print("Error: Debe especificar los pasos a reprocesar con --steps")
                return
            
            processor.reprocess_manual(args.reprocess, args.steps)
            return
        
        # Determinar archivos a procesar
        pdf_files = []
        
        if args.single_pdf:
            pdf_files = [Path(args.single_pdf)]
        elif args.pdf_dir:
            pdf_dir = Path(args.pdf_dir)
            pdf_files = list(pdf_dir.glob("*.pdf"))
        elif args.input:
            input_path = Path(args.input)
            if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                pdf_files = [input_path]
            elif input_path.is_dir():
                pdf_files = list(input_path.glob("*.pdf"))
        else:
            # Sin argumentos, usar directorio por defecto
            pdf_files = list(config.RAW_PDF_DIR.glob("*.pdf"))
        
        if not pdf_files:
            print("No se encontraron archivos PDF para procesar")
            return
        
        # Preparar opciones
        options = {
            'skip_analysis': args.skip_analysis,
            'skip_chunks': args.skip_chunks,
            'skip_images': args.skip_images,
            'skip_tables': args.skip_tables,
            'generate_embeddings': args.embeddings,
            'manufacturer': args.manufacturer,
            'model': args.model,
            'document_type': args.doc_type
        }
        
        # Procesar
        if len(pdf_files) == 1:
            processor.process_manual(pdf_files[0], options)
        else:
            processor.batch_process(pdf_files, options)
    
    finally:
        processor.close()


if __name__ == "__main__":
    import numpy as np  # Necesario para embeddings
    main()