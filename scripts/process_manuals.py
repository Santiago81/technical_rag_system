

"""
Script para procesar manuales PDF con análisis adaptativo
"""
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config
from extractors.adaptive_processor import AdaptiveManualProcessor
from extractors.document_analyzer import DocumentType
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManualProcessingScript:
    """Script principal para procesamiento de manuales"""
    
    def __init__(self):
        self.config = Config()
        self.processor = AdaptiveManualProcessor(self.config)
        
        # Directorio para logs
        self.log_dir = self.config.DATA_DIR / "processing_logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def process_pdfs(self, pdf_files: List[Path], args):
        """Procesar lista de PDFs con opciones especificadas"""
        
        # Log de inicio
        processing_log = {
            'start_time': datetime.now().isoformat(),
            'total_files': len(pdf_files),
            'mode': 'adaptive' if not args.force_type else f'forced_{args.force_type}',
            'files_processed': [],
            'errors': []
        }
        
        # Procesar cada PDF
        for pdf_path in tqdm(pdf_files, desc="Procesando PDFs"):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Procesando: {pdf_path.name}")
                logger.info(f"{'='*60}")
                
                # Determinar estrategia
                if args.force_type:
                    # Usar tipo forzado
                    strategy = self._get_forced_strategy(args.force_type, args)
                    results = self.processor.process_manual(pdf_path, strategy)
                elif args.analyze_only:
                    # Solo analizar, no procesar
                    analysis = self.processor.analyzer.analyze_document(str(pdf_path))
                    self._display_analysis(pdf_path, analysis)
                    continue
                else:
                    # Procesamiento adaptativo completo
                    results = self.processor.process_manual(pdf_path)
                
                # Registrar éxito
                processing_log['files_processed'].append({
                    'filename': pdf_path.name,
                    'status': 'success',
                    'document_type': results.get('document_type', 'unknown'),
                    'components_extracted': list(results.get('extracted_components', {}).keys())
                })
                
            except Exception as e:
                logger.error(f"Error procesando {pdf_path.name}: {str(e)}")
                processing_log['errors'].append({
                    'filename': pdf_path.name,
                    'error': str(e)
                })
        
        # Guardar log
        processing_log['end_time'] = datetime.now().isoformat()
        self._save_processing_log(processing_log)
        
        # Mostrar resumen
        self._show_summary(processing_log)
    
    def _get_forced_strategy(self, force_type: str, args) -> Dict[str, any]:
        """Obtener estrategia forzada según tipo especificado"""
        
        # Mapeo de tipos a estrategias
        type_strategies = {
            'technical': {
                'extract_text': True,
                'extract_tables': not args.no_tables,
                'extract_images': False,
                'extract_diagrams': True,
                'use_ocr': True,
                'diagram_dpi': 200,
                'chunk_size': 1024,
                'priority': 'diagrams'
            },
            'text': {
                'extract_text': True,
                'extract_tables': not args.no_tables,
                'extract_images': False,
                'extract_diagrams': False,
                'use_ocr': False,
                'chunk_size': 512,
                'priority': 'text'
            },
            'scanned': {
                'extract_text': False,
                'extract_tables': False,
                'extract_images': True,
                'extract_diagrams': True,
                'use_ocr': True,
                'diagram_dpi': 300,
                'chunk_size': 768,
                'priority': 'ocr'
            },
            'mixed': {
                'extract_text': True,
                'extract_tables': not args.no_tables,
                'extract_images': not args.no_images,
                'extract_diagrams': args.extract_diagrams,
                'use_ocr': False,
                'diagram_dpi': 150,
                'chunk_size': 512,
                'priority': 'balanced'
            }
        }
        
        strategy = type_strategies.get(force_type, type_strategies['mixed'])
        
        # Aplicar overrides de línea de comandos
        if args.no_images:
            strategy['extract_images'] = False
        if args.no_tables:
            strategy['extract_tables'] = False
        if args.extract_diagrams:
            strategy['extract_diagrams'] = True
        
        return strategy
    
    def _display_analysis(self, pdf_path: Path, analysis: Dict[str, any]):
        """Mostrar análisis de documento"""
        print(f"\n=== Análisis de {pdf_path.name} ===")
        print(f"Tipo detectado: {analysis.get('document_type', 'DESCONOCIDO')}")
        
        if 'basic_info' in analysis:
            info = analysis['basic_info']
            print(f"\nInformación básica:")
            print(f"  - Páginas: {info.get('total_pages', 0)}")
            print(f"  - Tamaño: {info.get('file_size_mb', 0):.1f} MB")
            print(f"  - Encriptado: {'Sí' if info.get('is_encrypted') else 'No'}")
            print(f"  - Tiene índice: {'Sí' if info.get('has_toc') else 'No'}")
        
        if 'page_analysis' in analysis:
            pa = analysis['page_analysis']
            print(f"\nAnálisis de contenido:")
            print(f"  - Densidad de texto: {pa.get('avg_text_density', 0):.3f}")
            print(f"  - Imágenes/página: {pa.get('avg_images_per_page', 0):.1f}")
            print(f"  - Gráficos vectoriales/página: {pa.get('avg_vector_graphics', 0):.1f}")
            print(f"  - Frecuencia de tablas: {pa.get('table_frequency', 0):.1%}")
            print(f"  - Documento a color: {'Sí' if pa.get('color_document') else 'No'}")
        
        if 'extraction_strategy' in analysis:
            strategy = analysis['extraction_strategy']
            print(f"\nEstrategia recomendada:")
            print(f"  - Extraer texto: {'Sí' if strategy.get('extract_text') else 'No'}")
            print(f"  - Extraer tablas: {'Sí' if strategy.get('extract_tables') else 'No'}")
            print(f"  - Extraer imágenes: {'Sí' if strategy.get('extract_images') else 'No'}")
            print(f"  - Extraer diagramas: {'Sí' if strategy.get('extract_diagrams') else 'No'}")
            print(f"  - Usar OCR: {'Sí' if strategy.get('use_ocr') else 'No'}")
            print(f"  - Tamaño de chunk: {strategy.get('chunk_size', 512)}")
        
        if 'recommendations' in analysis:
            print(f"\nRecomendaciones:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
    
    def analyze_directory(self, directory: Path):
        """Analizar todos los PDFs en un directorio sin procesarlos"""
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print("No se encontraron archivos PDF")
            return
        
        print(f"\nAnalizando {len(pdf_files)} PDFs...\n")
        
        # Estadísticas agregadas
        type_counts = {}
        all_analyses = []
        
        for pdf_path in pdf_files:
            try:
                analysis = self.processor.analyzer.analyze_document(str(pdf_path))
                doc_type = analysis.get('document_type', 'unknown')
                
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                all_analyses.append({
                    'filename': pdf_path.name,
                    'type': doc_type,
                    'pages': analysis.get('basic_info', {}).get('total_pages', 0)
                })
                
                print(f"✓ {pdf_path.name}: {doc_type}")
                
            except Exception as e:
                print(f"✗ {pdf_path.name}: Error - {str(e)}")
        
        # Mostrar resumen
        print(f"\n=== Resumen del Análisis ===")
        print(f"Total de PDFs analizados: {len(pdf_files)}")
        print(f"\nDistribución por tipo:")
        for doc_type, count in sorted(type_counts.items()):
            percentage = (count / len(pdf_files)) * 100
            print(f"  - {doc_type}: {count} ({percentage:.1f}%)")
        
        # Guardar resumen
        summary_file = directory / "document_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'total_files': len(pdf_files),
                'type_distribution': type_counts,
                'file_analyses': all_analyses
            }, f, indent=2)
        
        print(f"\nResumen guardado en: {summary_file}")
    
    def _save_processing_log(self, log_data: Dict[str, any]):
        """Guardar log de procesamiento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"processing_log_{timestamp}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Log guardado en: {log_file}")
    
    def _show_summary(self, log_data: Dict[str, any]):
        """Mostrar resumen del procesamiento"""
        print("\n" + "="*60)
        print("RESUMEN DEL PROCESAMIENTO")
        print("="*60)
        
        total = log_data['total_files']
        processed = len(log_data['files_processed'])
        errors = len(log_data['errors'])
        
        print(f"Archivos procesados: {processed}/{total}")
        print(f"Errores: {errors}")
        
        # Estadísticas por tipo de documento
        type_counts = {}
        for file_info in log_data['files_processed']:
            doc_type = file_info.get('document_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        if type_counts:
            print(f"\nDocumentos por tipo:")
            for doc_type, count in sorted(type_counts.items()):
                print(f"  - {doc_type}: {count}")
        
        if errors > 0:
            print(f"\nArchivos con errores:")
            for error in log_data['errors']:
                print(f"  - {error['filename']}: {error['error']}")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Procesar manuales PDF con análisis adaptativo'
    )
    
    # Opciones de entrada
    parser.add_argument('--pdf-dir', type=str,
                       help='Directorio con PDFs a procesar')
    parser.add_argument('--single-pdf', type=str,
                       help='Procesar un solo PDF')
    
    # Modo de operación
    parser.add_argument('--analyze-only', action='store_true',
                       help='Solo analizar documentos sin procesarlos')
    parser.add_argument('--analyze-dir', type=str,
                       help='Analizar todos los PDFs en un directorio')
    
    # Forzar tipo de documento
    parser.add_argument('--force-type', 
                       choices=['technical', 'text', 'scanned', 'mixed'],
                       help='Forzar tipo de documento en lugar de detectar')
    
    # Opciones de extracción
    parser.add_argument('--no-tables', action='store_true',
                       help='No extraer tablas')
    parser.add_argument('--no-images', action='store_true',
                       help='No extraer imágenes')
    parser.add_argument('--extract-diagrams', action='store_true',
                       help='Extraer diagramas (renderizar páginas)')
    
    # Otras opciones
    parser.add_argument('--verify', type=str,
                       help='Verificar datos procesados de un manual')
    
    args = parser.parse_args()
    
    # Inicializar script
    script = ManualProcessingScript()
    
    # Modo análisis de directorio
    if args.analyze_dir:
        script.analyze_directory(Path(args.analyze_dir))
        return
    
    # Determinar PDFs a procesar
    if args.single_pdf:
        pdf_files = [Path(args.single_pdf)]
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
    else:
        pdf_files = list(script.config.RAW_PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No se encontraron archivos PDF para procesar")
        return
    
    logger.info(f"Encontrados {len(pdf_files)} PDFs")
    
    # Procesar PDFs
    script.process_pdfs(pdf_files, args)

if __name__ == "__main__":
    main()