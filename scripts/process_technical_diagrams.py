
"""
Script especializado para procesar manuales técnicos con diagramas vectoriales
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List
import sys

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from extractors.enhanced_image_extractor import EnhancedImageExtractor, analyze_pdf_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalDiagramProcessor:
    """Procesador especializado para manuales técnicos"""
    
    def __init__(self, output_base_dir: Path = None):
        self.output_base_dir = output_base_dir or Path("data/processed/visual_content")
        self.extractor = EnhancedImageExtractor(
            min_size=(100, 100),
            output_format='png',  # PNG para diagramas técnicos
            diagram_dpi=200  # Alta calidad para diagramas
        )
        
    def process_manual(self, pdf_path: Path) -> Dict:
        """Procesar un manual técnico completo"""
        logger.info(f"\nProcesando: {pdf_path.name}")
        logger.info("="*60)
        
        # 1. Análisis inicial
        logger.info("Fase 1: Analizando estructura del documento...")
        analysis = analyze_pdf_content(str(pdf_path))
        self._print_analysis(analysis)
        
        # 2. Crear directorio de salida
        manual_name = pdf_path.stem
        output_dir = self.output_base_dir / manual_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Extracción adaptativa
        logger.info("\nFase 2: Extrayendo contenido visual...")
        results = self.extractor.extract_all_content(str(pdf_path), output_dir)
        
        # 4. Guardar metadatos y resultados
        self._save_results(output_dir, results, analysis)
        
        # 5. Mostrar resumen
        self._print_summary(results)
        
        return results
    
    def _print_analysis(self, analysis: Dict):
        """Imprimir análisis del documento"""
        print(f"\nAnálisis del documento:")
        print(f"  Total de páginas: {analysis['total_pages']}")
        print(f"  Distribución de contenido:")
        
        for content_type, count in analysis['content_summary'].items():
            if count > 0:
                print(f"    - {content_type}: {count}")
        
        if analysis['recommendations']:
            print(f"\n  Recomendaciones:")
            for rec in analysis['recommendations']:
                print(f"    • {rec}")
    
    def _save_results(self, output_dir: Path, results: Dict, analysis: Dict):
        """Guardar resultados y metadatos"""
        # Guardar metadatos completos
        metadata_file = output_dir / "extraction_metadata.json"
        
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'document_analysis': analysis,
            'extraction_results': {
                'statistics': results['statistics'],
                'raster_images': len(results['raster_images']),
                'vector_diagrams': len(results['vector_diagrams']),
                'mixed_pages': len(results['mixed_pages'])
            },
            'files': {
                'raster_images': [img['image_filename'] for img in results['raster_images']],
                'vector_diagrams': [diag['image_filename'] for diag in results['vector_diagrams']],
                'mixed_pages': [page['image_filename'] for page in results['mixed_pages']]
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Guardar índice de contenido visual
        index_file = output_dir / "visual_content_index.json"
        
        visual_index = {
            'manual_name': results['manual_name'],
            'content': []
        }
        
        # Agregar todas las imágenes y diagramas al índice
        for img in results['raster_images']:
            visual_index['content'].append({
                'page': img['page_number'],
                'type': img['content_type'],
                'file': img['image_filename'],
                'method': img['extraction_method']
            })
        
        for diag in results['vector_diagrams']:
            visual_index['content'].append({
                'page': diag['page_number'],
                'type': 'technical_diagram',
                'file': diag['image_filename'],
                'method': 'page_render',
                'has_ocr': 'ocr_text' in diag
            })
        
        # Ordenar por página
        visual_index['content'].sort(key=lambda x: x['page'])
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(visual_index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadatos guardados en: {metadata_file}")
        logger.info(f"Índice visual guardado en: {index_file}")
    
    def _print_summary(self, results: Dict):
        """Imprimir resumen de la extracción"""
        stats = results['statistics']
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE EXTRACCIÓN")
        print(f"{'='*60}")
        print(f"Manual: {results['manual_name']}")
        print(f"\nContenido extraído:")
        print(f"  - Imágenes raster: {stats['total_raster_images']}")
        print(f"  - Diagramas vectoriales: {stats['total_vector_diagrams']}")
        print(f"  - Páginas mixtas procesadas: {len(results['mixed_pages'])}")
        print(f"\nDistribución de páginas:")
        print(f"  - Páginas con vectores: {stats['vector_pages']}")
        print(f"  - Páginas con imágenes: {stats['raster_pages']}")
        print(f"  - Páginas mixtas: {stats['mixed_pages']}")
        print(f"{'='*60}\n")
    
    def process_specific_pages(self, pdf_path: Path, page_numbers: List[int]) -> List[Dict]:
        """Procesar solo páginas específicas como diagramas"""
        logger.info(f"\nProcesando páginas específicas de: {pdf_path.name}")
        logger.info(f"Páginas: {page_numbers}")
        
        manual_name = pdf_path.stem
        output_dir = self.output_base_dir / manual_name / "specific_diagrams"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        diagrams = self.extractor.extract_specific_diagrams(
            str(pdf_path), output_dir, page_numbers
        )
        
        logger.info(f"Extraídos {len(diagrams)} diagramas")
        return diagrams
    
    def batch_process(self, pdf_dir: Path) -> Dict[str, Dict]:
        """Procesar múltiples PDFs"""
        pdf_files = list(pdf_dir.glob("*.pdf"))
        results = {}
        
        logger.info(f"Procesando {len(pdf_files)} archivos PDF...")
        
        for pdf_path in pdf_files:
            try:
                result = self.process_manual(pdf_path)
                results[pdf_path.name] = {
                    'status': 'success',
                    'statistics': result['statistics']
                }
            except Exception as e:
                logger.error(f"Error procesando {pdf_path.name}: {e}")
                results[pdf_path.name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Guardar resumen del batch
        summary_file = self.output_base_dir / "batch_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'processing_date': datetime.now().isoformat(),
                'total_files': len(pdf_files),
                'results': results
            }, f, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Procesar manuales técnicos con diagramas vectoriales'
    )
    
    parser.add_argument('pdf_path', type=str,
                       help='Ruta al archivo PDF o directorio')
    
    parser.add_argument('--output-dir', type=str,
                       help='Directorio de salida')
    
    parser.add_argument('--pages', type=str,
                       help='Páginas específicas a procesar (ej: 1,3,5-7)')
    
    parser.add_argument('--batch', action='store_true',
                       help='Procesar todos los PDFs en el directorio')
    
    parser.add_argument('--analyze-only', action='store_true',
                       help='Solo analizar sin extraer')
    
    args = parser.parse_args()
    
    # Configurar procesador
    output_dir = Path(args.output_dir) if args.output_dir else None
    processor = TechnicalDiagramProcessor(output_dir)
    
    pdf_path = Path(args.pdf_path)
    
    # Solo análisis
    if args.analyze_only:
        if pdf_path.is_file():
            analysis = analyze_pdf_content(str(pdf_path))
            print(json.dumps(analysis, indent=2))
        else:
            print("Error: Se requiere un archivo PDF para análisis")
        return
    
    # Procesamiento batch
    if args.batch:
        if pdf_path.is_dir():
            processor.batch_process(pdf_path)
        else:
            print("Error: Se requiere un directorio para procesamiento batch")
        return
    
    # Páginas específicas
    if args.pages:
        page_numbers = []
        for part in args.pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_numbers.extend(range(start-1, end))  # Convertir a 0-indexed
            else:
                page_numbers.append(int(part)-1)
        
        processor.process_specific_pages(pdf_path, page_numbers)
        return
    
    # Procesamiento normal
    if pdf_path.is_file():
        processor.process_manual(pdf_path)
    else:
        print(f"Error: {pdf_path} no es un archivo válido")


if __name__ == "__main__":
    main()