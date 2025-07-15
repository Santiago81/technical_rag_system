
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

from .document_analyzer import DocumentAnalyzer, DocumentType
from .pdf_extractor import PDFExtractor
from .table_extractor import TableExtractor
from .enhanced_image_extractor import EnhancedImageExtractor as ImageExtractor
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)

class AdaptiveManualProcessor:
    """Procesador que adapta su estrategia según el tipo de documento"""
    
    def __init__(self, config):
        self.config = config
        self.analyzer = DocumentAnalyzer()
        
        # Inicializar extractores
        self.pdf_extractor = PDFExtractor(config)
        self.table_extractor = TableExtractor()
        self.image_extractor = ImageExtractor(
            min_size=config.MIN_IMAGE_SIZE,
            output_format=config.IMAGE_OUTPUT_FORMAT
        )
        
        # Directorio para análisis
        self.analysis_dir = config.DATA_DIR / "document_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
    
    def process_manual(self, pdf_path: Path, force_strategy: Optional[Dict] = None) -> Dict:
        """Procesar manual con estrategia adaptativa"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesando: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        # Fase 1: Análisis del documento
        if force_strategy:
            logger.info("Usando estrategia forzada por el usuario")
            analysis = {'extraction_strategy': force_strategy}
            doc_type = "forced"
        else:
            logger.info("Fase 1: Analizando tipo de documento...")
            analysis = self.analyzer.analyze_document(str(pdf_path))
            doc_type = analysis['document_type']
            
            # Guardar análisis
            self._save_analysis(pdf_path.stem, analysis)
            
            # Mostrar resultados del análisis
            self._display_analysis_results(analysis)
        
        # Fase 2: Aplicar estrategia de extracción
        logger.info(f"\nFase 2: Aplicando estrategia para tipo: {doc_type}")
        strategy = analysis['extraction_strategy']
        
        # Configurar procesador de texto según estrategia
        self.text_processor = TextProcessor(
            chunk_size=strategy.get('chunk_size', 512),
            chunk_overlap=50
        )
        
        # Ejecutar extracción según estrategia
        extraction_results = self._execute_extraction(pdf_path, strategy, analysis)
        
        # Fase 3: Post-procesamiento específico por tipo
        logger.info("\nFase 3: Post-procesamiento...")
        final_results = self._post_process_by_type(
            extraction_results, 
            doc_type, 
            strategy
        )
        
        # Guardar reporte de procesamiento
        self._save_processing_report(pdf_path.stem, {
            'analysis': analysis,
            'extraction_results': extraction_results,
            'final_results': final_results,
            'timestamp': datetime.now().isoformat()
        })
        
        return final_results
    
    def _execute_extraction(self, pdf_path: Path, strategy: Dict, 
                          analysis: Dict) -> Dict:
        """Ejecutar extracción según estrategia"""
        manual_name = pdf_path.stem
        results = {
            'manual_name': manual_name,
            'extraction_strategy': strategy,
            'extracted_components': {}
        }
        
        # 1. Extracción de texto (siempre primero si está habilitado)
        if strategy.get('extract_text', True):
            logger.info("Extrayendo texto...")
            text_data = self.pdf_extractor.extract_all_content(pdf_path)
            results['extracted_components']['text'] = {
                'pages': len(text_data['pages']),
                'data': text_data
            }
        
        # 2. Extracción de tablas
        if strategy.get('extract_tables', False):
            logger.info("Extrayendo tablas...")
            tables_dir = self.config.PROCESSED_DIR / 'tables' / manual_name
            tables_dir.mkdir(parents=True, exist_ok=True)
            
            # Usar método específico si está definido
            method = strategy.get('table_extraction_method', 'camelot_first')
            tables = self._extract_tables_adaptive(pdf_path, tables_dir, method)
            
            results['extracted_components']['tables'] = {
                'count': len(tables),
                'data': tables
            }
        
        # 3. Extracción de imágenes raster
        if strategy.get('extract_images', False):
            logger.info("Extrayendo imágenes...")
            images_dir = self.config.PROCESSED_DIR / 'images' / manual_name
            images_dir.mkdir(parents=True, exist_ok=True)
            
            images = self.image_extractor.extract_images_from_pdf(
                pdf_path, images_dir
            )
            
            results['extracted_components']['images'] = {
                'count': len(images),
                'data': images
            }
        
        # 4. Extracción de diagramas (renderizado de páginas)
        if strategy.get('extract_diagrams', False):
            logger.info("Extrayendo diagramas...")
            diagrams_dir = self.config.PROCESSED_DIR / 'diagrams' / manual_name
            diagrams_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar DPI según estrategia
            self.image_extractor.diagram_dpi = strategy.get('diagram_dpi', 150)
            
            diagrams = self._extract_diagrams_adaptive(
                pdf_path, diagrams_dir, analysis
            )
            
            results['extracted_components']['diagrams'] = {
                'count': len(diagrams),
                'data': diagrams
            }
        
        # 5. Aplicar OCR si es necesario
        if strategy.get('use_ocr', False):
            logger.info("Aplicando OCR...")
            ocr_results = self._apply_ocr_strategy(results, strategy)
            results['extracted_components']['ocr'] = ocr_results
        
        return results
    
    def _extract_tables_adaptive(self, pdf_path: Path, output_dir: Path, 
                                method: str) -> List[Dict]:
        """Extracción adaptativa de tablas según método preferido"""
        if method == 'camelot_first':
            # Intentar Camelot primero, luego alternativas
            tables = self.table_extractor.extract_tables_from_pdf(
                str(pdf_path), output_dir
            )
        else:
            # Método directo especificado
            tables = self.table_extractor.extract_tables_from_pdf(
                str(pdf_path), output_dir
            )
        
        return tables
    
    def _extract_diagrams_adaptive(self, pdf_path: Path, output_dir: Path,
                                  analysis: Dict) -> List[Dict]:
        """Extracción adaptativa de diagramas según análisis"""
        # Para documentos técnicos, ser más agresivo en la detección
        doc_type = analysis.get('document_type', '')
        
        if doc_type == DocumentType.TECHNICAL_DIAGRAM_HEAVY:
            # Extraer más páginas como diagramas
            self.image_extractor.diagram_threshold = 0.3  # Más permisivo
        else:
            self.image_extractor.diagram_threshold = 0.5  # Normal
        
        diagrams = self.image_extractor.extract_diagrams_as_images(
            str(pdf_path), output_dir
        )
        
        return diagrams
    
    def _apply_ocr_strategy(self, extraction_results: Dict, 
                           strategy: Dict) -> Dict:
        """Aplicar OCR según la estrategia"""
        ocr_results = {
            'pages_processed': 0,
            'text_extracted': []
        }
        
        priority = strategy.get('priority', 'balanced')
        
        if priority == 'ocr':
            # Para documentos escaneados, OCR en todas las páginas
            logger.info("Aplicando OCR a todas las páginas (documento escaneado)")
            # Implementar OCR completo
            pass
        
        elif priority == 'diagrams':
            # OCR solo en diagramas extraídos
            diagrams = extraction_results.get('extracted_components', {}).get('diagrams', {})
            if diagrams.get('data'):
                logger.info(f"Aplicando OCR a {diagrams['count']} diagramas")
                # Implementar OCR en diagramas
                pass
        
        return ocr_results
    
    def _post_process_by_type(self, extraction_results: Dict, 
                             doc_type: str, strategy: Dict) -> Dict:
        """Post-procesamiento específico según tipo de documento"""
        
        # Copiar resultados base
        final_results = extraction_results.copy()
        
        if doc_type == DocumentType.TECHNICAL_DIAGRAM_HEAVY:
            # Para manuales técnicos con diagramas
            final_results['processing_notes'] = [
                "Documento técnico con énfasis en diagramas",
                "Se recomienda revisar diagramas extraídos para referencias cruzadas",
                "Los chunks de texto son más grandes para preservar contexto técnico"
            ]
            
            # Crear índice de diagramas por página
            final_results['diagram_index'] = self._create_diagram_index(extraction_results)
        
        elif doc_type == DocumentType.TABLE_HEAVY:
            # Para documentos con muchas tablas
            final_results['processing_notes'] = [
                "Documento con alta densidad de tablas",
                "Las tablas se han extraído en formatos CSV y Excel",
                "Revisar integridad de datos tabulares"
            ]
            
            # Crear índice de tablas
            final_results['table_index'] = self._create_table_index(extraction_results)
        
        elif doc_type == DocumentType.SCANNED_DOCUMENT:
            # Para documentos escaneados
            final_results['processing_notes'] = [
                "Documento escaneado - calidad puede variar",
                "Se aplicó OCR para extracción de texto",
                "Verificar precisión del texto extraído"
            ]
        
        # Agregar estadísticas generales
        final_results['statistics'] = self._calculate_statistics(extraction_results)
        
        return final_results
    
    def _create_diagram_index(self, results: Dict) -> Dict:
        """Crear índice de diagramas por página"""
        diagrams = results.get('extracted_components', {}).get('diagrams', {}).get('data', [])
        
        index = {}
        for diagram in diagrams:
            page = diagram.get('page_number', 0)
            if page not in index:
                index[page] = []
            index[page].append({
                'filename': diagram.get('image_filename'),
                'path': diagram.get('image_path')
            })
        
        return index
    
    def _create_table_index(self, results: Dict) -> Dict:
        """Crear índice de tablas"""
        tables = results.get('extracted_components', {}).get('tables', {}).get('data', [])
        
        index = {
            'by_page': {},
            'total_tables': len(tables),
            'formats_available': ['csv', 'xlsx']
        }
        
        for table in tables:
            page = table.get('page_number', 0)
            if page not in index['by_page']:
                index['by_page'][page] = []
            
            index['by_page'][page].append({
                'table_index': table.get('table_index'),
                'rows': table.get('rows', 0),
                'columns': table.get('columns', 0),
                'csv_path': table.get('csv_path'),
                'excel_path': table.get('excel_path')
            })
        
        return index
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calcular estadísticas del procesamiento"""
        components = results.get('extracted_components', {})
        
        stats = {
            'total_pages': 0,
            'components_extracted': list(components.keys()),
            'extraction_summary': {}
        }
        
        # Estadísticas por componente
        if 'text' in components:
            text_data = components['text'].get('data', {})
            stats['total_pages'] = len(text_data.get('pages', []))
            stats['extraction_summary']['text'] = {
                'pages_with_content': sum(1 for p in text_data.get('pages', []) 
                                         if p.get('has_content', False))
            }
        
        for component in ['tables', 'images', 'diagrams']:
            if component in components:
                stats['extraction_summary'][component] = {
                    'count': components[component].get('count', 0)
                }
        
        return stats
    
    def _display_analysis_results(self, analysis: Dict):
        """Mostrar resultados del análisis de forma legible"""
        print(f"\nTipo de documento detectado: {analysis.get('document_type', 'DESCONOCIDO')}")
        
        if 'page_analysis' in analysis:
            pa = analysis['page_analysis']
            print(f"Características detectadas:")
            print(f"  - Densidad de texto: {pa.get('avg_text_density', 0):.2f}")
            print(f"  - Imágenes por página: {pa.get('avg_images_per_page', 0):.1f}")
            print(f"  - Gráficos vectoriales: {pa.get('avg_vector_graphics', 0):.1f}")
            print(f"  - Frecuencia de tablas: {pa.get('table_frequency', 0):.1%}")
        
        if 'recommendations' in analysis:
            print("\nRecomendaciones:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
    
    def _save_analysis(self, manual_name: str, analysis: Dict):
        """Guardar análisis del documento"""
        analysis_file = self.analysis_dir / f"{manual_name}_analysis.json"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Análisis guardado en: {analysis_file}")
    
    def _save_processing_report(self, manual_name: str, report: Dict):
        """Guardar reporte completo del procesamiento"""
        report_file = self.analysis_dir / f"{manual_name}_processing_report.json"
        
        # Limpiar datos grandes para el reporte
        clean_report = self._clean_report_for_saving(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Reporte de procesamiento guardado en: {report_file}")
    
    def _clean_report_for_saving(self, report: Dict) -> Dict:
        """Limpiar reporte para guardado (eliminar datos grandes)"""
        clean = report.copy()
        
        # Eliminar datos de páginas completas
        if 'extraction_results' in clean:
            components = clean['extraction_results'].get('extracted_components', {})
            if 'text' in components and 'data' in components['text']:
                # Solo mantener estadísticas
                components['text']['data'] = {
                    'pages_count': len(components['text']['data'].get('pages', []))
                }
        
        return clean

# Modificación del script process_manuals.py para usar el procesador adaptativo
class AdaptiveManualProcessorScript:
    """Script actualizado que usa procesamiento adaptativo"""
    
    @staticmethod
    def process_with_analysis(pdf_files: List[Path], config, 
                            force_type: Optional[str] = None):
        """Procesar PDFs con análisis adaptativo"""
        
        processor = AdaptiveManualProcessor(config)
        
        for pdf_path in pdf_files:
            try:
                # Si se fuerza un tipo específico
                if force_type:
                    strategy = processor._get_forced_strategy(force_type)
                    results = processor.process_manual(pdf_path, strategy)
                else:
                    # Procesamiento adaptativo completo
                    results = processor.process_manual(pdf_path)
                
                print(f"\n✓ Procesamiento completado para: {pdf_path.name}")
                print(f"  Componentes extraídos: {list(results.get('extracted_components', {}).keys())}")
                
            except Exception as e:
                logger.error(f"Error procesando {pdf_path}: {e}")
                print(f"\n✗ Error en: {pdf_path.name}")

def main():
    """Ejemplo de uso del procesador adaptativo"""
    from config.settings import Config
    
    config = Config()
    processor = AdaptiveManualProcessor(config)
    
    # Procesar un PDF con análisis automático
    pdf_path = Path("manual_ejemplo.pdf")
    results = processor.process_manual(pdf_path)
    
    print("\nProcesamiento completado!")
    print(f"Estadísticas: {results.get('statistics', {})}")

if __name__ == "__main__":
    main()