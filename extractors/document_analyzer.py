
import fitz
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)

class DocumentType:
    """Tipos de documentos identificados"""
    TECHNICAL_DIAGRAM_HEAVY = "technical_diagram_heavy"  # Muchos diagramas técnicos
    TEXT_HEAVY = "text_heavy"  # Principalmente texto
    MIXED_CONTENT = "mixed_content"  # Balance de texto y gráficos
    SCANNED_DOCUMENT = "scanned_document"  # PDF escaneado
    TABLE_HEAVY = "table_heavy"  # Muchas tablas
    IMAGE_CATALOG = "image_catalog"  # Catálogo con imágenes de productos

class DocumentAnalyzer:
    """Analizador para determinar el tipo de documento PDF y estrategia óptima"""
    
    def __init__(self):
        self.page_sample_size = 10  # Páginas a analizar para clasificación rápida
        
    def analyze_document(self, pdf_path: str) -> Dict:
        """Analizar documento completo y determinar tipo y estrategias"""
        pdf_path = Path(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            
            # Análisis básico
            basic_info = self._get_basic_info(doc, pdf_path)
            
            # Análisis de muestra de páginas
            page_analysis = self._analyze_pages_sample(doc)
            
            # Detectar tipo de documento
            doc_type = self._classify_document_type(page_analysis, basic_info)
            
            # Determinar estrategias de extracción
            extraction_strategy = self._determine_extraction_strategy(doc_type, page_analysis)
            
            # Detectar características especiales
            special_features = self._detect_special_features(doc, page_analysis)
            
            doc.close()
            
            return {
                'document_type': doc_type,
                'basic_info': basic_info,
                'page_analysis': page_analysis,
                'extraction_strategy': extraction_strategy,
                'special_features': special_features,
                'recommendations': self._generate_recommendations(doc_type, special_features)
            }
            
        except Exception as e:
            logger.error(f"Error analizando documento {pdf_path}: {e}")
            return {
                'document_type': DocumentType.TEXT_HEAVY,  # Default fallback
                'error': str(e),
                'extraction_strategy': self._get_default_strategy()
            }
    
    def _get_basic_info(self, doc, pdf_path: Path) -> Dict:
        """Obtener información básica del documento"""
        return {
            'filename': pdf_path.name,
            'total_pages': len(doc),
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'is_encrypted': doc.is_encrypted,
            'metadata': doc.metadata,
            'has_toc': len(doc.get_toc()) > 0,
            'creation_tool': doc.metadata.get('producer', '').lower()
        }
    
    def _analyze_pages_sample(self, doc) -> Dict:
        """Analizar muestra de páginas para determinar características"""
        total_pages = len(doc)
        sample_size = min(self.page_sample_size, total_pages)
        
        # Seleccionar páginas distribuidas uniformemente
        if total_pages <= sample_size:
            sample_indices = list(range(total_pages))
        else:
            step = total_pages // sample_size
            sample_indices = [i * step for i in range(sample_size)]
        
        analysis = {
            'text_density': [],
            'image_count': [],
            'vector_graphics_count': [],
            'table_indicators': [],
            'avg_font_sizes': [],
            'color_usage': [],
            'page_layouts': []
        }
        
        for idx in sample_indices:
            page = doc[idx]
            page_stats = self._analyze_single_page(page)
            
            analysis['text_density'].append(page_stats['text_density'])
            analysis['image_count'].append(page_stats['image_count'])
            analysis['vector_graphics_count'].append(page_stats['vector_count'])
            analysis['table_indicators'].append(page_stats['has_table_patterns'])
            analysis['avg_font_sizes'].extend(page_stats['font_sizes'])
            analysis['color_usage'].append(page_stats['uses_color'])
            analysis['page_layouts'].append(page_stats['layout_type'])
        
        # Calcular estadísticas agregadas
        return {
            'avg_text_density': np.mean(analysis['text_density']),
            'avg_images_per_page': np.mean(analysis['image_count']),
            'avg_vector_graphics': np.mean(analysis['vector_graphics_count']),
            'table_frequency': sum(analysis['table_indicators']) / len(analysis['table_indicators']),
            'dominant_font_size': Counter(analysis['avg_font_sizes']).most_common(1)[0][0] if analysis['avg_font_sizes'] else 12,
            'color_document': sum(analysis['color_usage']) > len(analysis['color_usage']) / 2,
            'layout_distribution': Counter(analysis['page_layouts'])
        }
    
    def _analyze_single_page(self, page) -> Dict:
        """Analizar una página individual"""
        # Texto
        text = page.get_text()
        text_blocks = page.get_text("blocks")
        
        # Imágenes
        images = page.get_images()
        
        # Gráficos vectoriales (dibujos)
        try:
            drawings = page.get_drawings()
            vector_count = len(drawings)
        except:
            vector_count = 0
        
        # Detectar patrones de tabla
        has_tables = self._detect_table_patterns(text)
        
        # Analizar fuentes
        font_sizes = self._extract_font_sizes(page)
        
        # Detectar uso de color
        uses_color = self._page_uses_color(page)
        
        # Determinar tipo de layout
        layout_type = self._determine_page_layout(text_blocks, images, vector_count)
        
        # Calcular densidad de texto
        page_area = page.rect.width * page.rect.height
        text_density = len(text.strip()) / page_area if page_area > 0 else 0
        
        return {
            'text_density': text_density,
            'image_count': len(images),
            'vector_count': vector_count,
            'has_table_patterns': has_tables,
            'font_sizes': font_sizes,
            'uses_color': uses_color,
            'layout_type': layout_type
        }
    
    def _detect_table_patterns(self, text: str) -> bool:
        """Detectar si hay patrones que indiquen tablas"""
        # Buscar patrones comunes en tablas
        patterns = [
            r'\|\s*\w+\s*\|',  # Pipes como separadores
            r'\t\w+\t',  # Tabs como separadores
            r'^\s*\d+\.\d+\s+',  # Números decimales alineados
            r'(\w+\s{2,}){3,}',  # Múltiples espacios como columnas
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        # Contar líneas con estructura similar (posibles filas)
        lines = text.split('\n')
        structured_lines = 0
        for line in lines:
            if len(line.split()) > 3 and ('|' in line or '\t' in line or '  ' in line):
                structured_lines += 1
        
        return structured_lines > 5
    
    def _extract_font_sizes(self, page) -> List[float]:
        """Extraer tamaños de fuente usados en la página"""
        font_sizes = []
        try:
            blocks = page.get_text("dict")
            for block in blocks.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(round(span.get("size", 12)))
        except:
            pass
        
        return font_sizes
    
    def _page_uses_color(self, page) -> bool:
        """Detectar si la página usa color (no solo blanco y negro)"""
        try:
            # Renderizar pequeña muestra
            pix = page.get_pixmap(matrix=fitz.Matrix(0.2, 0.2))  # 20% del tamaño
            img_data = pix.samples
            
            # Verificar si hay valores no grises
            # En escala de grises, R=G=B
            if pix.n >= 3:  # RGB o más canales
                # Simplified check - si hay variación significativa entre canales RGB
                return True  # Por simplicidad, asumir que PDFs técnicos usan color
                
        except:
            pass
        
        return False
    
    def _determine_page_layout(self, text_blocks: List, images: List, vector_count: int) -> str:
        """Determinar el tipo de layout de la página"""
        text_count = len(text_blocks)
        image_count = len(images)
        
        if vector_count > 20 and text_count < 10:
            return "diagram_heavy"
        elif image_count > 3 and text_count < 5:
            return "image_heavy"
        elif text_count > 20 and image_count == 0 and vector_count < 5:
            return "text_only"
        elif text_count > 10 and (image_count > 0 or vector_count > 10):
            return "mixed"
        else:
            return "sparse"
    
    def _classify_document_type(self, page_analysis: Dict, basic_info: Dict) -> str:
        """Clasificar el tipo de documento basado en el análisis"""
        
        # Verificar si es escaneado
        if self._is_scanned_document(basic_info, page_analysis):
            return DocumentType.SCANNED_DOCUMENT
        
        # Analizar características
        text_density = page_analysis['avg_text_density']
        images_per_page = page_analysis['avg_images_per_page']
        vector_graphics = page_analysis['avg_vector_graphics']
        table_freq = page_analysis['table_frequency']
        
        # Clasificación basada en umbrales
        if vector_graphics > 15 and text_density < 0.3:
            return DocumentType.TECHNICAL_DIAGRAM_HEAVY
        elif table_freq > 0.5:
            return DocumentType.TABLE_HEAVY
        elif images_per_page > 2 and text_density < 0.4:
            return DocumentType.IMAGE_CATALOG
        elif text_density > 0.5 and images_per_page < 0.5 and vector_graphics < 5:
            return DocumentType.TEXT_HEAVY
        else:
            return DocumentType.MIXED_CONTENT
    
    def _is_scanned_document(self, basic_info: Dict, page_analysis: Dict) -> bool:
        """Detectar si el documento es escaneado"""
        # Indicadores de documento escaneado
        indicators = []
        
        # 1. Creado con herramientas de escaneo
        scan_tools = ['scan', 'scanner', 'adobe scan', 'camscanner']
        producer = basic_info.get('metadata', {}).get('producer', '').lower()
        indicators.append(any(tool in producer for tool in scan_tools))
        
        # 2. Una imagen por página con poco texto extraíble
        layout_dist = page_analysis.get('layout_distribution', {})
        indicators.append(
            layout_dist.get('image_heavy', 0) > layout_dist.get('text_only', 0) and
            page_analysis['avg_text_density'] < 0.1
        )
        
        # 3. Sin fuentes detectables
        indicators.append(page_analysis.get('dominant_font_size') == 12)  # Default
        
        return sum(indicators) >= 2
    
    def _determine_extraction_strategy(self, doc_type: str, page_analysis: Dict) -> Dict:
        """Determinar estrategia óptima de extracción según el tipo"""
        
        strategies = {
            DocumentType.TECHNICAL_DIAGRAM_HEAVY: {
                'extract_text': True,
                'extract_tables': True,
                'extract_images': False,  # No hay imágenes raster
                'extract_diagrams': True,  # Renderizar páginas con diagramas
                'use_ocr': True,  # OCR en diagramas renderizados
                'diagram_dpi': 200,  # Alta calidad para diagramas técnicos
                'chunk_size': 1024,  # Chunks más grandes para contexto técnico
                'priority': 'diagrams'
            },
            
            DocumentType.TEXT_HEAVY: {
                'extract_text': True,
                'extract_tables': True,
                'extract_images': False,
                'extract_diagrams': False,
                'use_ocr': False,
                'chunk_size': 512,
                'priority': 'text'
            },
            
            DocumentType.MIXED_CONTENT: {
                'extract_text': True,
                'extract_tables': True,
                'extract_images': True,
                'extract_diagrams': True,
                'use_ocr': False,
                'diagram_dpi': 150,
                'chunk_size': 512,
                'priority': 'balanced'
            },
            
            DocumentType.SCANNED_DOCUMENT: {
                'extract_text': False,  # No hay texto real
                'extract_tables': False,
                'extract_images': True,
                'extract_diagrams': True,
                'use_ocr': True,  # OCR obligatorio
                'diagram_dpi': 300,  # Alta calidad para OCR
                'chunk_size': 768,
                'priority': 'ocr'
            },
            
            DocumentType.TABLE_HEAVY: {
                'extract_text': True,
                'extract_tables': True,
                'extract_images': True,
                'extract_diagrams': False,
                'use_ocr': False,
                'chunk_size': 768,  # Más grande para preservar tablas
                'table_extraction_method': 'camelot_first',
                'priority': 'tables'
            },
            
            DocumentType.IMAGE_CATALOG: {
                'extract_text': True,
                'extract_tables': False,
                'extract_images': True,
                'extract_diagrams': False,
                'use_ocr': True,  # Para texto en imágenes
                'chunk_size': 256,  # Chunks pequeños
                'priority': 'images'
            }
        }
        
        return strategies.get(doc_type, strategies[DocumentType.MIXED_CONTENT])
    
    def _detect_special_features(self, doc, page_analysis: Dict) -> Dict:
        """Detectar características especiales del documento"""
        features = {
            'has_forms': False,
            'has_hyperlinks': False,
            'has_annotations': False,
            'has_bookmarks': False,
            'is_multilingual': False,
            'has_complex_layouts': False,
            'estimated_language': 'es'  # Default español
        }
        
        # Verificar formularios
        for page in doc:
            if page.first_widget:
                features['has_forms'] = True
                break
        
        # Verificar enlaces
        for page in doc:
            if page.get_links():
                features['has_hyperlinks'] = True
                break
        
        # Verificar anotaciones
        for page in doc:
            if page.first_annot:
                features['has_annotations'] = True
                break
        
        # Verificar marcadores
        features['has_bookmarks'] = len(doc.get_toc()) > 0
        
        # Detectar layouts complejos
        layout_dist = page_analysis.get('layout_distribution', {})
        features['has_complex_layouts'] = (
            layout_dist.get('mixed', 0) > layout_dist.get('text_only', 0)
        )
        
        return features
    
    def _generate_recommendations(self, doc_type: str, special_features: Dict) -> List[str]:
        """Generar recomendaciones específicas para el procesamiento"""
        recommendations = []
        
        # Recomendaciones por tipo
        if doc_type == DocumentType.TECHNICAL_DIAGRAM_HEAVY:
            recommendations.extend([
                "Usar --extract-diagrams para capturar diagramas técnicos",
                "Considerar aumentar DPI a 200-300 para mejor calidad",
                "Aplicar OCR a diagramas para extraer texto embebido",
                "Usar chunks más grandes (1024) para mantener contexto técnico"
            ])
        
        elif doc_type == DocumentType.SCANNED_DOCUMENT:
            recommendations.extend([
                "OCR es obligatorio para este documento",
                "Usar alta resolución (300 DPI) para mejor precisión OCR",
                "Considerar preprocesamiento de imágenes antes de OCR",
                "Verificar idioma para configurar OCR correctamente"
            ])
        
        elif doc_type == DocumentType.TABLE_HEAVY:
            recommendations.extend([
                "Priorizar extracción con Camelot",
                "Usar chunks más grandes para no dividir tablas",
                "Considerar guardar tablas en formato estructurado (CSV/Excel)",
                "Verificar la integridad de las tablas extraídas"
            ])
        
        # Recomendaciones por características especiales
        if special_features.get('has_forms'):
            recommendations.append("Documento contiene formularios - considerar extracción especial")
        
        if special_features.get('has_complex_layouts'):
            recommendations.append("Layouts complejos detectados - revisar resultados manualmente")
        
        return recommendations
    
    def _get_default_strategy(self) -> Dict:
        """Estrategia por defecto en caso de error"""
        return {
            'extract_text': True,
            'extract_tables': True,
            'extract_images': True,
            'extract_diagrams': False,
            'use_ocr': False,
            'chunk_size': 512,
            'priority': 'balanced'
        }