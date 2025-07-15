import fitz
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from pdf2image import convert_from_path
import cv2

logger = logging.getLogger(__name__)

class EnhancedImageExtractor:
    """Extractor mejorado para imágenes y diagramas técnicos de PDFs"""
    
    def __init__(self, min_size: Tuple[int, int] = (100, 100), 
                 output_format: str = 'png',
                 diagram_dpi: int = 200):
        self.min_width, self.min_height = min_size
        self.output_format = output_format
        self.diagram_dpi = diagram_dpi
        self.vector_threshold = 0.3  # Umbral para detectar páginas con diagramas vectoriales
    
    def extract_all_content(self, pdf_path: str, output_dir: Path) -> Dict:
        """Extraer imágenes raster y diagramas vectoriales de un PDF"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        
        # Crear subdirectorios
        images_dir = output_dir / 'images'
        diagrams_dir = output_dir / 'diagrams'
        images_dir.mkdir(parents=True, exist_ok=True)
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'manual_name': manual_name,
            'raster_images': [],
            'vector_diagrams': [],
            'mixed_pages': [],
            'statistics': {}
        }
        
        doc = fitz.open(pdf_path)
        
        # Primera pasada: analizar cada página
        page_analysis = self._analyze_document_structure(doc)
        
        # Segunda pasada: extraer contenido según tipo
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_info = page_analysis[page_num]
            
            if page_info['type'] == 'vector_heavy':
                # Renderizar página completa como diagrama
                diagram_data = self._extract_page_as_diagram(
                    page, page_num + 1, manual_name, diagrams_dir
                )
                if diagram_data:
                    results['vector_diagrams'].append(diagram_data)
            
            elif page_info['type'] == 'raster_heavy':
                # Extraer imágenes raster normalmente
                images = self._extract_raster_images(
                    page, page_num + 1, manual_name, images_dir
                )
                results['raster_images'].extend(images)
            
            else:  # mixed
                # Extraer ambos tipos
                images = self._extract_raster_images(
                    page, page_num + 1, manual_name, images_dir
                )
                results['raster_images'].extend(images)
                
                # Si hay suficiente contenido vectorial, también renderizar
                if page_info['vector_ratio'] > 0.2:
                    diagram_data = self._extract_page_as_diagram(
                        page, page_num + 1, manual_name, diagrams_dir
                    )
                    if diagram_data:
                        results['mixed_pages'].append(diagram_data)
        
        doc.close()
        
        # Estadísticas
        results['statistics'] = {
            'total_pages': len(page_analysis),
            'vector_pages': sum(1 for p in page_analysis.values() if p['type'] == 'vector_heavy'),
            'raster_pages': sum(1 for p in page_analysis.values() if p['type'] == 'raster_heavy'),
            'mixed_pages': sum(1 for p in page_analysis.values() if p['type'] == 'mixed'),
            'total_raster_images': len(results['raster_images']),
            'total_vector_diagrams': len(results['vector_diagrams'])
        }
        
        logger.info(f"Extracción completada: {results['statistics']}")
        return results
    
    def _analyze_document_structure(self, doc) -> Dict[int, Dict]:
        """Analizar la estructura del documento para determinar tipos de contenido"""
        analysis = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Contar elementos
            text_instances = len(page.get_text("words"))
            image_list = page.get_images()
            drawings = page.get_drawings()
            
            # Calcular ratios
            total_elements = text_instances + len(image_list) + len(drawings)
            
            if total_elements > 0:
                text_ratio = text_instances / total_elements
                image_ratio = len(image_list) / total_elements
                vector_ratio = len(drawings) / total_elements
            else:
                text_ratio = image_ratio = vector_ratio = 0
            
            # Determinar tipo de página
            if vector_ratio > self.vector_threshold and image_ratio < 0.1:
                page_type = 'vector_heavy'
            elif image_ratio > 0.5 and vector_ratio < 0.1:
                page_type = 'raster_heavy'
            elif text_ratio > 0.8:
                page_type = 'text_only'
            else:
                page_type = 'mixed'
            
            analysis[page_num] = {
                'type': page_type,
                'text_ratio': text_ratio,
                'image_ratio': image_ratio,
                'vector_ratio': vector_ratio,
                'total_elements': total_elements,
                'has_complex_graphics': len(drawings) > 50
            }
        
        return analysis
    
    def _extract_raster_images(self, page, page_num: int,
                              manual_name: str, output_dir: Path) -> List[Dict]:
        """Extraer imágenes raster de una página"""
        images_metadata = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.width >= self.min_width and pix.height >= self.min_height:
                    # Manejar espacios de color
                    if pix.n - pix.alpha >= 4:  # CMYK
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix = None
                        pix = pix_rgb
                    
                    # Guardar imagen
                    img_filename = f"{manual_name}_p{page_num}_img{img_index}.{self.output_format}"
                    img_path = output_dir / img_filename
                    
                    img_data = pix.tobytes(self.output_format)
                    img_pil = Image.open(io.BytesIO(img_data))
                    
                    # Determinar si es una foto o diagrama
                    image_type = self._classify_image_content(img_pil)
                    
                    img_pil.save(img_path, optimize=True, quality=95 if self.output_format == 'jpeg' else None)
                    
                    metadata = {
                        'manual_name': manual_name,
                        'page_number': page_num,
                        'image_index': img_index,
                        'image_path': str(img_path),
                        'image_filename': img_filename,
                        'width': pix.width,
                        'height': pix.height,
                        'content_type': image_type,
                        'extraction_method': 'raster',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    images_metadata.append(metadata)
                
                pix = None
                
            except Exception as e:
                logger.error(f"Error extrayendo imagen {img_index} de página {page_num}: {e}")
        
        return images_metadata
    
    def _extract_page_as_diagram(self, page, page_num: int,
                                manual_name: str, output_dir: Path) -> Optional[Dict]:
        """Renderizar página completa como diagrama de alta calidad"""
        try:
            # Renderizar con alta resolución
            mat = fitz.Matrix(self.diagram_dpi/72.0, self.diagram_dpi/72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convertir a PIL
            img_data = pix.tobytes("png")
            img_pil = Image.open(io.BytesIO(img_data))
            
            # Aplicar mejoras para diagramas técnicos
            img_enhanced = self._enhance_technical_diagram(img_pil)
            
            # Guardar
            diagram_filename = f"{manual_name}_p{page_num}_diagram.png"
            diagram_path = output_dir / diagram_filename
            img_enhanced.save(diagram_path, "PNG", optimize=True)
            
            # Extraer texto de la página para contexto
            page_text = page.get_text()
            
            metadata = {
                'manual_name': manual_name,
                'page_number': page_num,
                'image_path': str(diagram_path),
                'image_filename': diagram_filename,
                'width': img_enhanced.width,
                'height': img_enhanced.height,
                'content_type': 'technical_diagram',
                'extraction_method': 'page_render',
                'render_dpi': self.diagram_dpi,
                'page_text_preview': page_text[:200] if page_text else "",
                'has_vector_graphics': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # OCR opcional para diagramas
            ocr_text = self._extract_text_from_diagram(img_enhanced)
            if ocr_text:
                metadata['ocr_text'] = ocr_text
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error renderizando página {page_num} como diagrama: {e}")
            return None
    
    def _enhance_technical_diagram(self, image: Image.Image) -> Image.Image:
        """Mejorar la calidad de diagramas técnicos"""
        # Convertir a numpy array
        img_array = np.array(image)
        
        # Convertir a escala de grises si es necesario
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Aplicar umbralización adaptativa para mejorar líneas
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Aplicar denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Convertir de vuelta a PIL
        return Image.fromarray(denoised)
    
    def _classify_image_content(self, image: Image.Image) -> str:
        """Clasificar el tipo de contenido de la imagen"""
        # Convertir a escala de grises
        gray = image.convert('L')
        pixels = np.array(gray)
        
        # Calcular estadísticas
        unique_colors = len(np.unique(pixels))
        std_dev = np.std(pixels)
        
        # Detectar características
        edges = cv2.Canny(pixels, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Clasificar
        if unique_colors < 10 and edge_ratio > 0.1:
            return 'line_diagram'
        elif std_dev > 50 and unique_colors > 100:
            return 'photograph'
        elif edge_ratio > 0.05 and unique_colors < 50:
            return 'technical_drawing'
        else:
            return 'mixed_graphic'
    
    def _extract_text_from_diagram(self, image: Image.Image) -> Optional[str]:
        """Extraer texto de diagramas usando OCR optimizado"""
        try:
            import pytesseract
            
            # Configuración específica para diagramas técnicos
            custom_config = r'--oem 3 --psm 11'
            
            # Preprocesar imagen para mejor OCR
            img_array = np.array(image)
            
            # Binarización
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(
                Image.fromarray(binary), 
                config=custom_config,
                lang='spa+eng'  # Español e inglés
            )
            
            # Limpiar y filtrar
            text = ' '.join(text.split())
            
            return text if len(text) > 10 else None
            
        except Exception as e:
            logger.debug(f"Error en OCR: {e}")
            return None
    
    def extract_specific_diagrams(self, pdf_path: str, output_dir: Path, 
                                 page_numbers: List[int] = None) -> List[Dict]:
        """Extraer diagramas de páginas específicas"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        diagrams = []
        
        doc = fitz.open(pdf_path)
        
        pages_to_process = page_numbers if page_numbers else range(len(doc))
        
        for page_num in pages_to_process:
            if page_num < len(doc):
                page = doc[page_num]
                diagram = self._extract_page_as_diagram(
                    page, page_num + 1, manual_name, output_dir
                )
                if diagram:
                    diagrams.append(diagram)
        
        doc.close()
        return diagrams
    
    def extract_diagrams_as_images(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        """Extraer páginas completas como imágenes (útil para diagramas complejos)"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        diagrams_dir = output_dir / 'diagrams'
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        
        # Usar el método extract_all_content que ya maneja la lógica
        results = self.extract_all_content(pdf_path, output_dir)
        
        # Combinar todos los diagramas extraídos
        all_diagrams = []
        all_diagrams.extend(results.get('vector_diagrams', []))
        all_diagrams.extend(results.get('mixed_pages', []))
        
        return all_diagrams


# Script para análisis rápido
def analyze_pdf_content(pdf_path: str) -> Dict:
    """Analizar un PDF y determinar la mejor estrategia de extracción"""
    doc = fitz.open(pdf_path)
    
    analysis = {
        'filename': Path(pdf_path).name,
        'total_pages': len(doc),
        'content_summary': {
            'text_heavy_pages': 0,
            'image_heavy_pages': 0,
            'vector_heavy_pages': 0,
            'mixed_pages': 0
        },
        'recommendations': []
    }
    
    extractor = EnhancedImageExtractor()
    page_analysis = extractor._analyze_document_structure(doc)
    
    for page_info in page_analysis.values():
        analysis['content_summary'][f"{page_info['type']}_pages"] += 1
    
    # Generar recomendaciones
    if analysis['content_summary']['vector_heavy_pages'] > 5:
        analysis['recommendations'].append(
            "Alto contenido vectorial detectado. Usar renderizado de páginas completas."
        )
    
    if analysis['content_summary']['image_heavy_pages'] > 10:
        analysis['recommendations'].append(
            "Muchas imágenes raster. Extracción estándar funcionará bien."
        )
    
    doc.close()
    return analysis