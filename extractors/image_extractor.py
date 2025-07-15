import fitz
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

class ImageExtractor:
    """Extractor especializado para imágenes de PDFs"""
    
    def __init__(self, min_size: Tuple[int, int] = (100, 100), 
                 output_format: str = 'png'):
        self.min_width, self.min_height = min_size
        self.output_format = output_format
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        """Extraer todas las imágenes de un PDF"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        all_images = []
        
        # Asegurar que el directorio de salida existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = self.extract_images_from_page(
                page, page_num + 1, manual_name, output_dir
            )
            all_images.extend(images)
        
        doc.close()
        
        logger.info(f"Extraídas {len(all_images)} imágenes de {pdf_path.name}")
        return all_images
    
    def extract_images_from_page(self, page, page_num: int,
                            manual_name: str, output_dir: Path) -> List[Dict]:
        """Extraer imágenes de una página específica"""
        images_metadata = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Extraer imagen
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Verificar tamaño mínimo primero
                if pix.width >= self.min_width and pix.height >= self.min_height:
                    
                    # Manejar diferentes espacios de color
                    if pix.n - pix.alpha >= 4:  # CMYK u otros espacios de color
                        # Convertir a RGB
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        # Liberar el pixmap original
                        pix = None
                        pix = pix_rgb
                    
                    # Ahora pix está en RGB o Grayscale, seguro para PNG
                    try:
                        # Convertir a bytes
                        if self.output_format.lower() == 'png':
                            img_data = pix.tobytes("png")
                        else:
                            # Para JPEG u otros formatos
                            img_data = pix.tobytes(self.output_format)
                        
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        # Generar nombre de archivo
                        img_filename = f"{manual_name}_p{page_num}_img{img_index}.{self.output_format}"
                        img_path = output_dir / img_filename
                        
                        # Guardar imagen
                        img_pil.save(img_path, optimize=True)
                        
                        # Crear metadata
                        metadata = {
                            'manual_name': manual_name,
                            'page_number': page_num,
                            'image_index': img_index,
                            'image_path': str(img_path),
                            'image_filename': img_filename,
                            'width': pix.width,
                            'height': pix.height,
                            'color_space': self._get_color_space(pix),
                            'file_size': img_path.stat().st_size,
                            'content_type': 'image',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Intentar extraer texto de la imagen si es un diagrama
                        ocr_text = self._extract_text_from_image(img_pil)
                        if ocr_text:
                            metadata['ocr_text'] = ocr_text
                        
                        images_metadata.append(metadata)
                        
                    except Exception as e:
                        logger.warning(f"No se pudo guardar imagen {img_index} de página {page_num}: {e}")
                        # Intentar con JPEG si PNG falla
                        if self.output_format.lower() == 'png':
                            try:
                                img_filename = f"{manual_name}_p{page_num}_img{img_index}.jpg"
                                img_path = output_dir / img_filename
                                img_pil.save(img_path, 'JPEG', optimize=True)
                                
                                metadata = {
                                    'manual_name': manual_name,
                                    'page_number': page_num,
                                    'image_index': img_index,
                                    'image_path': str(img_path),
                                    'image_filename': img_filename,
                                    'width': pix.width,
                                    'height': pix.height,
                                    'color_space': self._get_color_space(pix),
                                    'file_size': img_path.stat().st_size,
                                    'content_type': 'image',
                                    'format_fallback': 'jpeg',
                                    'timestamp': datetime.now().isoformat()
                                }
                                images_metadata.append(metadata)
                                logger.info(f"Imagen guardada como JPEG: {img_filename}")
                            except:
                                logger.error(f"No se pudo guardar imagen {img_index} en ningún formato")
                
                # Liberar memoria
                pix = None
                
            except Exception as e:
                logger.error(f"Error extrayendo imagen {img_index} de página {page_num}: {e}")
        
        return images_metadata
    
    def extract_diagrams_as_images(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        """Extraer páginas completas como imágenes (útil para diagramas complejos)"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        diagrams = []
        
        try:
            # Convertir páginas a imágenes
            images = convert_from_path(pdf_path, dpi=150)
            
            for page_num, image in enumerate(images, 1):
                # Analizar si la página parece ser un diagrama
                if self._is_likely_diagram(image):
                    # Guardar como diagrama
                    diagram_filename = f"{manual_name}_p{page_num}_diagram.{self.output_format}"
                    diagram_path = output_dir / diagram_filename
                    
                    image.save(diagram_path, optimize=True)
                    
                    metadata = {
                        'manual_name': manual_name,
                        'page_number': page_num,
                        'image_path': str(diagram_path),
                        'image_filename': diagram_filename,
                        'width': image.width,
                        'height': image.height,
                        'content_type': 'diagram',
                        'is_full_page': True,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    diagrams.append(metadata)
            
        except Exception as e:
            logger.error(f"Error extrayendo diagramas: {e}")
        
        return diagrams
    
    def _get_color_space(self, pix) -> str:
        """Determinar el espacio de color de la imagen"""
        n = pix.n
        alpha = pix.alpha
        
        # Considerar el canal alpha
        color_channels = n - alpha
        
        if color_channels == 1:
            return "GRAY"
        elif color_channels == 3:
            return "RGB"
        elif color_channels == 4:
            return "CMYK"
        else:
            return f"UNKNOWN_{color_channels}"
    
    def _extract_text_from_image(self, image: Image.Image) -> Optional[str]:
        """Extraer texto de imagen usando OCR (opcional)"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None
        except:
            return None
    
    def _is_likely_diagram(self, image: Image.Image) -> bool:
        """Determinar si una imagen es probablemente un diagrama"""
        # Convertir a escala de grises para análisis
        gray = image.convert('L')
        pixels = np.array(gray)
        
        # Calcular proporción de píxeles no blancos
        non_white_ratio = np.sum(pixels < 250) / pixels.size
        
        # Los diagramas suelen tener menos del 50% de contenido no blanco
        # y dimensiones específicas
        aspect_ratio = image.width / image.height
        
        return (0.1 < non_white_ratio < 0.5 and 
                0.5 < aspect_ratio < 2.0)
