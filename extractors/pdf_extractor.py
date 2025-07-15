import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extractor principal para texto de manuales PDF"""
    
    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config.PROCESSED_DIR)
        self.ensure_directories()
    
    def ensure_directories(self):
        """Crear directorios necesarios"""
        for subdir in ['texts', 'metadata']:
            (self.processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def extract_all_content(self, pdf_path: str) -> Dict:
        """Extraer todo el contenido de texto de un PDF"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        
        logger.info(f"Procesando manual: {manual_name}")
        
        # Inicializar resultados
        results = {
            'manual_name': manual_name,
            'pages': [],
            'total_pages': 0,
            'extraction_date': datetime.now().isoformat()
        }
        
        # Abrir PDF con PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            results['total_pages'] = len(doc)
            
            # Extraer texto de cada página
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self.extract_page_content(page, page_num + 1, manual_name)
                results['pages'].append(page_data)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error procesando PDF {pdf_path}: {e}")
            raise
        
        # Guardar metadatos
        self.save_metadata(results, manual_name)
        
        logger.info(f"Extracción completada: {manual_name} - {results['total_pages']} páginas")
        
        return results
    
    def extract_page_content(self, page, page_num: int, manual_name: str) -> Dict:
        """Extraer contenido de texto de una página"""
        # Extraer texto
        text = page.get_text()
        
        # Detectar secciones y capítulos (personalizar según formato de manuales)
        section = self.detect_section(text)
        chapter = self.detect_chapter(text)
        
        # Contar caracteres y palabras para estadísticas
        char_count = len(text)
        word_count = len(text.split())
        
        page_data = {
            'page_number': page_num,
            'text': text,
            'section': section,
            'chapter': chapter,
            'char_count': char_count,
            'word_count': word_count,
            'has_content': bool(text.strip()),
            'metadata': {
                'manual_name': manual_name,
                'page_number': page_num,
                'content_type': 'text',
                'section': section,
                'chapter': chapter,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Guardar texto solo si tiene contenido
        if page_data['has_content']:
            text_file = self.processed_dir / 'texts' / f"{manual_name}_page_{page_num}.txt"
            text_file.write_text(text, encoding='utf-8')
            page_data['text_file'] = str(text_file)
        else:
            logger.debug(f"Página {page_num} sin contenido de texto")
        
        return page_data
    
    def detect_section(self, text: str) -> Optional[str]:
        """Detectar sección del texto (personalizar según formato)"""
        import re
        
        # Patrones comunes para secciones
        section_patterns = [
            r'(?:Section|SECTION|Sección|SECCIÓN)\s*(\d+\.?\d*)',
            r'(?:^\d+\.)\s+([A-Z][^.]+)',  # "1. TÍTULO DE SECCIÓN"
            r'(?:^[IVX]+\.)\s+([A-Z][^.]+)'  # "IV. TÍTULO DE SECCIÓN"
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(0).strip()
        
        return None
    
    def detect_chapter(self, text: str) -> Optional[str]:
        """Detectar capítulo del texto"""
        import re
        
        # Patrones comunes para capítulos
        chapter_patterns = [
            r'(?:Chapter|CHAPTER|Capítulo|CAPÍTULO)\s*(\d+\.?\d*)',
            r'(?:Part|PART|Parte|PARTE)\s*(\d+\.?\d*)',
            r'(?:Unit|UNIT|Unidad|UNIDAD)\s*(\d+\.?\d*)'
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(0).strip()
        
        return None
    
    def save_metadata(self, results: Dict, manual_name: str):
        """Guardar todos los metadatos en un archivo JSON"""
        metadata_path = self.processed_dir / 'metadata' / f"{manual_name}_metadata.json"
        
        # Agregar estadísticas generales
        stats = self.calculate_statistics(results)
        results['statistics'] = stats
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadatos guardados en: {metadata_path}")
    
    def calculate_statistics(self, results: Dict) -> Dict:
        """Calcular estadísticas del documento"""
        total_chars = sum(page['char_count'] for page in results['pages'])
        total_words = sum(page['word_count'] for page in results['pages'])
        pages_with_content = sum(1 for page in results['pages'] if page['has_content'])
        
        # Contar secciones y capítulos únicos
        sections = set(page['section'] for page in results['pages'] if page['section'])
        chapters = set(page['chapter'] for page in results['pages'] if page['chapter'])
        
        return {
            'total_characters': total_chars,
            'total_words': total_words,
            'pages_with_content': pages_with_content,
            'empty_pages': results['total_pages'] - pages_with_content,
            'unique_sections': len(sections),
            'unique_chapters': len(chapters),
            'sections_list': sorted(list(sections)),
            'chapters_list': sorted(list(chapters)),
            'average_words_per_page': total_words / pages_with_content if pages_with_content > 0 else 0
        }
    
    def extract_text_only(self, pdf_path: str) -> str:
        """Método auxiliar para extraer solo el texto completo del PDF"""
        pdf_path = Path(pdf_path)
        full_text = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text.append(f"--- Página {page_num + 1} ---\n{text}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extrayendo texto de {pdf_path}: {e}")
            raise
        
        return "\n\n".join(full_text)
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """Obtener información básica del PDF sin extraer todo el contenido"""
        pdf_path = Path(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            
            info = {
                'filename': pdf_path.name,
                'total_pages': len(doc),
                'metadata': doc.metadata,
                'is_encrypted': doc.is_encrypted,
                'file_size': pdf_path.stat().st_size
            }
            
            doc.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo información de {pdf_path}: {e}")
            raise