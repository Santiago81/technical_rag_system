from typing import List, Dict, Optional, Tuple, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TextProcessor:
    """Procesador de texto para preparar chunks para embeddings"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Splitter principal para texto general
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Splitter especial para tablas (chunks más grandes)
        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 2,  # Tablas necesitan más contexto
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "|", ",", " "]
        )
        
        # Patrones de limpieza
        self.cleaning_patterns = {
            'multiple_spaces': (r'\s+', ' '),
            'special_chars': (r'[^\w\s\-\.,:;!?()áéíóúñÁÉÍÓÚÑ°%/]', ''),
            'multiple_newlines': (r'\n{3,}', '\n\n'),
            'trailing_spaces': (r' +\n', '\n')
        }
    
    def process_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Procesar texto y crear chunks con metadatos"""
        # Detectar tipo de contenido para procesamiento especial
        content_type = metadata.get('content_type', 'text')
        
        # Limpiar texto según el tipo
        clean_text = self.clean_text(text, content_type)
        
        # Si no hay contenido significativo, retornar vacío
        if not clean_text or len(clean_text.strip()) < 10:
            logger.debug(f"Texto muy corto o vacío, omitiendo: {metadata}")
            return []
        
        # Dividir en chunks
        chunks = self.text_splitter.split_text(clean_text)
        
        # Crear documentos con metadatos enriquecidos
        documents = []
        for i, chunk in enumerate(chunks):
            # Calcular posición relativa del chunk
            position_ratio = i / len(chunks) if len(chunks) > 1 else 0.5
            
            doc = {
                'content': chunk,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_position': self._get_position_label(position_ratio),
                    'chunk_size': len(chunk),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            # Agregar información contextual
            doc['metadata'].update(self._extract_chunk_features(chunk))
            
            documents.append(doc)
        
        return documents
    
    def process_table_text(self, table_text: str, metadata: Dict) -> List[Dict]:
        """Procesar texto de tablas de manera especial"""
        # Las tablas requieren procesamiento diferente
        metadata['content_type'] = 'table'
        metadata['is_structured'] = True
        
        # Si es un DataFrame, convertirlo a texto estructurado
        if isinstance(table_text, pd.DataFrame):
            table_text = self._dataframe_to_structured_text(table_text)
        
        # Limpiar pero preservar estructura
        clean_text = self._clean_table_text(table_text)
        
        # Determinar estrategia de chunking
        if len(clean_text) < self.chunk_size * 1.5:
            # Tabla pequeña: un solo chunk
            return [{
                'content': clean_text,
                'metadata': {
                    **metadata,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'table_complete': True,
                    'processed_at': datetime.now().isoformat()
                }
            }]
        else:
            # Tabla grande: dividir inteligentemente
            return self._chunk_large_table(clean_text, metadata)
    
    def process_image_context(self, image_metadata: Dict, ocr_text: Optional[str] = None) -> List[Dict]:
        """Procesar contexto de imagen para búsqueda"""
        # Crear descripción textual de la imagen
        content_parts = []
        
        # Información básica
        content_parts.append(f"Imagen en página {image_metadata.get('page_number', 'desconocida')}")
        
        # Agregar tipo si está disponible
        if image_metadata.get('content_type') == 'diagram':
            content_parts.append("Diagrama técnico")
        elif image_metadata.get('is_full_page'):
            content_parts.append("Imagen de página completa")
        
        # Agregar dimensiones si son relevantes
        width = image_metadata.get('width', 0)
        height = image_metadata.get('height', 0)
        if width > 0 and height > 0:
            content_parts.append(f"Dimensiones: {width}x{height}")
        
        # Agregar texto OCR si está disponible
        if ocr_text:
            clean_ocr = self.clean_text(ocr_text, 'ocr')
            if clean_ocr:
                content_parts.append(f"Texto detectado: {clean_ocr[:200]}")
        
        # Combinar todo
        content = " | ".join(content_parts)
        
        # Crear documento
        return [{
            'content': content,
            'metadata': {
                **image_metadata,
                'content_type': 'image_context',
                'has_ocr': bool(ocr_text),
                'processed_at': datetime.now().isoformat()
            }
        }]
    
    def process_mixed_content(self, text: str, tables: List[Dict], 
                            images: List[Dict], metadata: Dict) -> List[Dict]:
        """Procesar contenido mixto de una página (texto + tablas + imágenes)"""
        all_documents = []
        
        # Procesar texto principal
        if text.strip():
            text_docs = self.process_text(text, {**metadata, 'content_type': 'text'})
            all_documents.extend(text_docs)
        
        # Procesar tablas
        for i, table in enumerate(tables):
            table_metadata = {
                **metadata,
                'content_type': 'table',
                'table_index': i
            }
            
            if 'table_text' in table:
                table_docs = self.process_table_text(table['table_text'], table_metadata)
                all_documents.extend(table_docs)
        
        # Procesar referencias de imágenes
        for i, image in enumerate(images):
            image_metadata = {
                **metadata,
                **image,
                'content_type': 'image_context',
                'image_index': i
            }
            
            ocr_text = image.get('ocr_text')
            image_docs = self.process_image_context(image_metadata, ocr_text)
            all_documents.extend(image_docs)
        
        # Agregar relaciones entre documentos
        self._add_document_relationships(all_documents)
        
        return all_documents
    
    def clean_text(self, text: str, content_type: str = 'text') -> str:
        """Limpiar y normalizar texto según su tipo"""
        if not text:
            return ""
        
        # Aplicar limpieza básica
        for pattern_name, (pattern, replacement) in self.cleaning_patterns.items():
            # Skip algunos patrones para ciertos tipos
            if content_type == 'table' and pattern_name in ['special_chars']:
                continue
            if content_type == 'ocr' and pattern_name in ['multiple_newlines']:
                continue
                
            text = re.sub(pattern, replacement, text)
        
        # Limpieza específica por tipo
        if content_type == 'ocr':
            # OCR puede tener caracteres extraños
            text = self._clean_ocr_artifacts(text)
        elif content_type == 'table':
            # Preservar estructura de tabla
            text = self._normalize_table_structure(text)
        
        return text.strip()
    
    def _clean_table_text(self, table_text: str) -> str:
        """Limpiar texto de tabla preservando estructura"""
        # Normalizar separadores
        table_text = re.sub(r'\|\s+\|', '|', table_text)
        table_text = re.sub(r'\s*\|\s*', ' | ', table_text)
        
        # Eliminar líneas vacías pero preservar estructura
        lines = table_text.split('\n')
        cleaned_lines = [line for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines)
    
    def _chunk_large_table(self, table_text: str, metadata: Dict) -> List[Dict]:
        """Dividir tabla grande en chunks lógicos"""
        lines = table_text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Intentar mantener header en cada chunk si es posible
        header = lines[0] if lines else ""
        header_size = len(header)
        
        for i, line in enumerate(lines):
            line_size = len(line)
            
            # Si agregar esta línea excede el tamaño, crear nuevo chunk
            if current_size + line_size > self.chunk_size and current_chunk:
                # Agregar header si no es el primer chunk
                if i > 0 and header and not current_chunk[0] == header:
                    chunk_text = header + '\n' + '\n'.join(current_chunk)
                else:
                    chunk_text = '\n'.join(current_chunk)
                
                chunks.append(chunk_text)
                current_chunk = []
                current_size = header_size if i > 0 else 0
            
            current_chunk.append(line)
            current_size += line_size
        
        # Agregar último chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Crear documentos
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'table_complete': False,
                    'table_part': f"{i+1}/{len(chunks)}",
                    'processed_at': datetime.now().isoformat()
                }
            }
            documents.append(doc)
        
        return documents
    
    def _dataframe_to_structured_text(self, df: pd.DataFrame) -> str:
        """Convertir DataFrame a texto estructurado optimizado para embeddings"""
        # Incluir información sobre columnas
        text_parts = [f"Tabla con {len(df)} filas y {len(df.columns)} columnas"]
        text_parts.append(f"Columnas: {', '.join(df.columns)}")
        
        # Agregar estadísticas básicas si son numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text_parts.append("\nEstadísticas:")
            for col in numeric_cols[:3]:  # Limitar a 3 columnas
                text_parts.append(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, promedio={df[col].mean():.2f}")
        
        # Agregar primeras filas como muestra
        text_parts.append("\nDatos:")
        sample_size = min(10, len(df))
        for idx, row in df.head(sample_size).iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            text_parts.append(row_text)
        
        if len(df) > sample_size:
            text_parts.append(f"... y {len(df) - sample_size} filas más")
        
        return "\n".join(text_parts)
    
    def _extract_chunk_features(self, chunk: str) -> Dict:
        """Extraer características del chunk para mejorar búsqueda"""
        features = {}
        
        # Detectar si contiene números/datos
        numbers = re.findall(r'\d+\.?\d*', chunk)
        features['has_numbers'] = len(numbers) > 0
        features['number_count'] = len(numbers)
        
        # Detectar si parece ser una lista
        features['is_list'] = bool(re.search(r'^\s*[-•*]\s+', chunk, re.MULTILINE))
        
        # Detectar si contiene instrucciones (verbos imperativos)
        instruction_patterns = r'\b(conectar|desconectar|verificar|revisar|ajustar|instalar|remover|check|install|remove)\b'
        features['has_instructions'] = bool(re.search(instruction_patterns, chunk, re.IGNORECASE))
        
        # Detectar advertencias o precauciones
        warning_patterns = r'\b(advertencia|precaución|peligro|atención|importante|warning|caution|danger)\b'
        features['has_warnings'] = bool(re.search(warning_patterns, chunk, re.IGNORECASE))
        
        return features
    
    def _get_position_label(self, position_ratio: float) -> str:
        """Obtener etiqueta de posición del chunk"""
        if position_ratio < 0.2:
            return "inicio"
        elif position_ratio < 0.8:
            return "medio"
        else:
            return "final"
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        """Limpiar artefactos comunes de OCR"""
        # Corregir espacios entre letras
        text = re.sub(r'(\w)\s(\w)', r'\1\2', text)
        
        # Corregir caracteres comunes mal reconocidos
        replacements = {
            '|': 'I',  # Pipe confundido con I
            '0': 'O',  # Cero confundido con O (contexto dependiente)
            '§': 'S',  # Símbolo confundido con S
        }
        
        # Aplicar reemplazos con cuidado
        for old, new in replacements.items():
            # Solo en palabras, no en números
            text = re.sub(rf'\b{old}(?=[a-zA-Z])', new, text)
        
        return text
    
    def _normalize_table_structure(self, text: str) -> str:
        """Normalizar estructura de tabla para mejor procesamiento"""
        # Asegurar separadores consistentes
        text = re.sub(r'\t+', ' | ', text)  # Tabs a pipes
        text = re.sub(r' {2,}', ' | ', text)  # Múltiples espacios a pipes
        
        return text
    
    def _add_document_relationships(self, documents: List[Dict]):
        """Agregar información de relaciones entre documentos"""
        for i, doc in enumerate(documents):
            relationships = {
                'prev_chunk': documents[i-1]['metadata'].get('chunk_index') if i > 0 else None,
                'next_chunk': documents[i+1]['metadata'].get('chunk_index') if i < len(documents)-1 else None,
                'same_page_chunks': len(documents)
            }
            doc['metadata']['relationships'] = relationships
    
    def merge_chunks(self, chunks: List[Dict], max_size: Optional[int] = None) -> List[Dict]:
        """Fusionar chunks pequeños cuando sea apropiado"""
        if not chunks or len(chunks) <= 1:
            return chunks
        
        max_size = max_size or self.chunk_size
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            current_size = len(current['content'])
            next_size = len(next_chunk['content'])
            
            # Verificar si se pueden fusionar
            if (current_size + next_size < max_size and 
                current['metadata'].get('content_type') == next_chunk['metadata'].get('content_type')):
                
                # Fusionar contenido
                current['content'] += '\n' + next_chunk['content']
                
                # Actualizar metadatos
                current['metadata']['merged'] = True
                current['metadata']['original_chunks'] = (
                    current['metadata'].get('original_chunks', [current['metadata']['chunk_index']]) +
                    [next_chunk['metadata']['chunk_index']]
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged