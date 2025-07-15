import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import logging
from datetime import datetime
import unicodedata
import mimetypes

logger = logging.getLogger(__name__)

def generate_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """
    Generar ID único para un documento basado en contenido y metadatos
    """
    # Crear string único combinando contenido y metadatos clave
    unique_string = f"{metadata.get('manual_name', '')}_" \
                   f"{metadata.get('page_number', '')}_" \
                   f"{metadata.get('chunk_index', '')}_" \
                   f"{content[:100]}"
    
    # Generar hash MD5
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def clean_text(text: str) -> str:
    """
    Limpiar y normalizar texto
    """
    # Eliminar caracteres de control
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
    
    # Normalizar espacios en blanco
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text

def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extraer números de un texto
    """
    # Patrón para números decimales y enteros
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers

def split_into_sentences(text: str) -> List[str]:
    """
    Dividir texto en oraciones
    """
    # Patrón mejorado para detectar fin de oración
    sentence_endings = r'[.!?]+[\s\n]+'
    sentences = re.split(sentence_endings, text)
    
    # Limpiar oraciones vacías
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def estimate_tokens(text: str) -> int:
    """
    Estimar número de tokens en un texto (aproximación)
    """
    # Aproximación: ~4 caracteres por token en promedio
    return len(text) // 4

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    Truncar texto para no exceder límite de tokens
    """
    estimated_tokens = estimate_tokens(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Calcular proporción para truncar
    ratio = max_tokens / estimated_tokens
    target_length = int(len(text) * ratio * 0.95)  # 95% para margen de seguridad
    
    return text[:target_length] + "..."

def validate_pdf_file(file_path: Union[str, Path]) -> bool:
    """
    Validar que un archivo es un PDF válido
    """
    file_path = Path(file_path)
    
    # Verificar que existe
    if not file_path.exists():
        logger.error(f"Archivo no encontrado: {file_path}")
        return False
    
    # Verificar extensión
    if file_path.suffix.lower() != '.pdf':
        logger.error(f"El archivo no tiene extensión .pdf: {file_path}")
        return False
    
    # Verificar tipo MIME
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type != 'application/pdf':
        logger.warning(f"Tipo MIME no coincide con PDF: {mime_type}")
    
    # Verificar magic number (primeros bytes)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                logger.error(f"El archivo no tiene cabecera PDF válida")
                return False
    except Exception as e:
        logger.error(f"Error al leer archivo: {e}")
        return False
    
    return True

def merge_overlapping_chunks(chunks: List[Dict[str, Any]], 
                           overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Fusionar chunks que tienen demasiada superposición
    """
    if len(chunks) <= 1:
        return chunks
    
    merged = []
    current = chunks[0]
    
    for next_chunk in chunks[1:]:
        # Calcular superposición
        overlap = calculate_text_overlap(current['content'], next_chunk['content'])
        
        if overlap > overlap_threshold:
            # Fusionar chunks
            current = {
                'content': merge_texts(current['content'], next_chunk['content']),
                'metadata': {
                    **current['metadata'],
                    'merged': True,
                    'original_chunks': [
                        current['metadata'].get('chunk_index'),
                        next_chunk['metadata'].get('chunk_index')
                    ]
                }
            }
        else:
            merged.append(current)
            current = next_chunk
    
    merged.append(current)
    return merged

def calculate_text_overlap(text1: str, text2: str) -> float:
    """
    Calcular porcentaje de superposición entre dos textos
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def merge_texts(text1: str, text2: str) -> str:
    """
    Fusionar dos textos eliminando duplicación
    """
    # Buscar superposición al final de text1 y principio de text2
    min_overlap = 20  # Mínimo de caracteres para considerar superposición
    
    for i in range(min_overlap, min(len(text1), len(text2))):
        if text1[-i:] == text2[:i]:
            # Encontrada superposición
            return text1 + text2[i:]
    
    # No hay superposición clara, concatenar con separador
    return text1 + " " + text2

def format_file_size(size_bytes: int) -> str:
    """
    Formatear tamaño de archivo en formato legible
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"

def create_summary_statistics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crear estadísticas resumidas de una colección de documentos
    """
    if not documents:
        return {}
    
    total_chars = sum(len(doc.get('content', '')) for doc in documents)
    
    # Agrupar por tipo
    by_type = {}
    for doc in documents:
        content_type = doc.get('metadata', {}).get('content_type', 'unknown')
        if content_type not in by_type:
            by_type[content_type] = 0
        by_type[content_type] += 1
    
    # Agrupar por manual
    by_manual = {}
    for doc in documents:
        manual = doc.get('metadata', {}).get('manual_name', 'unknown')
        if manual not in by_manual:
            by_manual[manual] = 0
        by_manual[manual] += 1
    
    return {
        'total_documents': len(documents),
        'total_characters': total_chars,
        'estimated_tokens': estimate_tokens(str(total_chars)),
        'by_type': by_type,
        'by_manual': by_manual,
        'average_doc_length': total_chars / len(documents) if documents else 0
    }

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Cargar JSON de manera segura con valor por defecto
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Error al decodificar JSON: {json_string[:100]}...")
        return default

def get_timestamp() -> str:
    """
    Obtener timestamp actual en formato ISO
    """
    return datetime.now().isoformat()

def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalizar y resolver ruta
    """
    return Path(path).resolve()

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Asegurar que un directorio existe, creándolo si es necesario
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Dividir una lista en chunks de tamaño específico
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Aplanar un diccionario anidado
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extraer metadatos del nombre de archivo
    """
    metadata = {}
    
    # Eliminar extensión
    name = Path(filename).stem
    
    # Buscar patrones comunes
    # Ejemplo: "Manual_Tractor_X_v2.1_2023"
    
    # Versión
    version_match = re.search(r'v(\d+\.?\d*)', name, re.IGNORECASE)
    if version_match:
        metadata['version'] = version_match.group(1)
    
    # Año
    year_match = re.search(r'(19|20)\d{2}', name)
    if year_match:
        metadata['year'] = int(year_match.group(0))
    
    # Modelo/Tipo
    model_match = re.search(r'model[_-]?(\w+)', name, re.IGNORECASE)
    if model_match:
        metadata['model'] = model_match.group(1)
    
    return metadata

# Constantes útiles
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
SUPPORTED_TABLE_FORMATS = {'.csv', '.xlsx', '.xls'}
MAX_CHUNK_SIZE = 1000  # Caracteres
MIN_CHUNK_SIZE = 100   # Caracteres

# Funciones de validación
def is_valid_manual_name(name: str) -> bool:
    """
    Validar nombre de manual
    """
    # No debe contener caracteres especiales que causen problemas en rutas
    invalid_chars = r'<>:"|?*'
    return not any(char in name for char in invalid_chars)

def sanitize_filename(filename: str) -> str:
    """
    Sanitizar nombre de archivo para uso seguro
    """
    # Reemplazar caracteres problemáticos
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    # Eliminar espacios múltiples
    filename = re.sub(r'\s+', '_', filename)
    # Limitar longitud
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename