"""
Módulo de extractores para procesar PDFs
"""
# Extractores base
from .pdf_extractor import PDFExtractor
from .text_processor import TextProcessor
from .image_extractor import ImageExtractor
from .enhanced_image_extractor import EnhancedImageExtractor
from .table_extractor import TableExtractor

# Análisis y procesamiento adaptativo
from .document_analyzer import DocumentAnalyzer, DocumentType
from .adaptive_processor import AdaptiveManualProcessor

# Extractores SQLite (si el archivo existe)
try:
    from .sqlite_extractors import (
        SQLitePDFExtractor,
        SQLiteTextProcessor,
        SQLiteImageExtractor,
        SQLiteTableExtractor,
        SQLiteDocumentAnalyzer
    )
    _sqlite_available = True
except ImportError:
    _sqlite_available = False

# Lista de exportaciones
__all__ = [
    # Extractores base
    'PDFExtractor',
    'TextProcessor',
    'ImageExtractor',
    'EnhancedImageExtractor',
    'TableExtractor',
    # Análisis y procesamiento
    'DocumentAnalyzer',
    'DocumentType',
    'AdaptiveManualProcessor',
]

# Añadir extractores SQLite si están disponibles
if _sqlite_available:
    __all__.extend([
        'SQLitePDFExtractor',
        'SQLiteTextProcessor',
        'SQLiteImageExtractor',
        'SQLiteTableExtractor',
        'SQLiteDocumentAnalyzer'
    ])

# Versión del módulo
__version__ = '1.2.0'  # Actualizado por SQLite support