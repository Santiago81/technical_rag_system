# config/settings.py
"""
Configuración actualizada del sistema RAG con soporte SQLite
"""
from pathlib import Path
import os
from typing import Optional

class Config:
    """Configuración central del sistema"""
    
    def __init__(self):
        # Directorio base del proyecto
        self.BASE_DIR = Path(__file__).parent.parent
        
        # ========== DIRECTORIOS ==========
        # Directorio principal de datos
        self.DATA_DIR = self.BASE_DIR / "data"
        
        # Subdirectorios de datos
        self.RAW_PDF_DIR = self.DATA_DIR / "raw_pdfs"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.VECTOR_DB_DIR = self.DATA_DIR / "vectordb"
        self.SQLITE_DIR = self.DATA_DIR / "sqlite"
        self.LOGS_DIR = self.DATA_DIR / "logs"
        
        # Crear directorios si no existen
        for dir_path in [self.RAW_PDF_DIR, self.PROCESSED_DIR, 
                         self.VECTOR_DB_DIR, self.SQLITE_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # ========== BASE DE DATOS ==========
        # SQLite como backend principal
        self.USE_SQLITE = True  # Si False, usa el sistema de archivos antiguo
        self.SQLITE_DB_PATH = self.SQLITE_DIR / "manuals.db"
        self.SQLITE_SCHEMA_PATH = self.BASE_DIR / "database" / "schema.sql"
        
        # Configuración de SQLite
        self.SQLITE_CONFIG = {
            'timeout': 30.0,
            'check_same_thread': False,  # Permitir uso multi-thread
            'cached_statements': 100
        }
        
        # ========== PROCESAMIENTO DE TEXTO ==========
        # Configuración de chunks
        self.CHUNK_SIZE = 512
        self.CHUNK_OVERLAP = 50
        
        # Configuración adaptativa por tipo de documento
        self.ADAPTIVE_CHUNK_SIZES = {
            'technical_diagram_heavy': 1024,
            'text_heavy': 512,
            'table_heavy': 768,
            'scanned': 768,
            'mixed': 512
        }
        
        # ========== MODELOS Y EMBEDDINGS ==========
        # Modelo de embeddings
        self.EMBEDDING_MODEL = os.getenv(
            'EMBEDDING_MODEL',
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Mejor para español
        )
        self.EMBEDDING_DIMENSION = 384
        
        # Cache de embeddings
        self.CACHE_EMBEDDINGS = True
        self.EMBEDDING_CACHE_TTL = 86400  # 24 horas en segundos
        
        # ========== EXTRACCIÓN DE CONTENIDO ==========
        # Configuración de imágenes
        self.MIN_IMAGE_SIZE = (100, 100)
        self.IMAGE_OUTPUT_FORMAT = 'png'
        self.DIAGRAM_DPI = 200
        self.ENABLE_OCR = True
        self.OCR_LANGUAGES = ['spa', 'eng']
        
        # Configuración de tablas
        self.TABLE_EXTRACTION_METHOD = 'camelot_first'  # 'camelot_first', 'tabula', 'both'
        self.SAVE_TABLES_AS_EXCEL = False  # Solo CSV
        
        # ========== VECTOR STORE ==========
        # Configuración de ChromaDB (para compatibilidad)
        self.CHROMA_PERSIST_DIRECTORY = str(self.VECTOR_DB_DIR)
        self.CHROMA_COLLECTION_NAME = "technical_manuals"
        
        # Modo de operación del vector store
        self.VECTOR_STORE_BACKEND = 'sqlite'  # 'sqlite', 'chromadb', 'hybrid'
        
        # ========== BÚSQUEDA Y RECUPERACIÓN ==========
        # Configuración de búsqueda
        self.SEARCH_K = 10  # Número de resultados por defecto
        self.MIN_SIMILARITY_SCORE = 0.7
        self.USE_RERANKING = True
        self.RERANK_TOP_K = 5
        
        # Búsqueda híbrida
        self.HYBRID_SEARCH_ALPHA = 0.7  # Peso para búsqueda vectorial vs keyword
        
        # ========== API Y SERVICIOS ==========
        # Configuración de API
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.API_WORKERS = 4
        
        # Límites de API
        self.MAX_UPLOAD_SIZE_MB = 100
        self.MAX_QUERY_LENGTH = 500
        
        # ========== LOGGING Y MONITOREO ==========
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # ========== PERFORMANCE ==========
        # Configuración de procesamiento
        self.BATCH_SIZE = 32
        self.MAX_WORKERS = 4
        self.PROCESSING_TIMEOUT = 3600  # 1 hora
        
        # ========== FEATURES FLAGS ==========
        # Habilitar/deshabilitar características
        self.ENABLE_DOCUMENT_ANALYSIS = True
        self.ENABLE_AUTO_CHUNKING = True
        self.ENABLE_SEMANTIC_CACHE = True
        self.ENABLE_FEEDBACK_LEARNING = True
        
    def get_db_connection_string(self) -> str:
        """Obtener string de conexión para SQLite"""
        return f"sqlite:///{self.SQLITE_DB_PATH}"
    
    def get_processing_options(self, doc_type: str) -> dict:
        """Obtener opciones de procesamiento según tipo de documento"""
        return {
            'chunk_size': self.ADAPTIVE_CHUNK_SIZES.get(doc_type, self.CHUNK_SIZE),
            'chunk_overlap': self.CHUNK_OVERLAP,
            'extract_images': True,
            'extract_tables': True,
            'use_ocr': doc_type in ['scanned', 'technical_diagram_heavy'],
            'diagram_dpi': self.DIAGRAM_DPI if doc_type == 'technical_diagram_heavy' else 150
        }
    
    def validate(self):
        """Validar configuración"""
        errors = []
        
        # Verificar directorios
        if not self.BASE_DIR.exists():
            errors.append(f"Directorio base no existe: {self.BASE_DIR}")
        
        # Verificar modelo de embeddings
        if not self.EMBEDDING_MODEL:
            errors.append("Modelo de embeddings no configurado")
        
        # Verificar configuración de SQLite
        if self.USE_SQLITE and not self.SQLITE_SCHEMA_PATH.exists():
            errors.append(f"Schema SQLite no encontrado: {self.SQLITE_SCHEMA_PATH}")
        
        if errors:
            raise ValueError(f"Errores de configuración: {'; '.join(errors)}")
        
        return True

# Instancia global de configuración
config = Config()