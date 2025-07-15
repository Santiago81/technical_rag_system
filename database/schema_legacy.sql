-- Schema compatible con el código actual
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- Tabla de manuales (compatible con el código actual)
CREATE TABLE IF NOT EXISTS manuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    
    -- Metadatos
    manufacturer TEXT,
    model TEXT,
    version TEXT,
    document_type TEXT DEFAULT 'technical',
    language TEXT DEFAULT 'es',
    
    -- Estado de procesamiento
    processing_status TEXT DEFAULT 'pending',
    processing_duration_ms INTEGER,
    
    -- Fechas
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Estadísticas
    total_pages INTEGER,
    page_count INTEGER,
    total_chunks INTEGER,
    total_images INTEGER,
    total_tables INTEGER
);

-- Tabla de bloques de contenido
CREATE TABLE IF NOT EXISTS content_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    block_index INTEGER NOT NULL,
    block_type TEXT NOT NULL,
    content TEXT NOT NULL,
    
    -- Contexto estructural
    section TEXT,
    chapter TEXT,
    
    -- Metadatos
    confidence_score REAL DEFAULT 1.0,
    bounding_box TEXT,
    style_attributes TEXT,
    
    -- Estadísticas
    char_count INTEGER,
    word_count INTEGER,
    
    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE,
    UNIQUE(manual_id, page_number, block_index)
);

-- Tabla de chunks
CREATE TABLE IF NOT EXISTS content_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_text_processed TEXT,
    
    -- Posición
    chunk_index INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL,
    overlap_size INTEGER DEFAULT 0,
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    
    -- Contexto
    context_before TEXT,
    context_after TEXT,
    content_block_id INTEGER,
    
    -- Embeddings
    embedding BLOB,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    embedding_date TIMESTAMP,
    
    -- Metadatos de búsqueda
    keywords TEXT,
    entities TEXT,
    
    -- Scoring
    importance_score REAL DEFAULT 1.0,
    tf_idf_scores TEXT,
    metadata_json TEXT,
    
    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE,
    UNIQUE(manual_id, chunk_index)
);

-- Tabla de imágenes
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    image_index INTEGER NOT NULL,
    
    -- Tipo y formato
    image_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_format TEXT NOT NULL,
    file_size INTEGER,
    file_hash TEXT UNIQUE,
    
    -- Dimensiones
    width INTEGER,
    height INTEGER,
    color_space TEXT,
    
    -- OCR
    ocr_text TEXT,
    
    -- Embeddings visuales
    visual_embedding BLOB,
    
    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE,
    UNIQUE(manual_id, page_number, image_index)
);

-- Tabla de tablas extraídas
CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    table_index INTEGER NOT NULL,
    
    -- Método de extracción
    extraction_method TEXT,
    extraction_accuracy REAL,
    
    -- Estructura
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    headers TEXT,
    data_types TEXT,
    
    -- Contenido
    csv_path TEXT,
    table_content TEXT,
    
    -- Análisis
    has_numeric_data BOOLEAN,
    has_headers BOOLEAN,
    
    -- Metadatos
    metadata_json TEXT,
    
    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE,
    UNIQUE(manual_id, page_number, table_index)
);

-- Tabla de análisis de documentos (que faltaba)
CREATE TABLE IF NOT EXISTS document_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_id INTEGER NOT NULL,
    
    -- Tipo de documento
    document_type TEXT NOT NULL,
    
    -- Distribución de contenido
    text_pages INTEGER,
    image_pages INTEGER,
    mixed_pages INTEGER,
    empty_pages INTEGER,
    
    -- Métricas
    avg_text_per_page REAL,
    image_frequency REAL,
    table_frequency REAL,
    
    -- Estrategia recomendada
    recommended_chunk_size INTEGER,
    recommended_overlap INTEGER,
    use_ocr BOOLEAN DEFAULT 0,
    
    -- Timestamp
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE
);

-- Tabla de logs de procesamiento
CREATE TABLE IF NOT EXISTS processing_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manual_id INTEGER NOT NULL,
    process_type TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT,
    details_json TEXT,
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (manual_id) REFERENCES manuals(id) ON DELETE CASCADE
);

-- Índices para optimización
CREATE INDEX idx_manuals_status ON manuals(processing_status);
CREATE INDEX idx_chunks_manual ON content_chunks(manual_id);
CREATE INDEX idx_chunks_embedding ON content_chunks(manual_id) WHERE embedding IS NOT NULL;
CREATE INDEX idx_blocks_manual ON content_blocks(manual_id, page_number);
CREATE INDEX idx_images_manual ON images(manual_id, page_number);
CREATE INDEX idx_tables_manual ON tables(manual_id, page_number);
CREATE INDEX idx_logs_manual ON processing_logs(manual_id, timestamp DESC);