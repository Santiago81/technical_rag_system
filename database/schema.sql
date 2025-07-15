

-- Habilitar extensiones para mejor performance
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB cache
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 1073741824;  -- 1GB memory-mapped I/O

-- ============================================
-- TABLA PRINCIPAL: DOCUMENTOS (antes manuals)
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    
    -- Versionado mejorado
    version TEXT,
    parent_document_id INTEGER,
    is_latest BOOLEAN DEFAULT 1,
    
    -- Metadatos extendidos
    language TEXT DEFAULT 'es',
    languages_detected TEXT,  -- JSON array para multi-idioma
    manufacturer TEXT,
    model TEXT,
    document_type TEXT NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    
    -- Estado y procesamiento
    processing_status TEXT DEFAULT 'pending',
    processing_duration_ms INTEGER,
    processing_metadata TEXT,  -- JSON con detalles del procesamiento
    
    -- Índices para búsqueda
    tsv_content TEXT,  -- Text Search Vector para FTS5
    
    FOREIGN KEY (parent_document_id) REFERENCES documents(id) ON DELETE SET NULL,
    CHECK (json_valid(languages_detected) OR languages_detected IS NULL),
    CHECK (json_valid(processing_metadata) OR processing_metadata IS NULL)
);

-- Índices optimizados para documents
CREATE INDEX idx_documents_uuid ON documents(uuid);
CREATE INDEX idx_documents_hash ON documents(file_hash);
CREATE INDEX idx_documents_status ON documents(processing_status) WHERE processing_status != 'completed';
CREATE INDEX idx_documents_latest ON documents(is_latest, document_type) WHERE is_latest = 1;
CREATE INDEX idx_documents_version ON documents(parent_document_id, version) WHERE parent_document_id IS NOT NULL;

-- ============================================
-- CHUNKS OPTIMIZADOS CON PARTICIONAMIENTO
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    
    -- Posición y contenido
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_tokens INTEGER,
    
    -- Localización mejorada
    start_page INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    
    -- Embeddings optimizados
    embedding BLOB,  -- Almacenado como BLOB comprimido
    embedding_model TEXT,
    embedding_dimension INTEGER,
    
    -- Contexto expandido
    context_window_before TEXT,
    context_window_after TEXT,
    structural_context TEXT,  -- JSON con info de sección/capítulo
    
    -- Metadatos de búsqueda
    keywords TEXT,  -- JSON array
    entities TEXT,  -- JSON array con tipos
    summary TEXT,  -- Resumen generado del chunk
    
    -- Scoring y relevancia
    importance_score REAL DEFAULT 1.0,
    quality_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index),
    CHECK (end_page >= start_page),
    CHECK (end_char >= start_char OR (end_char IS NULL AND start_char IS NULL))
);

-- Índices compuestos para chunks
CREATE INDEX idx_chunks_document ON chunks(document_id, chunk_index);
CREATE INDEX idx_chunks_pages ON chunks(document_id, start_page, end_page);
CREATE INDEX idx_chunks_embedding ON chunks(embedding_model, document_id) WHERE embedding IS NOT NULL;
CREATE INDEX idx_chunks_importance ON chunks(importance_score DESC, quality_score DESC);
CREATE INDEX idx_chunks_access ON chunks(last_accessed DESC) WHERE access_count > 0;

-- ============================================
-- RELACIONES SEMÁNTICAS ENTRE CHUNKS
-- ============================================
CREATE TABLE IF NOT EXISTS chunk_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_chunk_id INTEGER NOT NULL,
    target_chunk_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence_score REAL DEFAULT 1.0,
    metadata TEXT,  -- JSON
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (target_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    UNIQUE(source_chunk_id, target_chunk_id, relationship_type),
    CHECK (source_chunk_id != target_chunk_id),
    CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

CREATE INDEX idx_chunk_rel_source ON chunk_relationships(source_chunk_id, relationship_type);
CREATE INDEX idx_chunk_rel_target ON chunk_relationships(target_chunk_id, relationship_type);

-- ============================================
-- CONTENIDO VISUAL UNIFICADO
-- ============================================
CREATE TABLE IF NOT EXISTS visual_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    content_index INTEGER NOT NULL,
    
    -- Tipo y clasificación
    content_type TEXT NOT NULL,  -- 'image', 'diagram', 'chart', 'table_image'
    content_subtype TEXT,  -- 'technical_drawing', 'flowchart', 'schematic'
    
    -- Almacenamiento
    file_path TEXT NOT NULL,
    thumbnail_path TEXT,
    file_format TEXT NOT NULL,
    file_size INTEGER,
    
    -- Dimensiones y calidad
    width INTEGER,
    height INTEGER,
    dpi INTEGER,
    color_depth INTEGER,
    
    -- Extracción y análisis
    extraction_method TEXT,
    extraction_confidence REAL,
    
    -- OCR y texto
    ocr_text TEXT,
    ocr_confidence REAL,
    detected_text_regions TEXT,  -- JSON
    
    -- Embeddings multimodales
    visual_embedding BLOB,
    text_embedding BLOB,
    combined_embedding BLOB,
    
    -- Análisis de contenido
    detected_objects TEXT,  -- JSON
    technical_symbols TEXT,  -- JSON
    color_histogram TEXT,  -- JSON
    
    -- Relaciones
    related_chunks TEXT,  -- JSON array de chunk IDs
    caption TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, page_number, content_index)
);

CREATE INDEX idx_visual_document_page ON visual_content(document_id, page_number);
CREATE INDEX idx_visual_type ON visual_content(content_type, content_subtype);
CREATE INDEX idx_visual_embeddings ON visual_content(document_id) WHERE combined_embedding IS NOT NULL;

-- ============================================
-- TABLAS ESTRUCTURADAS
-- ============================================
CREATE TABLE IF NOT EXISTS structured_tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    table_index INTEGER NOT NULL,
    
    -- Estructura
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    headers TEXT NOT NULL,  -- JSON
    data_types TEXT,  -- JSON
    
    -- Contenido
    table_data TEXT NOT NULL,  -- JSON
    table_markdown TEXT,
    table_html TEXT,
    
    -- Análisis
    table_type TEXT,
    has_numeric_data BOOLEAN,
    numeric_columns TEXT,  -- JSON
    
    -- Embeddings y búsqueda
    content_embedding BLOB,
    summary TEXT,
    keywords TEXT,  -- JSON
    
    -- Calidad
    extraction_confidence REAL,
    validation_status TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, page_number, table_index)
);

CREATE INDEX idx_tables_document ON structured_tables(document_id, page_number);
CREATE INDEX idx_tables_type ON structured_tables(table_type) WHERE table_type IS NOT NULL;

-- ============================================
-- CACHE INTELIGENTE
-- ============================================
CREATE TABLE IF NOT EXISTS search_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding BLOB,
    
    -- Resultados
    result_chunks TEXT NOT NULL,  -- JSON
    result_scores TEXT NOT NULL,  -- JSON
    result_metadata TEXT,  -- JSON
    
    -- Estadísticas
    hit_count INTEGER DEFAULT 1,
    avg_relevance_score REAL,
    user_feedback_score REAL,
    
    -- TTL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    CHECK (json_valid(result_chunks)),
    CHECK (json_valid(result_scores))
);

CREATE INDEX idx_cache_hash ON search_cache(query_hash);
CREATE INDEX idx_cache_accessed ON search_cache(last_accessed DESC);
CREATE INDEX idx_cache_expires ON search_cache(expires_at) WHERE expires_at IS NOT NULL;

-- ============================================
-- FTS5 OPTIMIZADO
-- ============================================
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_text,
    keywords,
    summary,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2',
    prefix='2 3 4'
);

-- Triggers para sincronización FTS5
CREATE TRIGGER chunks_fts_sync_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_text, keywords, summary)
    VALUES (new.id, new.chunk_text, new.keywords, new.summary);
END;

CREATE TRIGGER chunks_fts_sync_update AFTER UPDATE ON chunks BEGIN
    UPDATE chunks_fts 
    SET chunk_text = new.chunk_text, 
        keywords = new.keywords,
        summary = new.summary
    WHERE rowid = new.id;
END;

CREATE TRIGGER chunks_fts_sync_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
END;

-- ============================================
-- VISTAS MATERIALIZADAS (simuladas)
-- ============================================
CREATE TABLE IF NOT EXISTS mv_document_stats AS
SELECT 
    d.id as document_id,
    d.name,
    d.document_type,
    COUNT(DISTINCT c.id) as chunk_count,
    COUNT(DISTINCT v.id) as visual_count,
    COUNT(DISTINCT t.id) as table_count,
    AVG(c.importance_score) as avg_importance,
    MAX(c.last_accessed) as last_accessed
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN visual_content v ON d.id = v.document_id
LEFT JOIN structured_tables t ON d.id = t.document_id
WHERE d.is_latest = 1
GROUP BY d.id;

CREATE INDEX idx_mv_stats_document ON mv_document_stats(document_id);
CREATE INDEX idx_mv_stats_accessed ON mv_document_stats(last_accessed DESC);

-- Trigger para actualizar vista materializada
CREATE TRIGGER update_mv_stats AFTER INSERT ON chunks
BEGIN
    DELETE FROM mv_document_stats WHERE document_id = new.document_id;
    INSERT INTO mv_document_stats 
    SELECT 
        d.id as document_id,
        d.name,
        d.document_type,
        COUNT(DISTINCT c.id) as chunk_count,
        COUNT(DISTINCT v.id) as visual_count,
        COUNT(DISTINCT t.id) as table_count,
        AVG(c.importance_score) as avg_importance,
        MAX(c.last_accessed) as last_accessed
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    LEFT JOIN visual_content v ON d.id = v.document_id
    LEFT JOIN structured_tables t ON d.id = t.document_id
    WHERE d.id = new.document_id AND d.is_latest = 1
    GROUP BY d.id;
END;