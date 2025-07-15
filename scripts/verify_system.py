#!/usr/bin/env python3
"""
Script de verificación rápida del sistema RAG
Ejecuta pruebas básicas para confirmar que el sistema está operativo
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from config.settings import Config
import importlib.util

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_critical_imports():
    """Verificar que los imports críticos funcionen"""
    logger.info("Verificando imports críticos...")
    
    critical_imports = [
        ('PyPDF2', 'pypdf2'),
        ('fitz (PyMuPDF)', 'fitz'),
        ('langchain', 'langchain'),
        ('sentence_transformers', 'sentence_transformers'),
        ('chromadb', 'chromadb'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('PIL', 'PIL'),
        ('tqdm', 'tqdm')
    ]
    
    all_ok = True
    for name, module in critical_imports:
        try:
            importlib.import_module(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - No instalado")
            all_ok = False
    
    return all_ok

def verify_directory_structure():
    """Verificar estructura de directorios"""
    logger.info("\nVerificando estructura de directorios...")
    
    config = Config()
    directories = [
        config.RAW_PDF_DIR,
        config.PROCESSED_DIR,
        config.VECTOR_DB_DIR,
        config.SQLITE_DIR,
        config.LOGS_DIR
    ]
    
    all_ok = True
    for directory in directories:
        if directory.exists():
            logger.info(f"✓ {directory.relative_to(config.BASE_DIR)}")
        else:
            logger.error(f"✗ {directory.relative_to(config.BASE_DIR)} - No existe")
            all_ok = False
    
    return all_ok

def verify_pdf_files():
    """Verificar presencia de PDFs para procesar"""
    logger.info("\nVerificando archivos PDF...")
    
    config = Config()
    pdf_files = list(config.RAW_PDF_DIR.glob("*.pdf"))
    
    if pdf_files:
        logger.info(f"✓ {len(pdf_files)} PDFs encontrados en {config.RAW_PDF_DIR}")
        for pdf in pdf_files[:3]:  # Mostrar primeros 3
            logger.info(f"  - {pdf.name}")
        if len(pdf_files) > 3:
            logger.info(f"  ... y {len(pdf_files) - 3} más")
    else:
        logger.warning(f"⚠ No hay PDFs en {config.RAW_PDF_DIR}")
        logger.info("  Coloca archivos PDF en este directorio para procesarlos")
    
    return True

def verify_database():
    """Verificar base de datos SQLite"""
    logger.info("\nVerificando base de datos...")
    
    try:
        from database.sqlite_manager import SQLiteRAGManager
        config = Config()
        
        db_manager = SQLiteRAGManager(str(config.SQLITE_DB_PATH))
        
        # Verificar tablas
        with db_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['documents', 'chunks', 'visual_content', 'structured_tables']
        
        all_ok = True
        for table in expected_tables:
            if table in tables:
                logger.info(f"✓ Tabla '{table}' existe")
            else:
                logger.error(f"✗ Tabla '{table}' no encontrada")
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        logger.error(f"✗ Error verificando base de datos: {str(e)}")
        return False

def quick_test():
    """Prueba rápida de funcionalidad básica"""
    logger.info("\nPrueba rápida de funcionalidad...")
    
    try:
        # Probar extractor de PDF
        from extractors.pdf_extractor import PDFExtractor
        config = Config()
        extractor = PDFExtractor(config)
        logger.info("✓ PDFExtractor inicializado correctamente")
        
        # Probar analizador de documentos
        from extractors.document_analyzer import DocumentAnalyzer
        analyzer = DocumentAnalyzer()
        logger.info("✓ DocumentAnalyzer inicializado correctamente")
        
        # Probar procesador adaptativo
        from extractors.adaptive_processor import AdaptiveManualProcessor
        processor = AdaptiveManualProcessor(config)
        logger.info("✓ AdaptiveManualProcessor inicializado correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en prueba de funcionalidad: {str(e)}")
        return False

def main():
    logger.info("=== Verificación del Sistema RAG ===\n")
    
    results = {
        'imports': verify_critical_imports(),
        'directories': verify_directory_structure(),
        'pdfs': verify_pdf_files(),
        'database': verify_database(),
        'functionality': quick_test()
    }
    
    # Resumen
    logger.info("\n=== Resumen ===")
    all_ok = all(results.values())
    
    if all_ok:
        logger.info("✅ Sistema verificado y listo para usar")
        logger.info("\nPróximo paso: python scripts/process_manuals.py")
    else:
        logger.error("❌ Problemas detectados")
        logger.info("\nEjecuta 'python scripts/init_system.py' para resolver problemas")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())