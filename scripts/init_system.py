#!/usr/bin/env python3
"""
Script de inicialización del sistema RAG
Verifica dependencias, crea estructura de directorios y configura el entorno
"""
import sys
import subprocess
from pathlib import Path
import importlib.util
import logging
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SystemInitializer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def check_python_version(self):
        """Verificar versión de Python"""
        logger.info("Verificando versión de Python...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.errors.append(f"Python 3.8+ requerido. Versión actual: {version.major}.{version.minor}")
        else:
            logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    def check_dependencies(self):
        """Verificar dependencias requeridas"""
        logger.info("\nVerificando dependencias...")
        
        # Dependencias críticas
        critical_packages = {
            'PyPDF2': 'PyPDF2',
            'fitz': 'PyMuPDF',
            'sentence_transformers': 'sentence-transformers',
            'langchain': 'langchain',
            'chromadb': 'chromadb',
            'tabula': 'tabula-py',
            'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'pytesseract': 'pytesseract',
            'tqdm': 'tqdm',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'pdfplumber': 'pdfplumber'  # Alternative to camelot-py for Python 3.12
        }
        
        # Check for camelot or pdfplumber (Python 3.12 alternative)
        camelot_spec = importlib.util.find_spec('camelot')
        pdfplumber_spec = importlib.util.find_spec('pdfplumber')
        
        if camelot_spec is None and pdfplumber_spec is None:
            self.warnings.append("Neither camelot-py nor pdfplumber found. Install pdfplumber for Python 3.12+")
        elif pdfplumber_spec and not camelot_spec:
            logger.info("✓ Using pdfplumber (Python 3.12+ compatible alternative to camelot-py)")
        elif camelot_spec:
            logger.info("✓ camelot-py")
        
        missing_packages = []
        
        for module_name, package_name in critical_packages.items():
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                missing_packages.append(package_name)
                logger.warning(f"✗ {package_name} no encontrado")
            else:
                logger.info(f"✓ {package_name}")
        
        if missing_packages:
            self.errors.append(f"Paquetes faltantes: {', '.join(missing_packages)}")
            logger.error("\nInstalar paquetes faltantes con:")
            logger.error(f"pip install {' '.join(missing_packages)}")
    
    def check_system_dependencies(self):
        """Verificar dependencias del sistema"""
        logger.info("\nVerificando dependencias del sistema...")
        
        # Verificar Tesseract
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ Tesseract OCR instalado")
            else:
                self.warnings.append("Tesseract OCR no encontrado - OCR deshabilitado")
        except FileNotFoundError:
            self.warnings.append("Tesseract OCR no instalado - funcionalidad OCR no disponible")
            logger.warning("Para instalar Tesseract:")
            logger.warning("  macOS: brew install tesseract")
            logger.warning("  Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-spa")
        
        # Verificar Java para Tabula
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ Java instalado (requerido para Tabula)")
            else:
                self.warnings.append("Java no encontrado - extracción de tablas con Tabula deshabilitada")
        except FileNotFoundError:
            self.warnings.append("Java no instalado - funcionalidad Tabula no disponible")
    
    def create_directory_structure(self):
        """Crear estructura de directorios necesaria"""
        logger.info("\nCreando estructura de directorios...")
        
        directories = [
            'data/raw_pdfs',
            'data/processed/texts',
            'data/processed/tables',
            'data/processed/images',
            'data/processed/diagrams',
            'data/processed/metadata',
            'data/vectordb',
            'data/sqlite',
            'data/logs',
            'data/processing_logs',
            'data/cache/embedding_cache',
            'data/cache/search_cache'
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ {dir_path}")
    
    def create_env_file(self):
        """Crear archivo .env si no existe"""
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if not env_file.exists() and env_example.exists():
            logger.info("\nCreando archivo .env desde .env.example...")
            import shutil
            shutil.copy(env_example, env_file)
            logger.info("✓ Archivo .env creado")
            logger.warning("⚠ Revisa y configura las variables en .env")
        elif env_file.exists():
            logger.info("\n✓ Archivo .env ya existe")
    
    def initialize_database(self):
        """Inicializar base de datos SQLite"""
        logger.info("\nInicializando base de datos SQLite...")
        
        try:
            sys.path.append(str(self.project_root))
            from database.sqlite_manager import SQLiteRAGManager
            from config.settings import Config
            
            config = Config()
            db_manager = SQLiteRAGManager(str(config.SQLITE_DB_PATH))
            
            # Verificar si la base de datos existe
            if config.SQLITE_DB_PATH.exists():
                logger.info("✓ Base de datos SQLite ya existe")
            else:
                # Crear base de datos
                logger.info("Creando nueva base de datos...")
                # El schema se aplicará automáticamente al crear una conexión
                logger.info("✓ Base de datos SQLite creada")
                
        except Exception as e:
            self.errors.append(f"Error inicializando base de datos: {str(e)}")
            logger.error(f"✗ Error con base de datos: {str(e)}")
    
    def verify_configuration(self):
        """Verificar configuración del sistema"""
        logger.info("\nVerificando configuración...")
        
        try:
            sys.path.append(str(self.project_root))
            from config.settings import Config
            
            config = Config()
            config.validate()
            logger.info("✓ Configuración válida")
            
        except Exception as e:
            self.errors.append(f"Error en configuración: {str(e)}")
            logger.error(f"✗ Error de configuración: {str(e)}")
    
    def generate_report(self):
        """Generar reporte de inicialización"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if not self.errors else 'failed',
            'errors': self.errors,
            'warnings': self.warnings,
            'project_root': str(self.project_root)
        }
        
        report_file = self.project_root / 'data' / 'logs' / 'initialization_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReporte guardado en: {report_file}")
        
        return report
    
    def run(self):
        """Ejecutar proceso de inicialización completo"""
        logger.info("=== Inicialización del Sistema RAG ===\n")
        
        # Ejecutar verificaciones
        self.check_python_version()
        self.check_dependencies()
        self.check_system_dependencies()
        self.create_directory_structure()
        self.create_env_file()
        self.verify_configuration()
        self.initialize_database()
        
        # Generar reporte
        report = self.generate_report()
        
        # Mostrar resumen
        logger.info("\n=== Resumen de Inicialización ===")
        
        if self.errors:
            logger.error(f"\n❌ Sistema NO está listo para usar")
            logger.error(f"Errores encontrados: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  - {error}")
        else:
            logger.info(f"\n✅ Sistema listo para usar")
            
        if self.warnings:
            logger.warning(f"\nAdvertencias: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("\n=== Próximos pasos ===")
        logger.info("1. Configurar variables en .env si es necesario")
        logger.info("2. Colocar PDFs en data/raw_pdfs/")
        logger.info("3. Ejecutar: python scripts/process_manuals.py")
        
        return len(self.errors) == 0

def main():
    initializer = SystemInitializer()
    success = initializer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()