import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
import re
import tabula

# Try to import camelot, fall back to pdfplumber if not available
try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False
    try:
        import pdfplumber
        HAS_PDFPLUMBER = True
    except ImportError:
        HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)

class TableExtractor:
    """Extractor especializado para tablas de PDFs"""
    
    def __init__(self):
        if HAS_CAMELOT:
            self.extraction_methods = ['camelot-lattice', 'camelot-stream', 'tabula', 'ocr']
        elif HAS_PDFPLUMBER:
            self.extraction_methods = ['pdfplumber', 'tabula', 'ocr']
        else:
            self.extraction_methods = ['tabula', 'ocr']
    
    def extract_tables_from_pdf(self, pdf_path: str, output_dir: Path) -> List[Dict]:
        """Extraer todas las tablas de un PDF usando múltiples métodos"""
        pdf_path = Path(pdf_path)
        manual_name = pdf_path.stem
        all_tables = []
        
        # Asegurar que el directorio existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Intentar diferentes métodos de extracción
        for method in self.extraction_methods:
            try:
                if method == 'camelot-lattice' and HAS_CAMELOT:
                    tables = self._extract_with_camelot(pdf_path, 'lattice')
                elif method == 'camelot-stream' and HAS_CAMELOT:
                    tables = self._extract_with_camelot(pdf_path, 'stream')
                elif method == 'pdfplumber' and HAS_PDFPLUMBER:
                    tables = self._extract_with_pdfplumber(pdf_path)
                elif method == 'tabula':
                    tables = self._extract_with_tabula(pdf_path)
                elif method == 'ocr':
                    tables = self._extract_with_ocr(pdf_path)
                
                if tables:
                    logger.info(f"Extraídas {len(tables)} tablas con método {method}")
                    
                    # Procesar y guardar cada tabla
                    for idx, table_data in enumerate(tables):
                        table_info = self._process_table(
                            table_data, 
                            manual_name,
                            method,
                            idx,
                            output_dir
                        )
                        if table_info:
                            all_tables.append(table_info)
                            
            except Exception as e:
                logger.warning(f"Error con método {method}: {str(e)}")
                continue
        
        return all_tables
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extraer tablas usando pdfplumber"""
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table_data in page_tables:
                    if table_data and len(table_data) > 1:
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        df.attrs['page'] = page_num + 1
                        tables.append(df)
        
        return tables
    
    def _extract_with_camelot(self, pdf_path: Path, flavor: str = 'lattice') -> List[pd.DataFrame]:
        """Extraer tablas usando Camelot"""
        tables = []
        
        try:
            # Extraer tablas de todas las páginas
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                pages='all',
                flavor=flavor,
                suppress_stdout=True
            )
            
            for table in camelot_tables:
                df = table.df
                df.attrs['page'] = table.page
                df.attrs['accuracy'] = table.accuracy
                tables.append(df)
                
        except Exception as e:
            logger.warning(f"Error en Camelot {flavor}: {str(e)}")
            
        return tables
    
    def _extract_with_tabula(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extraer tablas usando Tabula"""
        tables = []
        
        try:
            # Intentar diferentes estrategias
            strategies = ['lattice', 'stream']
            
            for strategy in strategies:
                try:
                    dfs = tabula.read_pdf(
                        str(pdf_path),
                        pages='all',
                        multiple_tables=True,
                        lattice=(strategy == 'lattice'),
                        stream=(strategy == 'stream'),
                        silent=True
                    )
                    
                    if dfs:
                        tables.extend(dfs)
                        break
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error en Tabula: {str(e)}")
            
        return tables
    
    def _extract_with_ocr(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extraer tablas usando OCR (para PDFs escaneados)"""
        tables = []
        
        try:
            # Convertir páginas a imágenes
            images = convert_from_path(str(pdf_path), dpi=300)
            
            for page_num, image in enumerate(images):
                try:
                    # Aplicar OCR
                    text = pytesseract.image_to_string(image)
                    
                    # Buscar patrones de tabla
                    table_data = self._parse_table_from_text(text)
                    
                    if table_data:
                        df = pd.DataFrame(table_data)
                        df.attrs['page'] = page_num + 1
                        df.attrs['method'] = 'ocr'
                        tables.append(df)
                        
                except Exception as e:
                    logger.warning(f"Error en OCR página {page_num + 1}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Error en conversión a imagen: {str(e)}")
            
        return tables
    
    def _parse_table_from_text(self, text: str) -> Optional[List[List[str]]]:
        """Parsear tabla desde texto OCR"""
        lines = text.strip().split('\n')
        table_data = []
        
        for line in lines:
            # Buscar líneas con múltiples columnas separadas por espacios
            if len(line.split()) > 2:
                row = [cell.strip() for cell in re.split(r'\s{2,}|\t', line)]
                if row:
                    table_data.append(row)
        
        # Verificar que tenemos una tabla válida
        if len(table_data) > 2:
            # Normalizar número de columnas
            max_cols = max(len(row) for row in table_data)
            normalized_data = []
            
            for row in table_data:
                normalized_row = row + [''] * (max_cols - len(row))
                normalized_data.append(normalized_row[:max_cols])
            
            return normalized_data
        
        return None
    
    def _process_table(self, df: pd.DataFrame, manual_name: str, 
                      method: str, idx: int, output_dir: Path) -> Optional[Dict]:
        """Procesar y guardar tabla extraída"""
        try:
            # Limpiar tabla
            df = self._clean_table(df)
            
            if df.empty or len(df) < 2:
                return None
            
            # Generar metadata
            timestamp = datetime.now().isoformat()
            page_num = df.attrs.get('page', 'unknown')
            
            # Crear nombre único
            table_id = f"{manual_name}_page{page_num}_table{idx}_{method}"
            
            # Guardar como CSV
            csv_path = output_dir / f"{table_id}.csv"
            df.to_csv(csv_path, index=False)
            
            # Guardar como Excel
            excel_path = output_dir / f"{table_id}.xlsx"
            df.to_excel(excel_path, index=False)
            
            # Generar resumen
            summary = self._generate_table_summary(df)
            
            return {
                'table_id': table_id,
                'manual_name': manual_name,
                'page_number': page_num,
                'extraction_method': method,
                'num_rows': len(df),
                'num_cols': len(df.columns),
                'columns': df.columns.tolist(),
                'summary': summary,
                'csv_path': str(csv_path),
                'excel_path': str(excel_path),
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Error procesando tabla: {str(e)}")
            return None
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y normalizar tabla"""
        # Remover filas y columnas vacías
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Remover espacios en blanco extras
        df = df.map(lambda x: str(x).strip() if pd.notna(x) else '')
        
        # Intentar establecer headers si la primera fila parece ser encabezado
        if len(df) > 0:
            first_row = df.iloc[0]
            if all(isinstance(val, str) and val for val in first_row):
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
        
        return df
    
    def _generate_table_summary(self, df: pd.DataFrame) -> str:
        """Generar resumen descriptivo de la tabla"""
        summary_parts = []
        
        # Información básica
        summary_parts.append(f"Tabla con {len(df)} filas y {len(df.columns)} columnas")
        
        # Columnas
        cols_str = ", ".join(df.columns[:5])
        if len(df.columns) > 5:
            cols_str += f" ... ({len(df.columns)} columnas en total)"
        summary_parts.append(f"Columnas: {cols_str}")
        
        # Detectar tipo de contenido
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"Contiene {len(numeric_cols)} columnas numéricas")
        
        return ". ".join(summary_parts)