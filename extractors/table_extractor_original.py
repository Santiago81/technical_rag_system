import camelot
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
import re
import tabula

logger = logging.getLogger(__name__)

class TableExtractor:
    """Extractor especializado para tablas de PDFs"""
    
    def __init__(self):
        self.extraction_methods = ['camelot-lattice', 'camelot-stream', 'tabula', 'ocr']
    
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
                if method == 'camelot-lattice':
                    tables = self._extract_with_camelot(pdf_path, 'lattice')
                elif method == 'camelot-stream':
                    tables = self._extract_with_camelot(pdf_path, 'stream')
                elif method == 'tabula':
                    tables = self._extract_with_tabula(pdf_path)
                elif method == 'ocr':
                    tables = self._extract_with_ocr(pdf_path)
                
                if tables:
                    logger.info(f"Extraídas {len(tables)} tablas usando {method}")
                    
                    # Procesar y guardar tablas
                    for i, table_data in enumerate(tables):
                        metadata = self._process_table(
                            table_data, i, manual_name, output_dir, method
                        )
                        all_tables.append(metadata)
                    
                    break  # Si tuvo éxito, no intentar otros métodos
                    
            except Exception as e:
                logger.warning(f"Error con método {method}: {e}")
                continue
        
        return all_tables
    
    def _extract_with_camelot(self, pdf_path: Path, flavor: str) -> List[Dict]:
        """Extraer tablas usando Camelot"""
        tables = camelot.read_pdf(str(pdf_path), pages='all', flavor=flavor)
        
        result = []
        for table in tables:
            result.append({
                'df': table.df,
                'page': table.page,
                'accuracy': table.accuracy
            })
        
        return result if tables else []
    
    def _extract_with_tabula(self, pdf_path: Path) -> List[Dict]:
        """Extraer tablas usando Tabula como alternativa"""
        try:
            # Extraer todas las tablas
            tables = tabula.read_pdf(
                str(pdf_path), 
                pages='all',
                multiple_tables=True,
                lattice=True  # Intentar primero con lattice
            )
            
            if not tables:
                # Intentar con stream si lattice no funciona
                tables = tabula.read_pdf(
                    str(pdf_path), 
                    pages='all',
                    multiple_tables=True,
                    stream=True
                )
            
            result = []
            for i, df in enumerate(tables):
                if not df.empty:
                    result.append({
                        'df': df,
                        'page': i + 1,  # Aproximación
                        'accuracy': 0.8  # Valor estimado
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error con Tabula: {e}")
            return []
    
    def _extract_with_ocr(self, pdf_path: Path) -> List[Dict]:
        """Extraer tablas usando OCR"""
        tables = []
        
        try:
            # Convertir PDF a imágenes
            images = convert_from_path(str(pdf_path), dpi=200)
            
            for page_num, image in enumerate(images, 1):
                # Aplicar OCR
                text = pytesseract.image_to_string(image)
                
                # Buscar patrones de tabla
                table_data = self._detect_table_in_text(text)
                
                if table_data:
                    for table in table_data:
                        tables.append({
                            'df': table,
                            'page': page_num,
                            'accuracy': 0.6  # OCR es menos preciso
                        })
        
        except Exception as e:
            logger.error(f"Error en extracción OCR: {e}")
        
        return tables
    
    def _detect_table_in_text(self, text: str) -> List[pd.DataFrame]:
        """Detectar y parsear tablas en texto OCR"""
        tables = []
        
        # Buscar líneas que parezcan filas de tabla
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            # Detectar si la línea parece una fila de tabla
            if self._is_table_row(line):
                table_lines.append(line)
                in_table = True
            elif in_table and not line.strip():
                # Fin de tabla
                if len(table_lines) > 2:  # Al menos encabezado y una fila
                    df = self._parse_table_lines(table_lines)
                    if df is not None:
                        tables.append(df)
                table_lines = []
                in_table = False
        
        # Procesar última tabla si existe
        if len(table_lines) > 2:
            df = self._parse_table_lines(table_lines)
            if df is not None:
                tables.append(df)
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """Determinar si una línea parece ser parte de una tabla"""
        # Buscar patrones comunes en tablas
        # - Múltiples espacios o tabulaciones
        # - Caracteres separadores como |
        # - Patrones numéricos regulares
        
        if not line.strip():
            return False
        
        # Contar separadores
        pipe_count = line.count('|')
        tab_count = line.count('\t')
        multi_space = len(re.findall(r'\s{2,}', line))
        
        return pipe_count > 1 or tab_count > 1 or multi_space > 2
    
    def _parse_table_lines(self, lines: List[str]) -> Optional[pd.DataFrame]:
        """Parsear líneas de texto en DataFrame"""
        try:
            # Detectar separador más común
            separator = self._detect_separator(lines)
            
            # Parsear datos
            data = []
            for line in lines:
                if separator == 'whitespace':
                    row = re.split(r'\s{2,}', line.strip())
                else:
                    row = [cell.strip() for cell in line.split(separator)]
                
                if row:
                    data.append(row)
            
            if len(data) > 1:
                # Primera fila como encabezados
                df = pd.DataFrame(data[1:], columns=data[0])
                return df
                
        except Exception as e:
            logger.debug(f"Error parseando tabla: {e}")
        
        return None
    
    def _detect_separator(self, lines: List[str]) -> str:
        """Detectar el separador usado en la tabla"""
        # Contar ocurrencias de posibles separadores
        separators = {
            '|': 0,
            '\t': 0,
            'whitespace': 0
        }
        
        for line in lines[:5]:  # Analizar primeras líneas
            separators['|'] += line.count('|')
            separators['\t'] += line.count('\t')
            separators['whitespace'] += len(re.findall(r'\s{2,}', line))
        
        # Retornar el más común
        return max(separators, key=separators.get)
    
    def _process_table(self, table_data: Dict, index: int, 
                      manual_name: str, output_dir: Path, 
                      extraction_method: str) -> Dict:
        """Procesar y guardar una tabla extraída"""
        df = table_data['df']
        page = table_data.get('page', 'unknown')
        accuracy = table_data.get('accuracy', 0)
        
        # Limpiar DataFrame
        df = self._clean_dataframe(df)
        
        # Guardar como CSV
        csv_filename = f"{manual_name}_table_{index}_p{page}.csv"
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Guardar como Excel para mejor preservación de formato
        excel_filename = f"{manual_name}_table_{index}_p{page}.xlsx"
        excel_path = output_dir / excel_filename
        df.to_excel(excel_path, index=False)
        
        # Convertir a texto para embeddings
        table_text = self._dataframe_to_text(df)
        
        metadata = {
            'manual_name': manual_name,
            'page_number': page,
            'table_index': index,
            'content_type': 'table',
            'extraction_method': extraction_method,
            'accuracy': accuracy,
            'csv_path': str(csv_path),
            'excel_path': str(excel_path),
            'table_text': table_text,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y normalizar DataFrame"""
        # Eliminar filas y columnas completamente vacías
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Limpiar nombres de columnas
        df.columns = [str(col).strip() for col in df.columns]
        
        # Reemplazar valores NaN con strings vacíos
        df = df.fillna('')
        
        # Eliminar espacios extras
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        return df
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convertir DataFrame a texto para embeddings"""
        # Incluir nombres de columnas
        text_parts = [f"Columnas: {', '.join(df.columns)}"]
        
        # Agregar filas
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if val])
            text_parts.append(row_text)
        
        return "\n".join(text_parts)