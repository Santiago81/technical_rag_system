"""
Sistema de chunking inteligente que preserva contexto semántico
"""
from typing import List, Dict, Optional, Tuple, Iterator
import re
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

@dataclass
class ChunkMetadata:
    """Metadatos enriquecidos para cada chunk"""
    document_id: int
    chunk_index: int
    start_page: int
    end_page: int
    start_char: int
    end_char: int
    section: Optional[str]
    subsection: Optional[str]
    chunk_type: str  # 'text', 'table_context', 'diagram_context'
    importance_score: float
    semantic_density: float
    technical_terms: List[str]
    references: List[str]  # Referencias a figuras, tablas, etc.

class IntelligentChunker:
    """Chunker adaptativo que preserva contexto técnico"""
    
    def __init__(self, 
                 base_chunk_size: int = 512,
                 overlap_size: int = 128,
                 min_chunk_size: int = 256,
                 max_chunk_size: int = 1024):
        self.base_chunk_size = base_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Modelo para calcular semantic density
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Patrones para detectar estructura
        self.section_pattern = re.compile(
            r'^(?:(?:\d+\.?)+\s*)?(?:Chapter|Section|Capítulo|Sección)\s+\d+',
            re.MULTILINE | re.IGNORECASE
        )
        
        self.reference_pattern = re.compile(
            r'(?:(?:see|ver|véase)\s+)?(?:Figure|Fig\.|Figura|Table|Tabla)\s+\d+',
            re.IGNORECASE
        )
        
        # Términos técnicos comunes
        self.load_technical_vocabulary()
    
    def chunk_document(self, text: str, metadata: Dict) -> Iterator[Tuple[str, ChunkMetadata]]:
        """Generar chunks con contexto preservado"""
        
        # 1. Detectar estructura del documento
        structure = self._detect_document_structure(text)
        
        # 2. Segmentar por secciones
        sections = self._segment_by_sections(text, structure)
        
        # 3. Aplicar chunking adaptativo por sección
        for section in sections:
            yield from self._chunk_section(
                section['text'],
                section['metadata'],
                metadata
            )
    
    def _detect_document_structure(self, text: str) -> Dict:
        """Detectar estructura jerárquica del documento"""
        structure = {
            'sections': [],
            'references': [],
            'technical_density': 0.0
        }
        
        # Encontrar secciones
        for match in self.section_pattern.finditer(text):
            structure['sections'].append({
                'title': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Encontrar referencias
        for match in self.reference_pattern.finditer(text):
            structure['references'].append({
                'reference': match.group(0),
                'position': match.start()
            })
        
        # Calcular densidad técnica
        words = word_tokenize(text.lower())
        technical_count = sum(1 for word in words if word in self.technical_vocab)
        structure['technical_density'] = technical_count / len(words) if words else 0
        
        return structure
    
    def _chunk_section(self, 
                      section_text: str, 
                      section_metadata: Dict,
                      doc_metadata: Dict) -> Iterator[Tuple[str, ChunkMetadata]]:
        """Aplicar chunking inteligente a una sección"""
        
        # Ajustar tamaño según tipo de contenido
        chunk_size = self._determine_optimal_chunk_size(
            section_text,
            section_metadata
        )
        
        sentences = sent_tokenize(section_text)
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # Verificar si agregar la oración excede el tamaño
            if current_size + sentence_size > chunk_size and current_chunk:
                # Verificar integridad semántica antes de cortar
                if self._should_keep_together(current_chunk[-1], sentence):
                    # Extender chunk para mantener contexto
                    current_chunk.append(sentence)
                    current_size += sentence_size
                
                # Generar chunk
                chunk_text = ' '.join(current_chunk)
                metadata = self._create_chunk_metadata(
                    chunk_text,
                    chunk_index,
                    section_metadata,
                    doc_metadata
                )
                
                yield chunk_text, metadata
                
                # Preparar siguiente chunk con overlap
                overlap_sentences = self._calculate_overlap(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Último chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = self._create_chunk_metadata(
                chunk_text,
                chunk_index,
                section_metadata,
                doc_metadata
            )
            yield chunk_text, metadata
    
    def _determine_optimal_chunk_size(self, text: str, metadata: Dict) -> int:
        """Determinar tamaño óptimo de chunk según contenido"""
        
        # Detectar tipo de contenido
        if self._contains_table_references(text):
            # Chunks más grandes para preservar contexto de tablas
            return min(self.base_chunk_size * 1.5, self.max_chunk_size)
        
        elif self._contains_technical_procedures(text):
            # Chunks medianos para procedimientos
            return self.base_chunk_size
        
        elif metadata.get('technical_density', 0) > 0.3:
            # Chunks más grandes para contenido muy técnico
            return min(self.base_chunk_size * 1.2, self.max_chunk_size)
        
        else:
            # Tamaño estándar
            return self.base_chunk_size
    
    def _should_keep_together(self, last_sentence: str, next_sentence: str) -> bool:
        """Determinar si dos oraciones deben mantenerse juntas"""
        
        # Reglas para mantener contexto
        rules = [
            # Lista numerada o con viñetas
            (r'^\s*\d+\.\s*', r'^\s*\d+\.\s*'),
            (r'^\s*[a-z]\)\s*', r'^\s*[a-z]\)\s*'),
            (r'^\s*[-•]\s*', r'^\s*[-•]\s*'),
            
            # Referencia a figura/tabla en oración siguiente
            (r'.*', r'^\s*(?:See|Ver|Véase)\s+(?:Figure|Fig\.|Figura|Table|Tabla)'),
            
            # Continuación de procedimiento
            (r'.*:\s*$', r'.*'),  # Termina en dos puntos
            (r'.*(?:following|siguiente|siguientes?)\s*:\s*$', r'.*'),
        ]
        
        for last_pattern, next_pattern in rules:
            if (re.match(last_pattern, last_sentence) and 
                re.match(next_pattern, next_sentence)):
                return True
        
        return False
    
    def _calculate_overlap(self, sentences: List[str]) -> List[str]:
        """Calcular oraciones de overlap para contexto"""
        if not sentences:
            return []
        
        # Tomar últimas N palabras como overlap
        total_words = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            words_in_sentence = len(sentence.split())
            if total_words + words_in_sentence <= self.overlap_size:
                overlap_sentences.insert(0, sentence)
                total_words += words_in_sentence
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_metadata(self,
                              chunk_text: str,
                              chunk_index: int,
                              section_metadata: Dict,
                              doc_metadata: Dict) -> ChunkMetadata:
        """Crear metadatos enriquecidos para el chunk"""
        
        # Extraer términos técnicos
        technical_terms = self._extract_technical_terms(chunk_text)
        
        # Extraer referencias
        references = [match.group(0) for match in self.reference_pattern.finditer(chunk_text)]
        
        # Calcular importancia
        importance_score = self._calculate_importance_score(
            chunk_text,
            technical_terms,
            references
        )
        
        # Calcular densidad semántica
        semantic_density = self._calculate_semantic_density(chunk_text)
        
        return ChunkMetadata(
            document_id=doc_metadata.get('document_id'),
            chunk_index=chunk_index,
            start_page=section_metadata.get('start_page', 1),
            end_page=section_metadata.get('end_page', 1),
            start_char=section_metadata.get('start_char', 0),
            end_char=section_metadata.get('end_char', len(chunk_text)),
            section=section_metadata.get('section'),
            subsection=section_metadata.get('subsection'),
            chunk_type=self._determine_chunk_type(chunk_text),
            importance_score=importance_score,
            semantic_density=semantic_density,
            technical_terms=technical_terms,
            references=references
        )
    
    def _calculate_importance_score(self, 
                                  text: str, 
                                  technical_terms: List[str],
                                  references: List[str]) -> float:
        """Calcular score de importancia del chunk"""
        score = 1.0
        
        # Boost por términos técnicos
        if technical_terms:
            score *= (1 + min(len(technical_terms) / 10, 0.5))
        
        # Boost por referencias
        if references:
            score *= 1.2
        
        # Boost por palabras clave importantes
        important_keywords = [
            'warning', 'caution', 'danger', 'important',
            'advertencia', 'precaución', 'peligro', 'importante',
            'must', 'debe', 'required', 'requerido'
        ]
        
        text_lower = text.lower()
        for keyword in important_keywords:
            if keyword in text_lower:
                score *= 1.3
                break
        
        # Penalizar chunks muy cortos
        if len(text.split()) < 50:
            score *= 0.8
        
        return min(score, 10.0)
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calcular densidad semántica usando embeddings"""
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 1.0
        
        # Calcular embeddings de oraciones
        embeddings = self.embedding_model.encode(sentences)
        
        # Calcular diversidad semántica
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(similarity)
        
        # Mayor diversidad = mayor densidad de información
        avg_similarity = np.mean(similarities)
        semantic_density = 1 - avg_similarity
        
        return semantic_density
    
    def _determine_chunk_type(self, text: str) -> str:
        """Determinar tipo de contenido del chunk"""
        
        # Detectar tablas
        if 'Table' in text or 'Tabla' in text or '|' in text:
            return 'table_context'
        
        # Detectar referencias a diagramas
        elif 'Figure' in text or 'Figura' in text or 'Diagram' in text:
            return 'diagram_context'
        
        # Detectar procedimientos
        elif re.search(r'Step \d+|Paso \d+|^\s*\d+\.', text, re.MULTILINE):
            return 'procedure'
        
        # Detectar especificaciones
        elif re.search(r'\d+\s*(?:mm|cm|m|kg|V|A|W|°C)', text):
            return 'specification'
        
        else:
            return 'text'
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extraer términos técnicos del texto"""
        words = word_tokenize(text.lower())
        technical_terms = []
        
        for word in words:
            if word in self.technical_vocab and len(word) > 3:
                technical_terms.append(word)
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_terms = []
        for term in technical_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Limitar a 10 términos
    
    def _contains_table_references(self, text: str) -> bool:
        """Verificar si el texto contiene referencias a tablas"""
        return bool(re.search(r'Table\s+\d+|Tabla\s+\d+|following table|siguiente tabla', 
                            text, re.IGNORECASE))
    
    def _contains_technical_procedures(self, text: str) -> bool:
        """Verificar si el texto contiene procedimientos técnicos"""
        procedure_indicators = [
            r'Step \d+', r'Paso \d+',
            r'procedure', r'procedimiento',
            r'follow these steps', r'siga estos pasos',
            r'^\s*\d+\.\s+\w+', r'^\s*[a-z]\)\s+\w+'
        ]
        
        for pattern in procedure_indicators:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False
    
    def load_technical_vocabulary(self):
        """Cargar vocabulario técnico específico del dominio"""
        # En producción, cargar desde archivo o base de datos
        self.technical_vocab = {
            # Mecánica
            'torque', 'bearing', 'gear', 'shaft', 'pulley', 'belt',
            'piston', 'cylinder', 'valve', 'spring', 'clutch',
            
            # Electricidad
            'voltage', 'current', 'resistance', 'capacitor', 'inductor',
            'transformer', 'relay', 'switch', 'fuse', 'circuit',
            
            # Hidráulica
            'pump', 'hydraulic', 'pressure', 'flow', 'cylinder',
            'valve', 'filter', 'reservoir', 'hose', 'fitting',
            
            # Español
            'par', 'rodamiento', 'engranaje', 'eje', 'polea',
            'pistón', 'cilindro', 'válvula', 'resorte', 'embrague',
            'voltaje', 'corriente', 'resistencia', 'condensador',
            'bomba', 'hidráulico', 'presión', 'flujo', 'manguera'
        }
    
    def _segment_by_sections(self, text: str, structure: Dict) -> List[Dict]:
        """Segmentar texto por secciones detectadas"""
        sections = []
        section_starts = [s['start'] for s in structure['sections']]
        section_starts.append(len(text))  # Final del documento
        
        for i in range(len(section_starts) - 1):
            section_text = text[section_starts[i]:section_starts[i + 1]]
            
            sections.append({
                'text': section_text,
                'metadata': {
                    'section': structure['sections'][i]['title'] if i < len(structure['sections']) else None,
                    'start_char': section_starts[i],
                    'end_char': section_starts[i + 1],
                    'technical_density': structure['technical_density']
                }
            })
        
        return sections