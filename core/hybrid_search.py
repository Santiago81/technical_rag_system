"""
Sistema de búsqueda híbrida optimizado
"""
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import sqlite3
from sentence_transformers import CrossEncoder
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Resultado de búsqueda unificado"""
    chunk_id: int
    document_id: int
    document_name: str
    content: str
    score: float
    search_type: str  # 'keyword', 'semantic', 'hybrid'
    metadata: Dict[str, Any]
    highlights: List[str]
    context: Optional[Dict[str, str]] = None

class HybridSearchEngine:
    """Motor de búsqueda híbrida con re-ranking"""
    
    def __init__(self,
                 db_path: str,
                 embedding_pipeline: 'OptimizedEmbeddingPipeline',
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        self.db_path = db_path
        self.embedding_pipeline = embedding_pipeline
        self.reranker = CrossEncoder(reranker_model)
        
        # Cache de búsquedas frecuentes
        self.search_cache = {}
        
        # Vectorizador TF-IDF para búsqueda por keywords
        self.tfidf_vectorizer = None
        self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Inicializar vectorizador TF-IDF con corpus"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Cargar todos los chunks para TF-IDF
            cursor = conn.execute("""
                SELECT chunk_text FROM chunks 
                WHERE chunk_text IS NOT NULL
                LIMIT 10000
            """)
            
            corpus = [row['chunk_text'] for row in cursor]
            
            if corpus:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english'  # Ajustar para español
                )
                self.tfidf_vectorizer.fit(corpus)
    
    async def search(self,
                    query: str,
                    k: int = 10,
                    search_mode: str = 'hybrid',
                    filters: Optional[Dict] = None,
                    expand_context: bool = True) -> List[SearchResult]:
        """Búsqueda principal con múltiples estrategias"""
        
        # Verificar cache
        cache_key = self._generate_cache_key(query, filters)
        if cache_key in self.search_cache:
            logger.debug(f"Cache hit para query: {query[:50]}...")
            return self.search_cache[cache_key]
        
        results = []
        
        if search_mode in ['hybrid', 'keyword']:
            # Búsqueda por keywords
            keyword_results = await self._keyword_search(query, k * 2, filters)
            results.extend(keyword_results)
        
        if search_mode in ['hybrid', 'semantic']:
            # Búsqueda semántica
            semantic_results = await self._semantic_search(query, k * 2, filters)
            results.extend(semantic_results)
        
        # Eliminar duplicados y fusionar scores
        results = self._merge_and_deduplicate(results)
        
        # Re-ranking con cross-encoder
        if len(results) > k:
            results = await self._rerank_results(query, results, k)
        
        # Expandir contexto si se solicita
        if expand_context:
            results = await self._expand_context(results)
        
        # Limitar a k resultados
        results = results[:k]
        
        # Guardar en cache
        self.search_cache[cache_key] = results
        
        # Actualizar estadísticas de acceso
        self._update_access_stats([r.chunk_id for r in results])
        
        return results
    
    async def _keyword_search(self,
                            query: str,
                            k: int,
                            filters: Optional[Dict]) -> List[SearchResult]:
        """Búsqueda por keywords usando FTS5 y TF-IDF"""
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Construir query FTS5
            fts_query = self._build_fts_query(query)
            
            # Construir cláusulas WHERE
            where_clauses = ["1=1"]
            params = [fts_query]
            
            if filters:
                if 'document_id' in filters:
                    where_clauses.append("c.document_id = ?")
                    params.append(filters['document_id'])
                
                if 'document_type' in filters:
                    where_clauses.append("d.document_type = ?")
                    params.append(filters['document_type'])
                
                if 'date_range' in filters:
                    where_clauses.append("d.created_at BETWEEN ? AND ?")
                    params.extend(filters['date_range'])
            
            where_clause = " AND ".join(where_clauses)
            params.append(k)
            
            # Ejecutar búsqueda
            cursor = conn.execute(f"""
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    d.name as document_name,
                    c.chunk_text,
                    c.metadata,
                    snippet(chunks_fts, -1, '<mark>', '</mark>', '...', 20) as snippet,
                    rank as fts_rank
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ?
                    AND {where_clause}
                ORDER BY rank
                LIMIT ?
            """, params)
            
            for row in cursor:
                # Calcular score combinado
                fts_score = 1.0 / (1.0 + abs(row['fts_rank']))
                
                # Boost por TF-IDF si está disponible
                tfidf_score = self._calculate_tfidf_score(query, row['chunk_text'])
                combined_score = 0.7 * fts_score + 0.3 * tfidf_score
                
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    document_name=row['document_name'],
                    content=row['chunk_text'],
                    score=combined_score,
                    search_type='keyword',
                    metadata=self._parse_json(row['metadata']),
                    highlights=[row['snippet']]
                ))
        
        return results
    
    async def _semantic_search(self,
                             query: str,
                             k: int,
                             filters: Optional[Dict]) -> List[SearchResult]:
        """Búsqueda semántica usando embeddings"""
        
        # Generar embedding de la query
        query_embedding = await self.embedding_pipeline.embed_texts_batch([query])
        query_embedding = query_embedding[0]
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Construir filtros
            where_clauses = ["c.embedding IS NOT NULL"]
            params = []
            
            if filters:
                if 'document_id' in filters:
                    where_clauses.append("c.document_id = ?")
                    params.append(filters['document_id'])
                
                if 'document_type' in filters:
                    where_clauses.append("d.document_type = ?")
                    params.append(filters['document_type'])
            
            where_clause = " AND ".join(where_clauses)
            
            # Obtener candidatos
            cursor = conn.execute(f"""
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    d.name as document_name,
                    c.chunk_text,
                    c.embedding,
                    c.importance_score,
                    c.metadata
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE {where_clause}
            """, params)
            
            # Calcular similitudes
            candidates = []
            for row in cursor:
                # Reconstruir embedding
                chunk_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                
                # Calcular similitud coseno
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                # Score ponderado por importancia
                weighted_score = similarity * row['importance_score']
                
                candidates.append({
                    'row': row,
                    'similarity': similarity,
                    'score': weighted_score
                })
            
            # Ordenar por score y tomar top k
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            for candidate in candidates[:k]:
                row = candidate['row']
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    document_name=row['document_name'],
                    content=row['chunk_text'],
                    score=candidate['score'],
                    search_type='semantic',
                    metadata=self._parse_json(row['metadata']),
                    highlights=[]
                ))
        
        return results
    
    async def _rerank_results(self,
                            query: str,
                            results: List[SearchResult],
                            k: int) -> List[SearchResult]:
        """Re-rankear resultados usando cross-encoder"""
        
        # Preparar pares query-documento
        pairs = [[query, r.content] for r in results]
        
        # Calcular scores con cross-encoder
        rerank_scores = self.reranker.predict(pairs)
        
        # Combinar scores
        for i, result in enumerate(results):
            # Peso adaptativo según tipo de búsqueda
            if result.search_type == 'semantic':
                # Mayor peso al reranker para búsquedas semánticas
                result.score = 0.3 * result.score + 0.7 * rerank_scores[i]
            else:
                # Balanceado para búsquedas por keyword
                result.score = 0.5 * result.score + 0.5 * rerank_scores[i]
        
        # Reordenar
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:k]
    
    async def _expand_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """Expandir contexto de los resultados"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            for result in results:
                # Obtener chunks adyacentes
                cursor = conn.execute("""
                    SELECT 
                        chunk_text,
                        chunk_index
                    FROM chunks
                    WHERE document_id = ?
                        AND chunk_index BETWEEN ? AND ?
                    ORDER BY chunk_index
                """, (
                    result.document_id,
                    result.metadata.get('chunk_index', 0) - 1,
                    result.metadata.get('chunk_index', 0) + 1
                ))
                
                context = {
                    'before': None,
                    'after': None
                }
                
                for row in cursor:
                    if row['chunk_index'] < result.metadata.get('chunk_index', 0):
                        context['before'] = row['chunk_text']
                    elif row['chunk_index'] > result.metadata.get('chunk_index', 0):
                        context['after'] = row['chunk_text']
                
                result.context = context
        
        return results
    
    def _merge_and_deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Fusionar y deduplicar resultados"""
        
        # Agrupar por chunk_id
        merged = {}
        
        for result in results:
            if result.chunk_id not in merged:
                merged[result.chunk_id] = result
            else:
                # Fusionar scores
                existing = merged[result.chunk_id]
                if result.search_type != existing.search_type:
                    # Resultado aparece en ambas búsquedas
                    existing.score = (existing.score + result.score) / 2
                    existing.search_type = 'hybrid'
                    
                    # Combinar highlights
                    existing.highlights.extend(result.highlights)
        
        return list(merged.values())
    
    def _build_fts_query(self, query: str) -> str:
        """Construir query optimizada para FTS5"""
        # Tokenizar y limpiar
        tokens = query.lower().split()
        
        # Construir query con operadores
        fts_parts = []
        
        for token in tokens:
            if len(token) > 2:
                # Añadir variantes con prefix matching
                fts_parts.append(f'"{token}"')
                fts_parts.append(f'{token}*')
        
        return ' OR '.join(fts_parts)
    
    def _calculate_tfidf_score(self, query: str, document: str) -> float:
        """Calcular score TF-IDF entre query y documento"""
        if not self.tfidf_vectorizer:
            return 0.0
        
        try:
            # Vectorizar query y documento
            query_vec = self.tfidf_vectorizer.transform([query])
            doc_vec = self.tfidf_vectorizer.transform([document])
            
            # Calcular similitud coseno
            similarity = (query_vec * doc_vec.T).toarray()[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _update_access_stats(self, chunk_ids: List[int]):
        """Actualizar estadísticas de acceso"""
        if not chunk_ids:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Actualizar contador y timestamp
            placeholders = ','.join('?' * len(chunk_ids))
            conn.execute(f"""
                UPDATE chunks
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
            """, chunk_ids)
            conn.commit()
    
    def _generate_cache_key(self, query: str, filters: Optional[Dict]) -> str:
        """Generar clave de cache"""
        import hashlib
        
        key_parts = [query]
        if filters:
            key_parts.append(str(sorted(filters.items())))
        
        combined = '|'.join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _parse_json(self, json_str: Optional[str]) -> Dict:
        """Parsear JSON de forma segura"""
        if not json_str:
            return {}
        
        try:
            import json
            return json.loads(json_str)
        except:
            return {}
    
    def clear_cache(self):
        """Limpiar cache de búsquedas"""
        self.search_cache.clear()
        logger.info("Cache de búsquedas limpiado")