from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Resultado de recuperación estructurado"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'text' o 'image'
    highlights: Optional[List[str]] = None

class AdvancedRetrieval:
    """Sistema avanzado de recuperación con re-ranking y filtrado inteligente"""
    
    def __init__(self, vector_manager, use_reranking: bool = True):
        self.vector_manager = vector_manager
        self.use_reranking = use_reranking
        
        if use_reranking:
            # Modelo de re-ranking para mejorar relevancia
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def hybrid_search(self, 
                     query: str,
                     n_results: int = 10,
                     search_mode: str = 'all',  # 'all', 'text', 'images', 'tables'
                     manual_filter: Optional[str] = None,
                     page_range: Optional[Tuple[int, int]] = None,
                     min_score: float = 0.5) -> List[RetrievalResult]:
        """
        Búsqueda híbrida con múltiples estrategias
        """
        results = []
        
        # Construir filtros
        filters = self._build_filters(manual_filter, page_range, search_mode)
        
        # Búsqueda en texto
        if search_mode in ['all', 'text', 'tables']:
            text_results = self._search_text(query, n_results * 2, filters)
            results.extend(text_results)
        
        # Búsqueda en imágenes
        if search_mode in ['all', 'images']:
            image_results = self._search_images(query, n_results // 2, manual_filter)
            results.extend(image_results)
        
        # Re-ranking si está habilitado
        if self.use_reranking and results:
            results = self._rerank_results(query, results, n_results)
        
        # Filtrar por score mínimo
        results = [r for r in results if r.score >= min_score]
        
        # Limitar resultados
        return results[:n_results]
    
    def contextual_search(self,
                         query: str,
                         context_window: int = 2,
                         n_results: int = 5,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Búsqueda con contexto expandido (chunks anteriores y posteriores)
        """
        # Búsqueda inicial
        initial_results = self.hybrid_search(query, n_results, **kwargs)
        
        # Expandir contexto para cada resultado
        expanded_results = []
        
        for result in initial_results:
            expanded = self._expand_context(result, context_window)
            expanded_results.append({
                'main_result': result,
                'context': expanded,
                'full_text': self._merge_context(result, expanded)
            })
        
        return expanded_results
    
    def multi_query_search(self,
                          queries: List[str],
                          aggregation: str = 'union',  # 'union', 'intersection', 'weighted'
                          n_results: int = 10,
                          **kwargs) -> List[RetrievalResult]:
        """
        Búsqueda con múltiples queries
        """
        all_results = {}
        
        # Ejecutar cada query
        for query in queries:
            results = self.hybrid_search(query, n_results * 2, **kwargs)
            
            for result in results:
                key = f"{result.metadata.get('manual_name')}_{result.metadata.get('page_number')}_{result.metadata.get('chunk_index', 0)}"
                
                if key not in all_results:
                    all_results[key] = {
                        'result': result,
                        'scores': [],
                        'queries': []
                    }
                
                all_results[key]['scores'].append(result.score)
                all_results[key]['queries'].append(query)
        
        # Agregar resultados según estrategia
        if aggregation == 'union':
            # Todos los resultados únicos
            final_results = [data['result'] for data in all_results.values()]
        
        elif aggregation == 'intersection':
            # Solo resultados que aparecen en todas las queries
            final_results = [
                data['result'] for data in all_results.values()
                if len(data['queries']) == len(queries)
            ]
        
        elif aggregation == 'weighted':
            # Promedio ponderado de scores
            for data in all_results.values():
                data['result'].score = np.mean(data['scores'])
            
            final_results = [data['result'] for data in all_results.values()]
            final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:n_results]
    
    def find_similar_content(self,
                           reference_content: str,
                           n_results: int = 5,
                           exclude_self: bool = True,
                           **kwargs) -> List[RetrievalResult]:
        """
        Encontrar contenido similar a un texto de referencia
        """
        # Usar el contenido como query
        results = self.hybrid_search(reference_content, n_results + 1, **kwargs)
        
        # Excluir el mismo contenido si es necesario
        if exclude_self:
            results = [r for r in results if r.content != reference_content]
        
        return results[:n_results]
    
    def _search_text(self, query: str, n_results: int, filters: Dict) -> List[RetrievalResult]:
        """Búsqueda en colección de texto"""
        results = self.vector_manager.search(
            query,
            n_results=n_results,
            filter_manual=filters.get('manual_name'),
            filter_page=filters.get('page_number'),
            content_type=filters.get('content_type')
        )
        
        retrieval_results = []
        
        for result in results.get('text_results', []):
            retrieval_results.append(RetrievalResult(
                content=result['content'],
                metadata=result['metadata'],
                score=1.0 - result.get('distance', 0),  # Convertir distancia a score
                source='text'
            ))
        
        return retrieval_results
    
    def _search_images(self, query: str, n_results: int, 
                      manual_filter: Optional[str]) -> List[RetrievalResult]:
        """Búsqueda en colección de imágenes"""
        results = self.vector_manager.search(
            query,
            n_results=n_results,
            filter_manual=manual_filter,
            content_type='image'
        )
        
        retrieval_results = []
        
        for result in results.get('image_results', []):
            retrieval_results.append(RetrievalResult(
                content=f"Imagen: {result['metadata'].get('image_filename', 'Unknown')}",
                metadata=result['metadata'],
                score=1.0 - result.get('distance', 0),
                source='image'
            ))
        
        return retrieval_results
    
    def _rerank_results(self, query: str, results: List[RetrievalResult], 
                       n_results: int) -> List[RetrievalResult]:
        """Re-rankear resultados usando modelo de cross-encoding"""
        if not results:
            return results
        
        # Preparar pares query-documento
        pairs = [[query, r.content] for r in results]
        
        # Calcular scores de re-ranking
        rerank_scores = self.reranker.predict(pairs)
        
        # Combinar scores (70% rerank, 30% original)
        for i, result in enumerate(results):
            combined_score = 0.7 * rerank_scores[i] + 0.3 * result.score
            result.score = combined_score
        
        # Ordenar por nuevo score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:n_results]
    
    def _build_filters(self, manual_filter: Optional[str], 
                      page_range: Optional[Tuple[int, int]],
                      search_mode: str) -> Dict:
        """Construir filtros para búsqueda"""
        filters = {}
        
        if manual_filter:
            filters['manual_name'] = manual_filter
        
        if page_range:
            # ChromaDB no soporta rangos directamente, necesitamos otra estrategia
            # Por ahora, usamos página específica si el rango es de una sola página
            if page_range[0] == page_range[1]:
                filters['page_number'] = page_range[0]
        
        if search_mode == 'tables':
            filters['content_type'] = 'table'
        
        return filters
    
    def _expand_context(self, result: RetrievalResult, 
                       window: int) -> List[RetrievalResult]:
        """Expandir contexto obteniendo chunks adyacentes"""
        metadata = result.metadata
        manual = metadata.get('manual_name')
        page = metadata.get('page_number')
        chunk_index = metadata.get('chunk_index', 0)
        
        context_results = []
        
        # Buscar chunks anteriores y posteriores
        for offset in range(-window, window + 1):
            if offset == 0:
                continue  # Skip el chunk actual
            
            target_index = chunk_index + offset
            
            # Buscar chunk específico
            # Esto requeriría una implementación más sofisticada en vector_manager
            # Por ahora, es un placeholder
            
        return context_results
    
    def _merge_context(self, main_result: RetrievalResult, 
                      context: List[RetrievalResult]) -> str:
        """Combinar resultado principal con contexto"""
        # Ordenar por índice de chunk
        all_results = context + [main_result]
        all_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
        
        # Combinar textos
        merged = "\n".join([r.content for r in all_results])
        
        return merged

class SemanticCache:
    """Cache semántico para queries frecuentes"""
    
    def __init__(self, embedding_model, threshold: float = 0.95):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.cache = {}
    
    def get(self, query: str) -> Optional[List[RetrievalResult]]:
        """Obtener resultados del cache si existe query similar"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        for cached_query, (cached_embedding, results) in self.cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                logger.info(f"Cache hit for query similar to: {cached_query}")
                return results
        
        return None
    
    def set(self, query: str, results: List[RetrievalResult]):
        """Guardar resultados en cache"""
        query_embedding = self.embedding_model.encode([query])[0]
        self.cache[query] = (query_embedding, results)
        
        # Limitar tamaño del cache
        if len(self.cache) > 100:
            # Eliminar entrada más antigua
            oldest = next(iter(self.cache))
            del self.cache[oldest]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcular similitud coseno"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Funciones de utilidad para recuperación
def format_results_for_llm(results: List[RetrievalResult], 
                          max_tokens: int = 2000) -> str:
    """Formatear resultados para enviar a LLM"""
    formatted = []
    current_tokens = 0
    
    for i, result in enumerate(results):
        # Estimar tokens (aproximadamente 4 caracteres por token)
        result_text = f"\n[Fuente {i+1}] Manual: {result.metadata.get('manual_name')}, " \
                     f"Página: {result.metadata.get('page_number')}\n" \
                     f"{result.content}\n"
        
        estimated_tokens = len(result_text) // 4
        
        if current_tokens + estimated_tokens > max_tokens:
            break
        
        formatted.append(result_text)
        current_tokens += estimated_tokens
    
    return "\n".join(formatted)

def group_results_by_manual(results: List[RetrievalResult]) -> Dict[str, List[RetrievalResult]]:
    """Agrupar resultados por manual"""
    grouped = {}
    
    for result in results:
        manual = result.metadata.get('manual_name', 'Unknown')
        
        if manual not in grouped:
            grouped[manual] = []
        
        grouped[manual].append(result)
    
    return grouped

def deduplicate_results(results: List[RetrievalResult], 
                       similarity_threshold: float = 0.9) -> List[RetrievalResult]:
    """Eliminar resultados duplicados o muy similares"""
    unique_results = []
    
    for result in results:
        is_duplicate = False
        
        for unique in unique_results:
            # Comparar contenido
            if result.content == unique.content:
                is_duplicate = True
                break
            
            # Aquí podrías agregar comparación semántica si es necesario
        
        if not is_duplicate:
            unique_results.append(result)
    
    return unique_results