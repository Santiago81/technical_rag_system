from typing import Dict, List, Optional, Set, Tuple
import json
from pathlib import Path
from collections import defaultdict
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class IndexingSystem:
    """Sistema avanzado de indexación para búsquedas eficientes"""
    
    def __init__(self, vector_manager):
        self.vector_manager = vector_manager
        
        # Índices principales
        self.indices = {
            'by_manual': defaultdict(list),
            'by_page': defaultdict(list),
            'by_section': defaultdict(list),
            'by_chapter': defaultdict(list),
            'by_content_type': defaultdict(list),
            # Nuevos índices para retrieval avanzado
            'by_date': defaultdict(list),
            'by_chunk_sequence': defaultdict(list),
            'by_semantic_cluster': defaultdict(list)
        }
        
        # Índices inversos para búsquedas rápidas
        self.inverted_indices = {
            'doc_to_manual': {},
            'doc_to_page': {},
            'doc_to_section': {},
            'doc_to_type': {}
        }
        
        # Mapa de adyacencia para contexto expandido
        self.adjacency_map = defaultdict(dict)
        
        # Cache de metadatos
        self.metadata_cache = {}
    
    def build_indices(self, include_semantic_clustering: bool = False):
        """Construir todos los índices"""
        logger.info("Construyendo índices...")
        
        # Obtener todos los documentos de texto
        all_docs = self.vector_manager.collection.get()
        
        # Construir índices básicos
        self._build_basic_indices(all_docs)
        
        # Construir mapa de adyacencia para recuperación contextual
        self._build_adjacency_map(all_docs)
        
        # Construir índices inversos
        self._build_inverted_indices(all_docs)
        
        # Opcionalmente, construir clusters semánticos
        if include_semantic_clustering:
            self._build_semantic_clusters(all_docs)
        
        # Procesar también imágenes
        self._index_images()
        
        logger.info("Índices construidos exitosamente")
        logger.info(f"  - Documentos indexados: {len(all_docs['ids'])}")
        logger.info(f"  - Manuales únicos: {len(set(self.inverted_indices['doc_to_manual'].values()))}")
    
    def _build_basic_indices(self, all_docs: Dict):
        """Construir índices básicos"""
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i]
            
            # Cache de metadatos
            self.metadata_cache[doc_id] = metadata
            
            # Indexar por manual
            manual = metadata.get('manual_name')
            if manual:
                self.indices['by_manual'][manual].append(doc_id)
                self.inverted_indices['doc_to_manual'][doc_id] = manual
            
            # Indexar por página
            page = metadata.get('page_number')
            if page:
                page_key = f"{manual}_{page}"
                self.indices['by_page'][page_key].append(doc_id)
                self.inverted_indices['doc_to_page'][doc_id] = page
            
            # Indexar por sección
            section = metadata.get('section')
            if section:
                section_key = f"{manual}_{section}"
                self.indices['by_section'][section_key].append(doc_id)
                self.inverted_indices['doc_to_section'][doc_id] = section
            
            # Indexar por capítulo
            chapter = metadata.get('chapter')
            if chapter:
                self.indices['by_chapter'][f"{manual}_{chapter}"].append(doc_id)
            
            # Indexar por tipo de contenido
            content_type = metadata.get('content_type', 'text')
            self.indices['by_content_type'][content_type].append(doc_id)
            self.inverted_indices['doc_to_type'][doc_id] = content_type
            
            # Indexar por fecha (si existe)
            timestamp = metadata.get('timestamp')
            if timestamp:
                date_key = timestamp.split('T')[0]  # YYYY-MM-DD
                self.indices['by_date'][date_key].append(doc_id)
            
            # Indexar por secuencia de chunks
            chunk_index = metadata.get('chunk_index')
            if chunk_index is not None and manual and page:
                sequence_key = f"{manual}_{page}_{chunk_index}"
                self.indices['by_chunk_sequence'][sequence_key] = doc_id
    
    def _build_adjacency_map(self, all_docs: Dict):
        """Construir mapa de adyacencia para recuperación de contexto"""
        logger.info("Construyendo mapa de adyacencia...")
        
        # Agrupar documentos por manual y página
        docs_by_location = defaultdict(list)
        
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i]
            manual = metadata.get('manual_name')
            page = metadata.get('page_number')
            chunk_index = metadata.get('chunk_index', 0)
            
            if manual and page is not None:
                location_key = f"{manual}_{page}"
                docs_by_location[location_key].append({
                    'id': doc_id,
                    'chunk_index': chunk_index
                })
        
        # Construir adyacencias
        for location, docs in docs_by_location.items():
            # Ordenar por chunk_index
            docs.sort(key=lambda x: x['chunk_index'])
            
            # Crear enlaces de adyacencia
            for i, doc in enumerate(docs):
                doc_id = doc['id']
                
                # Chunk anterior
                if i > 0:
                    self.adjacency_map[doc_id]['previous'] = docs[i-1]['id']
                
                # Chunk siguiente
                if i < len(docs) - 1:
                    self.adjacency_map[doc_id]['next'] = docs[i+1]['id']
                
                # Chunks cercanos (ventana de 2)
                nearby = []
                for j in range(max(0, i-2), min(len(docs), i+3)):
                    if j != i:
                        nearby.append(docs[j]['id'])
                self.adjacency_map[doc_id]['nearby'] = nearby
    
    def _build_inverted_indices(self, all_docs: Dict):
        """Construir índices inversos para búsquedas rápidas"""
        # Los índices inversos ya se construyen en _build_basic_indices
        pass
    
    def _build_semantic_clusters(self, all_docs: Dict):
        """Construir clusters semánticos basados en embeddings"""
        logger.info("Construyendo clusters semánticos...")
        
        if not all_docs['embeddings']:
            logger.warning("No hay embeddings disponibles para clustering")
            return
        
        try:
            from sklearn.cluster import KMeans
            
            # Convertir embeddings a array numpy
            embeddings = np.array(all_docs['embeddings'])
            
            # Determinar número óptimo de clusters (regla simple)
            n_clusters = min(max(5, len(embeddings) // 50), 20)
            
            # Aplicar KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Asignar documentos a clusters
            for i, doc_id in enumerate(all_docs['ids']):
                cluster_id = int(cluster_labels[i])
                self.indices['by_semantic_cluster'][cluster_id].append(doc_id)
                
                # Agregar cluster a metadatos
                if doc_id in self.metadata_cache:
                    self.metadata_cache[doc_id]['semantic_cluster'] = cluster_id
            
            logger.info(f"Creados {n_clusters} clusters semánticos")
            
        except ImportError:
            logger.warning("sklearn no disponible para clustering semántico")
        except Exception as e:
            logger.error(f"Error en clustering semántico: {e}")
    
    def _index_images(self):
        """Indexar referencias de imágenes"""
        try:
            image_docs = self.vector_manager.image_collection.get()
            
            for i, img_id in enumerate(image_docs['ids']):
                metadata = image_docs['metadatas'][i]
                
                # Indexar por manual
                manual = metadata.get('manual_name')
                if manual:
                    self.indices['by_manual'][manual].append(img_id)
                
                # Agregar a tipo de contenido
                self.indices['by_content_type']['image'].append(img_id)
                
        except Exception as e:
            logger.warning(f"Error indexando imágenes: {e}")
    
    def get_documents_by_criteria(self, 
                                  manual: Optional[str] = None,
                                  page: Optional[int] = None,
                                  section: Optional[str] = None,
                                  chapter: Optional[str] = None,
                                  content_type: Optional[str] = None,
                                  date_range: Optional[Tuple[str, str]] = None) -> List[str]:
        """Obtener documentos que cumplan criterios específicos"""
        
        result_sets = []
        
        if manual:
            result_sets.append(set(self.indices['by_manual'][manual]))
        
        if page and manual:
            result_sets.append(set(self.indices['by_page'][f"{manual}_{page}"]))
        
        if section and manual:
            result_sets.append(set(self.indices['by_section'][f"{manual}_{section}"]))
        
        if chapter and manual:
            result_sets.append(set(self.indices['by_chapter'][f"{manual}_{chapter}"]))
        
        if content_type:
            result_sets.append(set(self.indices['by_content_type'][content_type]))
        
        if date_range:
            # Filtrar por rango de fechas
            date_docs = set()
            start_date, end_date = date_range
            for date_key, docs in self.indices['by_date'].items():
                if start_date <= date_key <= end_date:
                    date_docs.update(docs)
            if date_docs:
                result_sets.append(date_docs)
        
        # Intersección de todos los conjuntos
        if result_sets:
            return list(result_sets[0].intersection(*result_sets[1:]))
        
        return []
    
    def get_context_documents(self, doc_id: str, window_size: int = 2) -> Dict[str, List[str]]:
        """Obtener documentos de contexto para expansión"""
        context = {
            'previous': [],
            'next': [],
            'same_page': [],
            'same_section': []
        }
        
        if doc_id not in self.adjacency_map:
            return context
        
        # Obtener chunks adyacentes
        adjacency = self.adjacency_map[doc_id]
        
        # Chunks anteriores
        current_id = doc_id
        for _ in range(window_size):
            if 'previous' in self.adjacency_map.get(current_id, {}):
                prev_id = self.adjacency_map[current_id]['previous']
                context['previous'].append(prev_id)
                current_id = prev_id
            else:
                break
        
        # Chunks siguientes
        current_id = doc_id
        for _ in range(window_size):
            if 'next' in self.adjacency_map.get(current_id, {}):
                next_id = self.adjacency_map[current_id]['next']
                context['next'].append(next_id)
                current_id = next_id
            else:
                break
        
        # Documentos de la misma página
        if doc_id in self.inverted_indices['doc_to_manual'] and doc_id in self.inverted_indices['doc_to_page']:
            manual = self.inverted_indices['doc_to_manual'][doc_id]
            page = self.inverted_indices['doc_to_page'][doc_id]
            page_key = f"{manual}_{page}"
            
            context['same_page'] = [
                d for d in self.indices['by_page'].get(page_key, [])
                if d != doc_id
            ]
        
        # Documentos de la misma sección
        if doc_id in self.inverted_indices['doc_to_section']:
            manual = self.inverted_indices['doc_to_manual'].get(doc_id)
            section = self.inverted_indices['doc_to_section'][doc_id]
            section_key = f"{manual}_{section}"
            
            context['same_section'] = [
                d for d in self.indices['by_section'].get(section_key, [])
                if d != doc_id
            ][:5]  # Limitar a 5 documentos
        
        return context
    
    def get_semantic_neighbors(self, doc_id: str, n_neighbors: int = 5) -> List[str]:
        """Obtener documentos del mismo cluster semántico"""
        if doc_id in self.metadata_cache:
            cluster_id = self.metadata_cache[doc_id].get('semantic_cluster')
            if cluster_id is not None:
                neighbors = self.indices['by_semantic_cluster'][cluster_id]
                return [n for n in neighbors if n != doc_id][:n_neighbors]
        
        return []
    
    def get_index_statistics(self) -> Dict[str, any]:
        """Obtener estadísticas de los índices"""
        stats = {
            'total_documents': len(self.metadata_cache),
            'indices': {},
            'coverage': {}
        }
        
        # Estadísticas por índice
        for index_name, index_data in self.indices.items():
            stats['indices'][index_name] = {
                'keys': len(index_data),
                'total_entries': sum(len(v) for v in index_data.values())
            }
        
        # Cobertura de metadatos
        stats['coverage'] = {
            'with_manual': len(self.inverted_indices['doc_to_manual']),
            'with_page': len(self.inverted_indices['doc_to_page']),
            'with_section': len(self.inverted_indices['doc_to_section']),
            'with_type': len(self.inverted_indices['doc_to_type'])
        }
        
        # Estadísticas de adyacencia
        stats['adjacency'] = {
            'documents_with_context': len(self.adjacency_map),
            'avg_nearby_docs': np.mean([
                len(adj.get('nearby', [])) 
                for adj in self.adjacency_map.values()
            ]) if self.adjacency_map else 0
        }
        
        return stats
    
    def optimize_indices(self):
        """Optimizar índices para mejor rendimiento"""
        logger.info("Optimizando índices...")
        
        # Convertir defaultdicts a dicts normales para mejor serialización
        for index_name in self.indices:
            self.indices[index_name] = dict(self.indices[index_name])
        
        # Ordenar listas de documentos para búsqueda binaria
        for index_data in self.indices.values():
            for key in index_data:
                if isinstance(index_data[key], list):
                    index_data[key].sort()
        
        logger.info("Índices optimizados")
    
    def save_indices(self, path: Path):
        """Guardar índices en disco"""
        self.optimize_indices()
        
        indices_data = {
            'indices': self.indices,
            'inverted_indices': self.inverted_indices,
            'adjacency_map': dict(self.adjacency_map),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0',
                'statistics': self.get_index_statistics()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(indices_data, f, indent=2)
        
        logger.info(f"Índices guardados en {path}")
    
    def load_indices(self, path: Path):
        """Cargar índices desde disco"""
        if path.exists():
            with open(path, 'r') as f:
                indices_data = json.load(f)
            
            # Restaurar índices
            self.indices = {
                k: defaultdict(list, v) 
                for k, v in indices_data['indices'].items()
            }
            
            self.inverted_indices = indices_data['inverted_indices']
            self.adjacency_map = defaultdict(dict, indices_data['adjacency_map'])
            
            # Reconstruir cache de metadatos si es necesario
            if not self.metadata_cache:
                self._rebuild_metadata_cache()
            
            logger.info("Índices cargados exitosamente")
    
    def _rebuild_metadata_cache(self):
        """Reconstruir cache de metadatos desde la base de datos"""
        all_docs = self.vector_manager.collection.get()
        
        for i, doc_id in enumerate(all_docs['ids']):
            self.metadata_cache[doc_id] = all_docs['metadatas'][i]