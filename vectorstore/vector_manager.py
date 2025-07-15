
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import json
from pathlib import Path
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorManager:
    """Gestor principal de la base de datos vectorial"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Inicializar ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Crear o obtener colección principal
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Colección para embeddings de imágenes
        self.image_collection = self.client.get_or_create_collection(
            name=f"{config.CHROMA_COLLECTION_NAME}_images",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Inicializar sistema de indexación (lazy loading)
        self._indexing_system = None
        
        # Cache de embeddings para eficiencia
        self._embedding_cache = {}
    
    @property
    def indexing_system(self):
        """Lazy loading del sistema de indexación"""
        if self._indexing_system is None:
            from .indexing import IndexingSystem
            self._indexing_system = IndexingSystem(self)
        return self._indexing_system
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Añadir documentos a la base vectorial"""
        logger.info(f"Añadiendo {len(documents)} documentos a la base vectorial")
        
        # Procesar en lotes
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Preparar datos para ChromaDB
            texts = [doc['content'] for doc in batch]
            metadatas = [self._enrich_metadata(doc['metadata']) for doc in batch]
            ids = [self.generate_doc_id(doc) for doc in batch]
            
            # Generar embeddings
            embeddings = self._get_embeddings(texts)
            
            # Añadir a ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        logger.info("Documentos añadidos exitosamente")
        
        # Invalidar índices para forzar reconstrucción
        if self._indexing_system:
            self._indexing_system = None
    
    def add_image_references(self, image_metadata: List[Dict]):
        """Añadir referencias de imágenes con sus metadatos"""
        for img_meta in image_metadata:
            # Generar embedding basado en metadatos textuales
            text_description = self.generate_image_description(img_meta)
            embedding = self._get_embeddings([text_description])[0]
            
            # ID único para la imagen
            img_id = f"img_{img_meta['manual_name']}_{img_meta['page_number']}_{img_meta['image_index']}"
            
            # Enriquecer metadatos
            enriched_meta = self._enrich_metadata(img_meta)
            
            self.image_collection.add(
                embeddings=[embedding],
                documents=[text_description],
                metadatas=[enriched_meta],
                ids=[img_id]
            )
    
    def search(self, query: str, n_results: int = 5, 
               filter_manual: Optional[str] = None,
               filter_page: Optional[int] = None,
               content_type: Optional[str] = None,
               include_distances: bool = True) -> Dict:
        """Búsqueda básica con filtros opcionales"""
        
        # Construir filtros
        where = self._build_where_clause(filter_manual, filter_page, content_type)
        
        # Generar embedding de la consulta
        query_embedding = self._get_embeddings([query])[0]
        
        # Buscar en colección de texto
        text_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where if where else None,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Buscar también en imágenes si no hay filtro de tipo de contenido
        image_results = None
        if not content_type or content_type == "image":
            image_where = {"manual_name": filter_manual} if filter_manual else None
            image_results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(3, n_results),
                where=image_where,
                include=['metadatas', 'documents', 'distances']
            )
        
        return {
            'text_results': self.format_results(text_results, include_distances),
            'image_results': self.format_results(image_results, include_distances) if image_results else []
        }
    
    def search_by_manual(self, query: str, manual_name: str, n_results: int = 5) -> Dict:
        """Búsqueda específica dentro de un manual"""
        return self.search(query, n_results, filter_manual=manual_name)
    
    def get_documents_by_ids(self, doc_ids: List[str]) -> Dict:
        """Obtener documentos específicos por sus IDs"""
        if not doc_ids:
            return {'ids': [], 'documents': [], 'metadatas': []}
        
        try:
            results = self.collection.get(
                ids=doc_ids,
                include=['metadatas', 'documents', 'embeddings']
            )
            return results
        except Exception as e:
            logger.error(f"Error obteniendo documentos por IDs: {e}")
            return {'ids': [], 'documents': [], 'metadatas': []}
    
    def get_similar_documents(self, doc_id: str, n_results: int = 5) -> List[Dict]:
        """Encontrar documentos similares a uno dado"""
        # Obtener el documento original
        doc_data = self.collection.get(ids=[doc_id], include=['embeddings'])
        
        if not doc_data['ids']:
            return []
        
        # Usar el embedding del documento como query
        doc_embedding = doc_data['embeddings'][0]
        
        # Buscar similares (excluyendo el mismo documento)
        results = self.collection.query(
            query_embeddings=[doc_embedding],
            n_results=n_results + 1,  # +1 porque incluirá el mismo documento
            include=['metadatas', 'documents', 'distances']
        )
        
        # Formatear y excluir el documento original
        formatted = self.format_results(results, include_distances=True)
        return [r for r in formatted if r['id'] != doc_id][:n_results]
    
    def update_document_metadata(self, doc_id: str, metadata_updates: Dict):
        """Actualizar metadatos de un documento"""
        try:
            # Obtener documento actual
            current = self.collection.get(ids=[doc_id], include=['metadatas'])
            
            if current['ids']:
                # Combinar metadatos
                new_metadata = {**current['metadatas'][0], **metadata_updates}
                new_metadata['last_updated'] = datetime.now().isoformat()
                
                # Actualizar en ChromaDB
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[new_metadata]
                )
                
                logger.info(f"Metadatos actualizados para documento {doc_id}")
        except Exception as e:
            logger.error(f"Error actualizando metadatos: {e}")
    
    def get_manual_list(self) -> List[str]:
        """Obtener lista de manuales en la base de datos"""
        # Usar el sistema de indexación si está disponible
        if self._indexing_system and hasattr(self._indexing_system, 'indices'):
            return sorted(list(self._indexing_system.indices['by_manual'].keys()))
        
        # Fallback: obtener de metadatos
        all_metadata = self.collection.get(limit=10000)['metadatas']
        manuals = set(meta.get('manual_name', '') for meta in all_metadata if meta)
        return sorted(list(manuals))
    
    def create_manual_index(self, manual_name: str) -> Dict:
        """Crear índice específico para un manual"""
        # Delegar al sistema de indexación
        if not self._indexing_system:
            self.indexing_system.build_indices()
        
        # Obtener documentos del manual
        manual_docs = self.indexing_system.get_documents_by_criteria(manual=manual_name)
        
        # Obtener metadatos
        results = self.get_documents_by_ids(manual_docs)
        
        # Organizar información
        index = {
            'manual_name': manual_name,
            'total_documents': len(manual_docs),
            'pages': {},
            'sections': {},
            'chapters': {},
            'statistics': {}
        }
        
        for meta in results['metadatas']:
            page = meta.get('page_number')
            section = meta.get('section')
            chapter = meta.get('chapter')
            
            # Indexar por página
            if page:
                if page not in index['pages']:
                    index['pages'][page] = 0
                index['pages'][page] += 1
            
            # Indexar por sección
            if section:
                if section not in index['sections']:
                    index['sections'][section] = []
                if page and page not in index['sections'][section]:
                    index['sections'][section].append(page)
            
            # Indexar por capítulo
            if chapter:
                if chapter not in index['chapters']:
                    index['chapters'][chapter] = []
                if page and page not in index['chapters'][chapter]:
                    index['chapters'][chapter].append(page)
        
        # Calcular estadísticas
        index['statistics'] = {
            'total_pages': len(index['pages']),
            'total_sections': len(index['sections']),
            'total_chapters': len(index['chapters']),
            'avg_chunks_per_page': np.mean(list(index['pages'].values())) if index['pages'] else 0
        }
        
        return index
    
    def get_collection_stats(self) -> Dict:
        """Obtener estadísticas de las colecciones"""
        stats = {
            'text_collection': {
                'count': self.collection.count(),
                'name': self.collection.name
            },
            'image_collection': {
                'count': self.image_collection.count(),
                'name': self.image_collection.name
            },
            'total_documents': self.collection.count() + self.image_collection.count()
        }
        
        # Agregar estadísticas de indexación si están disponibles
        if self._indexing_system:
            stats['indexing'] = self._indexing_system.get_index_statistics()
        
        return stats
    
    def _enrich_metadata(self, metadata: Dict) -> Dict:
        """Enriquecer metadatos con información adicional"""
        enriched = metadata.copy()
        
        # Agregar timestamp si no existe
        if 'timestamp' not in enriched:
            enriched['timestamp'] = datetime.now().isoformat()
        
        # Agregar versión del modelo de embeddings
        enriched['embedding_model'] = self.config.EMBEDDING_MODEL
        
        return enriched
    
    def _build_where_clause(self, manual: Optional[str], 
                           page: Optional[int], 
                           content_type: Optional[str]) -> Optional[Dict]:
        """Construir cláusula where para ChromaDB"""
        where = {}
        
        if manual:
            where["manual_name"] = manual
        if page is not None:
            where["page_number"] = page
        if content_type:
            where["content_type"] = content_type
        
        return where if where else None
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtener embeddings con cache"""
        embeddings = []
        
        for text in texts:
            # Verificar cache
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
            else:
                # Generar embedding
                embedding = self.embedding_model.encode([text])[0].tolist()
                self._embedding_cache[text_hash] = embedding
                embeddings.append(embedding)
                
                # Limitar tamaño del cache
                if len(self._embedding_cache) > 10000:
                    # Eliminar entradas más antiguas (simple FIFO)
                    first_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[first_key]
        
        return embeddings
    
    def generate_doc_id(self, doc: Dict) -> str:
        """Generar ID único para documento"""
        content = doc['content']
        metadata = doc['metadata']
        
        id_string = f"{metadata.get('manual_name', '')}_{metadata.get('page_number', '')}_{metadata.get('chunk_index', '')}_{content[:50]}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def generate_image_description(self, img_meta: Dict) -> str:
        """Generar descripción textual de imagen para embedding"""
        parts = [
            f"Imagen del manual {img_meta['manual_name']}",
            f"página {img_meta['page_number']}"
        ]
        
        # Agregar información OCR si existe
        if 'ocr_text' in img_meta and img_meta['ocr_text']:
            parts.append(f"texto detectado: {img_meta['ocr_text'][:100]}")
        
        # Agregar tipo si es diagrama
        if img_meta.get('content_type') == 'diagram':
            parts.append("diagrama técnico")
        
        return ", ".join(parts)
    
    def format_results(self, results: Dict, include_distances: bool = True) -> List[Dict]:
        """Formatear resultados de búsqueda"""
        formatted = []
        
        if not results or 'ids' not in results or not results['ids']:
            return formatted
        
        # ChromaDB devuelve resultados anidados
        ids = results['ids'][0] if results['ids'] else []
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if include_distances and 'distances' in results else []
        
        for i in range(len(ids)):
            result_dict = {
                'id': ids[i],
                'content': documents[i] if i < len(documents) else '',
                'metadata': metadatas[i] if i < len(metadatas) else {}
            }
            
            if include_distances and i < len(distances):
                result_dict['distance'] = distances[i]
                result_dict['score'] = 1.0 - distances[i]  # Convertir distancia a score
            
            formatted.append(result_dict)
        
        return formatted