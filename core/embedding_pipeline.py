"""
Pipeline de embeddings optimizado con batching y cache
"""
import asyncio
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import hashlib
import pickle
from pathlib import Path
import lmdb
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Cache persistente de embeddings usando LMDB"""
    
    def __init__(self, cache_dir: str, map_size: int = 10 * 1024 * 1024 * 1024):  # 10GB
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = lmdb.open(
            str(self.cache_dir),
            map_size=map_size,
            max_dbs=2
        )
        
        # Bases de datos separadas para texto e imágenes
        self.text_db = self.env.open_db(b'text_embeddings')
        self.image_db = self.env.open_db(b'image_embeddings')
    
    def get_text_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Obtener embedding de texto del cache"""
        key = self._generate_key(text, model_name)
        
        with self.env.begin() as txn:
            data = txn.get(key.encode(), db=self.text_db)
            if data:
                return pickle.loads(data)
        return None
    
    def set_text_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Guardar embedding de texto en cache"""
        key = self._generate_key(text, model_name)
        
        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                pickle.dumps(embedding),
                db=self.text_db
            )
    
    def _generate_key(self, content: str, model_name: str) -> str:
        """Generar clave única para contenido"""
        combined = f"{model_name}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def close(self):
        """Cerrar base de datos"""
        self.env.close()

class OptimizedEmbeddingPipeline:
    """Pipeline optimizado para generación de embeddings"""
    
    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-large",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 max_length: int = 512,
                 cache_dir: Optional[str] = None):
        
        # Detectar dispositivo
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Cargar modelo y tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Cache
        if cache_dir:
            self.cache = EmbeddingCache(cache_dir)
        else:
            self.cache = None
        
        # Pool de threads para procesamiento paralelo
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def embed_texts_batch(self, 
                               texts: List[str],
                               show_progress: bool = True) -> np.ndarray:
        """Generar embeddings para un lote de textos"""
        
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        # Verificar cache primero
        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get_text_embedding(text, self.model.name_or_path)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue
            
            texts_to_process.append(text)
            indices_to_process.append(i)
        
        # Procesar textos no cacheados
        if texts_to_process:
            new_embeddings = await self._generate_embeddings_batch(
                texts_to_process,
                show_progress
            )
            
            # Guardar en cache y combinar resultados
            for idx, text, embedding in zip(indices_to_process, texts_to_process, new_embeddings):
                if self.cache:
                    self.cache.set_text_embedding(text, self.model.name_or_path, embedding)
                embeddings.append((idx, embedding))
        
        # Ordenar por índice original
        embeddings.sort(key=lambda x: x[0])
        
        return np.array([emb for _, emb in embeddings])
    
    async def _generate_embeddings_batch(self,
                                       texts: List[str],
                                       show_progress: bool) -> np.ndarray:
        """Generar embeddings en lotes optimizados"""
        
        all_embeddings = []
        
        # Procesar en lotes
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenización
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )
                
                # Normalizar
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
            
            if show_progress and i % (self.batch_size * 10) == 0:
                logger.info(f"Procesados {i}/{len(texts)} textos")
        
        return np.vstack(all_embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling considerando attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    async def embed_multimodal(self,
                              texts: List[str],
                              images: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generar embeddings multimodales"""
        
        # Procesar textos e imágenes en paralelo
        text_task = asyncio.create_task(self.embed_texts_batch(texts))
        image_task = asyncio.create_task(self._embed_images_batch(images))
        
        text_embeddings, image_embeddings = await asyncio.gather(text_task, image_task)
        
        return text_embeddings, image_embeddings
    
    async def _embed_images_batch(self, image_paths: List[str]) -> np.ndarray:
        """Generar embeddings de imágenes (placeholder para CLIP)"""
        # Aquí integrarías CLIP o similar
        # Por ahora retornamos embeddings dummy
        return np.random.randn(len(image_paths), 512)
    
    def create_hybrid_embedding(self,
                              text_embedding: np.ndarray,
                              metadata: Dict) -> np.ndarray:
        """Crear embedding híbrido combinando texto y metadatos"""
        
        # Características adicionales
        features = []
        
        # Importancia del chunk
        if 'importance_score' in metadata:
            features.append(metadata['importance_score'])
        
        # Densidad técnica
        if 'technical_density' in metadata:
            features.append(metadata['technical_density'])
        
        # Tipo de contenido (one-hot encoding simplificado)
        content_types = ['text', 'table_context', 'diagram_context', 'procedure']
        content_type = metadata.get('chunk_type', 'text')
        for ct in content_types:
            features.append(1.0 if ct == content_type else 0.0)
        
        # Combinar con embedding de texto
        if features:
            feature_vector = np.array(features)
            # Normalizar features
            feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
            
            # Concatenar con peso
            text_weight = 0.95
            feature_weight = 0.05
            
            hybrid = np.concatenate([
                text_embedding * text_weight,
                feature_vector * feature_weight
            ])
            
            # Normalizar resultado final
            hybrid = hybrid / (np.linalg.norm(hybrid) + 1e-8)
            
            return hybrid
        
        return text_embedding
    
    def close(self):
        """Limpiar recursos"""
        if self.cache:
            self.cache.close()
        self.thread_pool.shutdown()