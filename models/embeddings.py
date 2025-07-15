from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class EmbeddingManager:
    """Gestor de embeddings para texto e imágenes"""
    
    def __init__(self, text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_clip: bool = False):
        self.text_model = SentenceTransformer(text_model_name)
        self.use_clip = use_clip
        
        if use_clip:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Verificar si hay GPU disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model.to(self.device)
        
        if use_clip:
            self.clip_model.to(self.device)
    
    def encode_text(self, texts: Union[str, List[str]], 
                   batch_size: int = 32) -> np.ndarray:
        """Codificar texto(s) a embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Codificar imagen a embedding usando CLIP"""
        if not self.use_clip:
            raise ValueError("CLIP no está habilitado. Inicializar con use_clip=True")
        
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()[0]
    
    def encode_text_for_image_search(self, text: str) -> np.ndarray:
        """Codificar texto para búsqueda de imágenes usando CLIP"""
        if not self.use_clip:
            raise ValueError("CLIP no está habilitado. Inicializar con use_clip=True")
        
        inputs = self.clip_processor(text=[text], return_tensors="pt", 
                                   padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        
        return text_features.cpu().numpy()[0]
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Calcular similitud coseno entre dos embeddings"""
        # Normalizar vectores
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Producto punto = similitud coseno para vectores normalizados
        return np.dot(norm1, norm2)
