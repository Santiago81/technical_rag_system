"""
Arquitectura unificada optimizada para el sistema RAG
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

@dataclass
class ProcessingConfig:
    """Configuración unificada del sistema"""
    max_workers: int = 4
    batch_size: int = 32
    chunk_size: int = 512
    chunk_overlap: int = 128  # Aumentado para mejor contexto
    enable_cache: bool = True
    cache_ttl: int = 3600
    streaming_threshold_mb: int = 50
    
class DocumentProcessor(ABC):
    """Interfaz base para procesadores de documentos"""
    
    @abstractmethod
    async def process(self, document_path: str) -> Iterator[Dict[str, Any]]:
        """Procesar documento de forma asíncrona con streaming"""
        pass

class UnifiedRAGPipeline:
    """Pipeline unificado con procesamiento paralelo y streaming"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers // 2)
        
    async def process_documents_parallel(self, document_paths: List[str]) -> None:
        """Procesar múltiples documentos en paralelo"""
        tasks = []
        
        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_with_limit(path: str):
            async with semaphore:
                return await self.process_single_document(path)
        
        for path in document_paths:
            task = asyncio.create_task(process_with_limit(path))
            tasks.append(task)
        
        # Procesar con progress tracking
        for completed in asyncio.as_completed(tasks):
            result = await completed
            yield result