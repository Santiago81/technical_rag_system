"""
Configuration for QA Generator
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class QAGeneratorConfig:
    """Configuration for QA generation"""
    
    # Model Configuration
    use_local_model: bool = False  # Changed to use OpenAI API by default
    local_model_url: str = 'http://localhost:1234'
    local_model_endpoint: str = '/v1/chat/completions'
    
    # OpenAI Configuration (for backwards compatibility)
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    model: str = 'gpt-4o-mini'
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Database Configuration
    db_path: str = 'docs.db'
    
    # Output Configuration
    output_dir: str = 'qa_dataset'
    
    # Generation Parameters
    batch_size: int = 10
    context_window: int = 2  # chunks before/after
    quality_threshold: float = 0.7
    
    # Domain Configuration
    domain: str = 'technical documentation'
    
    # Cache Configuration
    cache_dir: str = 'qa_cache'
    
    # Rate Limiting
    requests_per_minute: int = 50
    delay_between_requests: float = 0.5
    
    # Dataset Parameters
    questions_per_chunk: int = 3
    variation_count: int = 3
    
    # Question Type Distribution
    question_type_weights: dict = None
    
    def __post_init__(self):
        """Initialize default values"""
        if not self.question_type_weights:
            self.question_type_weights = {
                'factual': 0.3,
                'synthesis': 0.2,
                'causal': 0.15,
                'application': 0.15,
                'comparison': 0.1,
                'analysis': 0.1
            }
        
        # Create directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.cache_dir).mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.use_local_model and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when not using local model. Set OPENAI_API_KEY environment variable.")
        
        if not Path(self.db_path).exists():
            raise ValueError(f"Database file not found: {self.db_path}")
        
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        
        total_weight = sum(self.question_type_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Question type weights must sum to 1.0, got {total_weight}")
        
        return True
    
    @classmethod
    def from_file(cls, config_path: str) -> 'QAGeneratorConfig':
        """Load configuration from JSON file"""
        import json
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'use_local_model': self.use_local_model,
            'local_model_url': self.local_model_url,
            'local_model_endpoint': self.local_model_endpoint,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'db_path': self.db_path,
            'output_dir': self.output_dir,
            'batch_size': self.batch_size,
            'context_window': self.context_window,
            'quality_threshold': self.quality_threshold,
            'domain': self.domain,
            'cache_dir': self.cache_dir,
            'requests_per_minute': self.requests_per_minute,
            'delay_between_requests': self.delay_between_requests,
            'questions_per_chunk': self.questions_per_chunk,
            'variation_count': self.variation_count,
            'question_type_weights': self.question_type_weights
        }