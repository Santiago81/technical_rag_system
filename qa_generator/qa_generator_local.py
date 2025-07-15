"""
Question-Answer Generator for Technical Documentation with Local Model Support
"""
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import pickle
import requests

from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from chunk_manager import ChunkManager, Chunk
from prompt_templates import PromptTemplates, QuestionType, DifficultyLevel
from quality_evaluator import QualityEvaluator, QualityScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalChatModel:
    """Custom LangChain-compatible wrapper for local models"""
    
    def __init__(self, base_url: str, endpoint: str, model: str, temperature: float = 0.7):
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.url = f"{self.base_url}{self.endpoint}"
    
    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Async invoke method compatible with LangChain"""
        # Convert messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            else:
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the assistant's response
            content = result["choices"][0]["message"]["content"]
            return HumanMessage(content=content)  # Return as HumanMessage for compatibility
            
        except Exception as e:
            logger.error(f"Error calling local model: {e}")
            raise


@dataclass
class QAPair:
    question: str
    answer: str
    source_chunk_id: str
    difficulty: str
    type: str
    context: str
    metadata: Dict = None
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class QAGenerator:
    """Generates high-quality Q&A pairs from technical documentation chunks"""
    
    def __init__(self, 
                 config: 'QAGeneratorConfig'):
        
        # Initialize the appropriate model
        if config.use_local_model:
            self.llm = LocalChatModel(
                base_url=config.local_model_url,
                endpoint=config.local_model_endpoint,
                model=config.model,
                temperature=config.temperature
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                api_key=config.openai_api_key,
                model=config.model,
                temperature=config.temperature
            )
        
        self.config = config
        self.chunk_manager = ChunkManager(config.db_path)
        self.prompt_templates = PromptTemplates()
        self.quality_evaluator = QualityEvaluator()
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.domain = config.domain
        self.stats = {
            'total_generated': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'by_type': {},
            'by_difficulty': {}
        }
    
    async def generate_qa_for_chunk(self, 
                                  chunk: Chunk, 
                                  question_type: QuestionType,
                                  difficulty: DifficultyLevel,
                                  context_window: int = 2) -> List[Dict]:
        """Generate Q&A pairs for a single chunk"""
        
        # Get surrounding context
        context = await self._get_context(chunk, context_window)
        
        # Get the appropriate prompt
        template_name = f"{question_type.value}_{difficulty.value}"
        prompt_template_obj = self.prompt_templates.get_template(template_name)
        if not prompt_template_obj:
            # Fallback to a basic template
            prompt_template_obj = self.prompt_templates.get_template("factual_basic")
        
        # Extract the human prompt from the template object
        if hasattr(prompt_template_obj, 'human_prompt'):
            prompt_template = prompt_template_obj.human_prompt
        else:
            # Default prompt if template not found
            prompt_template = """Based on the following content, generate 3 questions and their answers.
            
Content: {content}
Context: {context}

Format your response as JSON:
[
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
]"""
        
        # Create the prompt
        system_prompt = prompt_template_obj.system_prompt if prompt_template_obj and hasattr(prompt_template_obj, 'system_prompt') else "You are a helpful assistant."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", prompt_template)
        ])
        
        # Format the prompt with chunk content
        messages = prompt.format_messages(
            content=chunk.content,
            context=context,
            domain=self.domain,
            manual_name=chunk.source_pdf or "Technical Manual"
        )
        
        try:
            # Generate Q&A pairs
            if self.config.use_local_model:
                response = await self.llm.ainvoke(messages)
                # No callback tracking for local models
                tokens_used = 0
                cost = 0
            else:
                # Use OpenAI callback for token tracking
                from langchain_community.callbacks.manager import get_openai_callback
                with get_openai_callback() as cb:
                    response = await self.llm.ainvoke(messages)
                    tokens_used = cb.total_tokens
                    cost = cb.total_cost
                    
                    self.stats['total_tokens'] += tokens_used
                    self.stats['total_cost'] += cost
            
            # Parse the response
            qa_pairs = self._parse_response(response.content, chunk, question_type, difficulty)
            
            # Evaluate quality
            evaluated_pairs = []
            for qa in qa_pairs:
                quality = await self.quality_evaluator.evaluate_qa_pair(qa)
                if quality.overall_score >= self.config.quality_threshold:
                    qa.quality_score = quality.overall_score
                    evaluated_pairs.append(qa)
                    self.stats['total_accepted'] += 1
                else:
                    self.stats['total_rejected'] += 1
            
            self.stats['total_generated'] += len(qa_pairs)
            
            return [qa.to_dict() for qa in evaluated_pairs]
            
        except Exception as e:
            logger.error(f"Error generating Q&A for chunk {chunk.id}: {e}")
            return []
    
    async def _get_context(self, chunk: Chunk, window: int) -> str:
        """Get surrounding chunks for context"""
        try:
            # Get chunks from the same manual
            all_chunks = self.chunk_manager.get_chunks_by_pdf(chunk.source_pdf)
            
            # Sort by page and position
            sorted_chunks = sorted(all_chunks, key=lambda c: (c.page_number, getattr(c, 'position', 0)))
            
            # Find current chunk index
            current_idx = next((i for i, c in enumerate(sorted_chunks) if c.id == chunk.id), -1)
            
            if current_idx == -1:
                return ""
            
            # Get surrounding chunks
            start_idx = max(0, current_idx - window)
            end_idx = min(len(sorted_chunks), current_idx + window + 1)
            
            context_chunks = sorted_chunks[start_idx:end_idx]
            
            # Build context string
            context_parts = []
            for c in context_chunks:
                if c.id != chunk.id:
                    context_parts.append(f"[Page {c.page_number}] {c.content[:200]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    def _parse_response(self, response: str, chunk: Chunk, 
                       question_type: QuestionType, 
                       difficulty: DifficultyLevel) -> List[QAPair]:
        """Parse LLM response into QAPair objects"""
        qa_pairs = []
        
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict) and 'questions' in parsed:
                items = parsed['questions']
            else:
                items = [parsed]
            
            for item in items:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    qa = QAPair(
                        question=item['question'],
                        answer=item['answer'],
                        source_chunk_id=chunk.id,
                        difficulty=difficulty.value,
                        type=question_type.value,
                        context=chunk.content,
                        metadata={
                            'manual_name': chunk.source_pdf,
                            'page_num': chunk.page_number,
                            'section': getattr(chunk, 'section', '')
                        }
                    )
                    qa_pairs.append(qa)
                    
        except json.JSONDecodeError:
            # Try to parse as text format
            lines = response.strip().split('\n')
            current_q = None
            current_a = None
            
            for line in lines:
                if line.strip().startswith('Q:') or line.strip().startswith('Question:'):
                    if current_q and current_a:
                        qa = QAPair(
                            question=current_q,
                            answer=current_a,
                            source_chunk_id=chunk.id,
                            difficulty=difficulty.value,
                            type=question_type.value,
                            context=chunk.content,
                            metadata={
                                'manual_name': chunk.source_pdf,
                                'page_num': chunk.page_number,
                                'section': getattr(chunk, 'section', '')
                            }
                        )
                        qa_pairs.append(qa)
                    current_q = line.strip()[2:].strip() if line.strip().startswith('Q:') else line.strip()[9:].strip()
                    current_a = None
                elif line.strip().startswith('A:') or line.strip().startswith('Answer:'):
                    current_a = line.strip()[2:].strip() if line.strip().startswith('A:') else line.strip()[7:].strip()
            
            # Add the last pair
            if current_q and current_a:
                qa = QAPair(
                    question=current_q,
                    answer=current_a,
                    source_chunk_id=chunk.id,
                    difficulty=difficulty.value,
                    type=question_type.value,
                    context=chunk.content,
                    metadata={
                        'manual_name': chunk.manual_name,
                        'page_num': chunk.page_num,
                        'section': chunk.section
                    }
                )
                qa_pairs.append(qa)
        
        return qa_pairs
    
    async def generate_dataset(self, 
                             num_chunks: Optional[int] = None,
                             questions_per_chunk: int = 3) -> Dict:
        """Generate a complete Q&A dataset"""
        
        # Get all chunks
        chunks = self.chunk_manager.get_chunks_batch(limit=1000)
        
        if num_chunks:
            chunks = random.sample(chunks, min(num_chunks, len(chunks)))
        
        logger.info(f"Generating Q&A pairs for {len(chunks)} chunks")
        
        all_qa_pairs = []
        
        # Process chunks in batches
        batch_size = self.config.batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            tasks = []
            for chunk in batch:
                # Generate questions of different types
                for _ in range(questions_per_chunk):
                    question_type = self._select_question_type()
                    difficulty = self._select_difficulty()
                    
                    task = self.generate_qa_for_chunk(
                        chunk, question_type, difficulty, 
                        self.config.context_window
                    )
                    tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for result in batch_results:
                all_qa_pairs.extend(result)
            
            # Save intermediate results
            if (i + batch_size) % (batch_size * 5) == 0:
                self._save_intermediate_results(all_qa_pairs)
            
            # Rate limiting
            await asyncio.sleep(self.config.delay_between_requests)
        
        # Save final dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_file = self.output_dir / f"qa_dataset_{timestamp}.json"
        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'total_qa_pairs': len(all_qa_pairs),
                'config': self.config.to_dict(),
                'stats': self.stats
            },
            'qa_pairs': all_qa_pairs
        }
        
        with open(json_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Also save as JSONL for fine-tuning
        jsonl_file = self.output_dir / f"qa_dataset_{timestamp}.jsonl"
        with open(jsonl_file, 'w') as f:
            for qa_pair in all_qa_pairs:
                # Format for fine-tuning
                formatted = {
                    "messages": [
                        {"role": "system", "content": f"You are a helpful assistant specializing in {self.domain}."},
                        {"role": "user", "content": qa_pair['question']},
                        {"role": "assistant", "content": qa_pair['answer']}
                    ]
                }
                f.write(json.dumps(formatted) + '\n')
        
        logger.info(f"Dataset saved to:")
        logger.info(f"  - JSON: {json_file}")
        logger.info(f"  - JSONL: {jsonl_file}")
        logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs")
        logger.info(f"Statistics: {self.stats}")
        
        return dataset
    
    def _select_question_type(self) -> QuestionType:
        """Select question type based on configured weights"""
        weights = self.config.question_type_weights
        types = list(QuestionType)
        type_weights = [weights.get(t.value, 0.1) for t in types]
        
        return random.choices(types, weights=type_weights)[0]
    
    def _select_difficulty(self) -> DifficultyLevel:
        """Select difficulty level with balanced distribution"""
        return random.choice(list(DifficultyLevel))
    
    def _save_intermediate_results(self, qa_pairs: List[Dict]):
        """Save intermediate results to prevent data loss"""
        intermediate_file = self.output_dir / "intermediate_results.json"
        
        with open(intermediate_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'count': len(qa_pairs),
                'qa_pairs': qa_pairs
            }, f, indent=2)
        
        logger.info(f"Saved {len(qa_pairs)} intermediate results")
    
    async def create_variations(self, qa_pair: Dict, num_variations: int = 3) -> List[Dict]:
        """Create variations of a Q&A pair"""
        variations = []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating variations of questions while maintaining the same meaning."),
            ("human", """Create {num_variations} variations of this question-answer pair.
            Each variation should:
            - Ask for the same information in a different way
            - Maintain the same difficulty level
            - Be natural and grammatically correct
            
            Original Question: {question}
            Original Answer: {answer}
            
            Return as JSON array with format:
            [
                {{
                    "question": "variation 1",
                    "answer": "adapted answer if needed"
                }},
                ...
            ]
            """)
        ])
        
        messages = prompt.format_messages(
            num_variations=num_variations,
            question=qa_pair['question'],
            answer=qa_pair['answer']
        )
        
        try:
            response = await self.llm.ainvoke(messages)
            parsed = json.loads(response.content)
            
            for var in parsed:
                variation = qa_pair.copy()
                variation['question'] = var['question']
                variation['answer'] = var.get('answer', qa_pair['answer'])
                variation['is_variation'] = True
                variation['original_id'] = qa_pair.get('id', '')
                variations.append(variation)
                
        except Exception as e:
            logger.error(f"Error creating variations: {e}")
        
        return variations