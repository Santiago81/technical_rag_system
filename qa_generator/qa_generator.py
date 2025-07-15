"""
Main QA Generator using LangChain for fine-tuning dataset creation
"""
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import random
from concurrent.futures import ThreadPoolExecutor
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback

from chunk_manager import ChunkManager, Chunk
from prompt_templates import PromptTemplates, QuestionType, DifficultyLevel
from quality_evaluator import QualityEvaluator, QualityScore

logger = logging.getLogger(__name__)

@dataclass
class QAExample:
    """Structure for a Q&A training example"""
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format for fine-tuning"""
        return json.dumps({
            'messages': self.messages,
            'metadata': self.metadata
        }, ensure_ascii=False)

class QAGenerator:
    """Main class for generating Q&A datasets from chunks"""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
                 db_path: str = "docs.db",
                 output_dir: str = "qa_dataset",
                 domain: str = "technical documentation",
                 temperature: float = 0.7):
        
        # Initialize with optional API key for compatibility
        if api_key:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        else:
            # Assume API key is in environment
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature
            )
        
        self.chunk_manager = ChunkManager(db_path)
        self.prompt_templates = PromptTemplates()
        self.quality_evaluator = QualityEvaluator()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.domain = domain
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
        
        # Get context chunks
        context = self.chunk_manager.get_context_chunks(chunk.id, context_window)
        
        # Select appropriate template
        templates = self.prompt_templates.get_templates_by_type(question_type)
        templates = [t for t in templates if t.difficulty == difficulty]
        
        if not templates:
            logger.warning(f"No template found for {question_type} at {difficulty} level")
            return []
        
        template = random.choice(templates)
        
        # Prepare prompt
        if template.requires_multi_chunk:
            # For synthesis questions, we need multiple chunks
            if context['before']:
                chunk1 = context['before'][-1]
                chunk2 = chunk
            else:
                chunk1 = chunk
                chunk2 = context['after'][0] if context['after'] else chunk
            
            user_prompt = template.user_prompt.format(
                chunk1_id=chunk1.id,
                chunk1_content=chunk1.content,
                chunk2_id=chunk2.id,
                chunk2_content=chunk2.content,
                additional_chunks=""
            )
        else:
            # Single chunk questions
            context_text = ""
            if context['before']:
                context_text += "Contexto anterior: " + context['before'][-1].content[:200] + "...\n"
            if context['after']:
                context_text += "Contexto posterior: " + context['after'][0].content[:200] + "..."
            
            user_prompt = template.user_prompt.format(
                content=chunk.content,
                context=context_text
            )
        
        # Generate with LangChain
        try:
            with get_openai_callback() as cb:
                messages = [
                    SystemMessage(content=template.system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = await self.llm.ainvoke(messages)
                response = response.content
                
                # Update stats
                self.stats['total_tokens'] += cb.total_tokens
                self.stats['total_cost'] += cb.total_cost
            
            # Parse response
            qa_pairs = self._parse_qa_response(response)
            
            # Add metadata
            for qa in qa_pairs:
                qa['source_chunk_ids'] = [str(chunk.id)]
                qa['source_pdfs'] = [chunk.source_pdf]
                qa['page_numbers'] = [chunk.page_number]
                qa['difficulty'] = difficulty.value
                qa['reasoning_type'] = question_type.value
                qa['chunk_content'] = chunk.content
                
                if template.requires_multi_chunk:
                    qa['requires_chunks'] = [str(chunk1.id), str(chunk2.id)]
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating QA for chunk {chunk.id}: {e}")
            return []
    
    async def generate_question_variations(self, original_question: str) -> List[str]:
        """Generate variations of a question"""
        prompt = self.prompt_templates.get_variation_prompt()
        
        try:
            messages = [
                HumanMessage(content=prompt.format(original_question=original_question))
            ]
            response_msg = await self.llm.ainvoke(messages)
            response = response_msg.content
            
            # Parse variations
            try:
                data = json.loads(response)
                return data.get('variations', [])
            except:
                # Fallback to line parsing
                variations = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        variations.append(line[1:-1])
                return variations[:5]
                
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return []
    
    def _parse_qa_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract Q&A pairs"""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            return data.get('questions', [])
        except:
            # Fallback parsing
            qa_pairs = []
            lines = response.split('\n')
            
            current_qa = {}
            for line in lines:
                if line.strip().startswith('Pregunta:') or line.strip().startswith('Question:'):
                    if current_qa:
                        qa_pairs.append(current_qa)
                    current_qa = {'question': line.split(':', 1)[1].strip()}
                elif line.strip().startswith('Respuesta:') or line.strip().startswith('Answer:'):
                    current_qa['answer'] = line.split(':', 1)[1].strip()
            
            if current_qa and 'question' in current_qa and 'answer' in current_qa:
                qa_pairs.append(current_qa)
            
            return qa_pairs
    
    async def process_batch(self, 
                          batch_size: int = 10,
                          chunks_per_type: int = 20,
                          quality_threshold: float = 0.7) -> List[QAExample]:
        """Process a batch of chunks to generate Q&A examples"""
        
        # Get unprocessed chunks
        chunks = self.chunk_manager.get_chunks_batch(limit=batch_size, filter_processed=True)
        
        if not chunks:
            logger.info("No more unprocessed chunks")
            return []
        
        logger.info(f"Processing batch of {len(chunks)} chunks")
        
        all_examples = []
        processed_chunk_ids = []
        
        # Define generation tasks
        question_configs = [
            (QuestionType.FACTUAL, DifficultyLevel.BASIC),
            (QuestionType.FACTUAL, DifficultyLevel.INTERMEDIATE),
            (QuestionType.SYNTHESIS, DifficultyLevel.INTERMEDIATE),
            (QuestionType.CAUSAL, DifficultyLevel.ADVANCED),
            (QuestionType.APPLICATION, DifficultyLevel.INTERMEDIATE),
            (QuestionType.ANALYSIS, DifficultyLevel.ADVANCED),
        ]
        
        # Process each chunk
        for chunk in chunks:
            chunk_examples = []
            
            # Generate different types of questions
            for q_type, difficulty in question_configs:
                qa_pairs = await self.generate_qa_for_chunk(chunk, q_type, difficulty)
                
                for qa in qa_pairs:
                    # Evaluate quality
                    quality_score = self.quality_evaluator.evaluate_qa_pair(
                        qa['question'],
                        qa['answer'],
                        chunk.content,
                        difficulty.value,
                        q_type.value
                    )
                    
                    if quality_score.is_acceptable(quality_threshold):
                        # Generate variations - DISABLED FOR SPEED
                        # variations = await self.generate_question_variations(qa['question'])
                        qa['question_variations'] = []  # variations
                        qa['quality_score'] = quality_score.overall
                        
                        # Create training example
                        context = f"Chunk ID: {chunk.id}, PDF: {chunk.source_pdf}, Página: {chunk.page_number}\n\n{chunk.content}"
                        
                        example = QAExample(
                            messages=[
                                {
                                    "role": "system",
                                    "content": self.prompt_templates.format_system_message(
                                        self.domain, 
                                        context
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": qa['question']
                                },
                                {
                                    "role": "assistant",
                                    "content": qa['answer']
                                }
                            ],
                            metadata={
                                'source_chunk_ids': qa['source_chunk_ids'],
                                'source_pdfs': qa['source_pdfs'],
                                'page_numbers': qa['page_numbers'],
                                'difficulty': qa['difficulty'],
                                'reasoning_type': qa['reasoning_type'],
                                'requires_chunks': qa.get('requires_chunks', qa['source_chunk_ids']),
                                'quality_score': qa['quality_score'],
                                'question_variations': qa['question_variations']
                            }
                        )
                        
                        chunk_examples.append(example)
                        
                        # Update stats
                        self.stats['total_generated'] += 1
                        self.stats['total_accepted'] += 1
                        self.stats['by_type'][q_type.value] = self.stats['by_type'].get(q_type.value, 0) + 1
                        self.stats['by_difficulty'][difficulty.value] = self.stats['by_difficulty'].get(difficulty.value, 0) + 1
                    else:
                        self.stats['total_rejected'] += 1
                        logger.debug(f"Rejected Q&A with score {quality_score.overall}")
            
            all_examples.extend(chunk_examples)
            processed_chunk_ids.append(chunk.id)
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        # Mark chunks as processed
        self.chunk_manager.mark_as_processed(processed_chunk_ids)
        
        return all_examples
    
    async def generate_multi_chunk_synthesis(self, n_examples: int = 10) -> List[QAExample]:
        """Generate synthesis questions that require multiple chunks"""
        
        examples = []
        chunk_pairs = self.chunk_manager.get_random_chunk_pairs(n_examples)
        
        for chunk1, chunk2 in chunk_pairs:
            # Generate synthesis question
            template = self.prompt_templates.get_template('synthesis_intermediate')
            
            user_prompt = template.user_prompt.format(
                chunk1_id=chunk1.id,
                chunk1_content=chunk1.content,
                chunk2_id=chunk2.id,
                chunk2_content=chunk2.content,
                additional_chunks=""
            )
            
            try:
                messages = [
                    SystemMessage(content=template.system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response_msg = await self.llm.ainvoke(messages)
                response = response_msg.content
                
                qa_pairs = self._parse_qa_response(response)
                
                for qa in qa_pairs:
                    # Create multi-chunk context
                    context = f"""Chunk 1 (ID: {chunk1.id}, PDF: {chunk1.source_pdf}, Página: {chunk1.page_number}):
{chunk1.content}

Chunk 2 (ID: {chunk2.id}, PDF: {chunk2.source_pdf}, Página: {chunk2.page_number}):
{chunk2.content}"""
                    
                    example = QAExample(
                        messages=[
                            {
                                "role": "system",
                                "content": self.prompt_templates.format_system_message(
                                    self.domain,
                                    context
                                )
                            },
                            {
                                "role": "user",
                                "content": qa['question']
                            },
                            {
                                "role": "assistant",
                                "content": qa['answer']
                            }
                        ],
                        metadata={
                            'source_chunk_ids': [str(chunk1.id), str(chunk2.id)],
                            'source_pdfs': list(set([chunk1.source_pdf, chunk2.source_pdf])),
                            'page_numbers': [chunk1.page_number, chunk2.page_number],
                            'difficulty': 'intermediate',
                            'reasoning_type': 'synthesis',
                            'requires_chunks': [str(chunk1.id), str(chunk2.id)],
                            'quality_score': 0.0  # Will be evaluated
                        }
                    )
                    
                    examples.append(example)
                    
            except Exception as e:
                logger.error(f"Error generating multi-chunk synthesis: {e}")
        
        return examples
    
    def save_dataset(self, examples: List[QAExample], filename: str = None, append: bool = False):
        """Save dataset to JSONL file
        
        Args:
            examples: List of QA examples to save
            filename: Output filename (optional)
            append: If True, append to existing file instead of overwriting
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qa_dataset_{timestamp}.jsonl"
        
        filepath = self.output_dir / filename
        
        mode = 'a' if append and filepath.exists() else 'w'
        
        with open(filepath, mode, encoding='utf-8') as f:
            for example in examples:
                f.write(example.to_jsonl() + '\n')
        
        action = "Appended" if mode == 'a' else "Saved"
        logger.info(f"{action} {len(examples)} examples to {filepath}")
        
        # Save statistics
        stats_file = self.output_dir / f"stats_{filename.replace('.jsonl', '.json')}"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
    
    async def run_generation_pipeline(self,
                                    total_examples: int = 1000,
                                    batch_size: int = 10,
                                    save_interval: int = 100,
                                    filename: str = None,
                                    append: bool = False):
        """Run the complete generation pipeline"""
        
        logger.info(f"Starting generation pipeline for {total_examples} examples")
        
        all_examples = []
        start_time = time.time()
        
        while len(all_examples) < total_examples:
            # Generate batch
            batch_examples = await self.process_batch(batch_size=batch_size)
            
            if not batch_examples:
                logger.info("No more chunks to process")
                break
            
            all_examples.extend(batch_examples)
            
            # Save periodically
            if len(all_examples) >= save_interval:
                self.save_dataset(all_examples[:save_interval], filename=filename, append=append)
                all_examples = all_examples[save_interval:]
                # After first save, always append
                append = True
            
            # Log progress
            logger.info(f"Progress: {len(all_examples) + (save_interval * (len(all_examples) // save_interval))}/{total_examples}")
            logger.info(f"Stats: {self.stats}")
        
        # Save remaining examples
        if all_examples:
            self.save_dataset(all_examples, filename=filename, append=append)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Final statistics: {self.stats}")
        
        # Generate quality report
        self._generate_quality_report()
    
    def _parse_json_response(self, response_content: str, expected_fields: List[str]) -> Optional[Dict]:
        """Helper to parse JSON responses with fallback extraction"""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, attempting text extraction")
            
            # Try to extract fields from text
            result = {}
            for field in expected_fields:
                pattern = f'"{field}":'
                if pattern in response_content:
                    start = response_content.find(pattern) + len(pattern)
                    # Skip whitespace
                    while start < len(response_content) and response_content[start] in ' \t\n':
                        start += 1
                    
                    # Check if it's an array
                    if start < len(response_content) and response_content[start] == '[':
                        # Find matching closing bracket
                        bracket_count = 1
                        end = start + 1
                        while end < len(response_content) and bracket_count > 0:
                            if response_content[end] == '[':
                                bracket_count += 1
                            elif response_content[end] == ']':
                                bracket_count -= 1
                            end += 1
                        
                        if bracket_count == 0:
                            try:
                                result[field] = json.loads(response_content[start:end])
                            except:
                                result[field] = []
                    
                    # Check if it's a string
                    elif start < len(response_content) and response_content[start] == '"':
                        start += 1
                        end = start
                        while end < len(response_content) and response_content[end] != '"':
                            if response_content[end] == '\\' and end + 1 < len(response_content):
                                end += 2  # Skip escaped character
                            else:
                                end += 1
                        
                        if end < len(response_content):
                            result[field] = response_content[start:end]
                    
                    # Otherwise try to find the value up to comma or closing brace
                    else:
                        end = start
                        while end < len(response_content) and response_content[end] not in ',}':
                            end += 1
                        
                        value = response_content[start:end].strip()
                        if value:
                            result[field] = value
            
            return result if result else None
    
    def _generate_quality_report(self):
        """Generate a quality report of the dataset"""
        report = {
            'generation_stats': self.stats,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'model': self.llm.model_name,
                'temperature': self.llm.temperature,
                'domain': self.domain
            }
        }
        
        report_file = self.output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality report saved to {report_file}")
    
    async def generate_single_qa(self, chunk_text: str, reasoning_type: str, 
                                difficulty: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single QA pair for a chunk"""
        try:
            prompt = f"""Generate a {difficulty} {reasoning_type} question and answer based on this technical documentation chunk.

Chunk content:
{chunk_text}

Requirements:
- Question must be specific and technical
- Answer must be comprehensive and accurate
- Avoid generic questions
- Focus on technical details, specifications, or procedures

Return in JSON format:
{{
    "question": "specific technical question",
    "answer": "detailed technical answer",
    "reasoning_type": "{reasoning_type}",
    "difficulty": "{difficulty}"
}}"""
            
            messages = [
                SystemMessage(content="You are an expert in technical documentation analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse response with fallback
            qa_data = self._parse_json_response(
                response.content, 
                ['question', 'answer', 'reasoning_type', 'difficulty']
            )
            
            if not qa_data or 'question' not in qa_data or 'answer' not in qa_data:
                logger.error(f"Invalid response format: {response.content[:200]}...")
                return None
            
            # Ensure required fields have default values
            qa_data.setdefault('reasoning_type', reasoning_type)
            qa_data.setdefault('difficulty', difficulty)
            
            return {
                'question': qa_data['question'],
                'answer': qa_data['answer'],
                'reasoning_type': reasoning_type,
                'difficulty': difficulty,
                'quality_score': 0.0  # Will be evaluated separately
            }
            
        except Exception as e:
            logger.error(f"Error generating single QA: {e}")
            return None
    
    async def generate_multi_chunk_qa(self, chunks: List[Dict[str, Any]], reasoning_type: str,
                                     difficulty: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate QA that requires information from multiple chunks"""
        try:
            # Combine chunk texts
            combined_text = "\n\n---CHUNK SEPARATOR---\n\n".join([
                f"Chunk {i+1} (ID: {c['id']}, Page: {c.get('start_page', 'N/A')}):\n{c['chunk_text']}"
                for i, c in enumerate(chunks[:5])  # Limit to 5 chunks max
            ])
            
            prompt = f"""Generate an advanced synthesis question that requires combining information from multiple chunks.

Combined chunks:
{combined_text}

Context:
- Total chunks: {context['chunk_count']}
- Source PDFs: {', '.join(context['source_pdfs'])}
- Topics covered: {', '.join(context['topics'])}

Requirements:
- Question MUST require information from at least 2 different chunks
- Answer should synthesize information across chunks
- Focus on relationships, comparisons, or comprehensive understanding
- Be specific and technical

Return in JSON format:
{{
    "question": "synthesis question requiring multiple chunks",
    "answer": "comprehensive answer drawing from multiple sources",
    "source_chunk_ids": ["list of chunk IDs used"],
    "aggregation_type": "synthesis"
}}"""
            
            messages = [
                SystemMessage(content="You are an expert in technical documentation synthesis and analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            qa_data = self._parse_json_response(
                response.content,
                ['question', 'answer', 'source_chunk_ids', 'aggregation_type']
            )
            
            return {
                'question': qa_data['question'],
                'answer': qa_data['answer'],
                'reasoning_type': 'synthesis',
                'difficulty': 'advanced',
                'source_chunk_ids': [str(c['id']) for c in chunks],
                'source_pdfs': list(set(c.get('filename', 'unknown') for c in chunks)),
                'page_numbers': [c.get('start_page', -1) for c in chunks],
                'aggregation_type': 'synthesis',
                'quality_score': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error generating multi-chunk QA: {e}")
            return None
    
    async def generate_comparison_qa(self, chunks: List[Dict[str, Any]], 
                                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate comparison questions across chunks"""
        try:
            if len(chunks) < 2:
                return None
                
            chunk1, chunk2 = chunks[0], chunks[1]
            
            prompt = f"""Generate a comparison question that contrasts information from two chunks.

Chunk 1 (ID: {chunk1['id']}, Page: {chunk1.get('start_page', 'N/A')}):
{chunk1['chunk_text'][:500]}...

Chunk 2 (ID: {chunk2['id']}, Page: {chunk2.get('start_page', 'N/A')}):
{chunk2['chunk_text'][:500]}...

Requirements:
- Question must compare/contrast specific aspects from both chunks
- Answer should highlight differences and similarities
- Focus on technical specifications, procedures, or features

Return in JSON format:
{{
    "question": "comparison question",
    "answer": "detailed comparison answer",
    "source_chunk_ids": ["{chunk1['id']}", "{chunk2['id']}"],
    "aggregation_type": "comparison"
}}"""
            
            messages = [
                SystemMessage(content="You are an expert in technical documentation comparison and analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            qa_data = self._parse_json_response(
                response.content,
                ['question', 'answer', 'source_chunk_ids', 'aggregation_type']
            )
            
            return {
                'question': qa_data['question'],
                'answer': qa_data['answer'],
                'reasoning_type': 'comparison',
                'difficulty': 'advanced',
                'source_chunk_ids': [str(chunk1['id']), str(chunk2['id'])],
                'source_pdfs': list(set([chunk1.get('filename', 'unknown'), chunk2.get('filename', 'unknown')])),
                'page_numbers': [chunk1.get('start_page', -1), chunk2.get('start_page', -1)],
                'aggregation_type': 'comparison',
                'quality_score': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison QA: {e}")
            return None
    
    async def generate_comprehensive_analysis(self, chunks: List[Dict[str, Any]], 
                                            context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate comprehensive analysis questions from multiple chunks"""
        try:
            # Create a summary of all chunks
            chunk_summaries = []
            for i, chunk in enumerate(chunks[:5]):
                summary = f"Chunk {i+1}: {chunk['chunk_text'][:200]}..."
                chunk_summaries.append(summary)
            
            prompt = f"""Generate a comprehensive analysis question that requires deep understanding of multiple chunks.

Chunk summaries:
{chr(10).join(chunk_summaries)}

Context:
- Topics: {', '.join(context['topics'])}
- Page range: {context['page_range']['start']} to {context['page_range']['end']}

Requirements:
- Question should require analyzing patterns, trends, or system-wide understanding
- Answer must integrate information from multiple sources
- Focus on technical insights, design decisions, or operational implications
- Be highly specific to the technical domain

Return in JSON format:
{{
    "question": "comprehensive analysis question",
    "answer": "detailed analytical answer with insights",
    "source_chunk_ids": ["list of relevant chunk IDs"],
    "aggregation_type": "comprehensive_analysis"
}}"""
            
            messages = [
                SystemMessage(content="You are an expert in technical systems analysis and documentation."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            qa_data = self._parse_json_response(
                response.content,
                ['question', 'answer', 'source_chunk_ids', 'aggregation_type']
            )
            
            return {
                'question': qa_data['question'],
                'answer': qa_data['answer'],
                'reasoning_type': 'comprehensive_analysis',
                'difficulty': 'advanced',
                'source_chunk_ids': [str(c['id']) for c in chunks],
                'source_pdfs': list(set(c.get('filename', 'unknown') for c in chunks)),
                'page_numbers': [c.get('start_page', -1) for c in chunks],
                'aggregation_type': 'comprehensive_analysis',
                'quality_score': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis QA: {e}")
            return None