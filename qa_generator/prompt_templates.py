"""
Prompt templates for different types of question generation
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    """Types of questions that can be generated"""
    FACTUAL = "factual"
    SYNTHESIS = "synthesis"
    CAUSAL = "causal"
    APPLICATION = "application"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"

class DifficultyLevel(Enum):
    """Difficulty levels for questions"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class PromptTemplate:
    """Structure for a prompt template"""
    name: str
    system_prompt: str
    user_prompt: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    requires_multi_chunk: bool = False

class PromptTemplates:
    """Collection of prompt templates for QA generation"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates"""
        templates = {}
        
        # Factual Questions Template
        templates['factual_basic'] = PromptTemplate(
            name="Factual Basic",
            system_prompt="""Eres un experto en crear preguntas y respuestas educativas.
Tu tarea es generar preguntas factuales claras y sus respuestas basándote en el contenido proporcionado.
Las preguntas deben ser directas y las respuestas deben ser precisas y completas.""",
            user_prompt="""Basándote en el siguiente contenido, genera 3-5 preguntas factuales diferentes con sus respuestas.
Cada pregunta debe enfocarse en hechos específicos, definiciones o datos concretos.

Contenido:
{content}

Genera las preguntas en el siguiente formato JSON:
{{
    "questions": [
        {{
            "question": "¿Cuál es...?",
            "answer": "La respuesta completa...",
            "key_facts": ["hecho1", "hecho2"]
        }}
    ]
}}""",
            question_type=QuestionType.FACTUAL,
            difficulty=DifficultyLevel.BASIC
        )
        
        # Factual Intermediate Questions Template
        templates['factual_intermediate'] = PromptTemplate(
            name="Factual Intermediate",
            system_prompt="""Eres un experto en crear preguntas y respuestas educativas de nivel intermedio.
Tu tarea es generar preguntas factuales que requieran comprensión más profunda del contenido.
Las preguntas deben ir más allá de simples definiciones e incluir relaciones entre conceptos.""",
            user_prompt="""Basándote en el siguiente contenido, genera 3-4 preguntas factuales de nivel intermedio con sus respuestas.
Las preguntas deben requerir comprensión de conceptos y sus relaciones.

Contenido:
{content}

Contexto adicional:
{context}

Genera las preguntas en el siguiente formato JSON:
{{
    "questions": [
        {{
            "question": "¿Cómo se relaciona X con Y?",
            "answer": "Explicación detallada de la relación...",
            "key_concepts": ["concepto1", "concepto2"],
            "relationships": ["relación1", "relación2"]
        }}
    ]
}}""",
            question_type=QuestionType.FACTUAL,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # Synthesis Questions Template
        templates['synthesis_intermediate'] = PromptTemplate(
            name="Synthesis Intermediate",
            system_prompt="""Eres un experto en crear preguntas que requieren síntesis de información.
Tu tarea es generar preguntas que combinen conceptos de múltiples fuentes y sus respuestas comprehensivas.
Las preguntas deben requerir comprensión profunda y conexión entre ideas.""",
            user_prompt="""Basándote en los siguientes chunks de contenido, genera 2-3 preguntas de síntesis que requieran 
información de múltiples chunks para responder completamente.

Chunk 1 (ID: {chunk1_id}):
{chunk1_content}

Chunk 2 (ID: {chunk2_id}):
{chunk2_content}

{additional_chunks}

Genera preguntas que conecten conceptos entre los chunks:
{{
    "questions": [
        {{
            "question": "¿Cómo se relaciona [concepto del chunk 1] con [concepto del chunk 2]?",
            "answer": "Respuesta que integra información de ambos chunks...",
            "required_chunks": [{chunk1_id}, {chunk2_id}],
            "key_connections": ["conexión1", "conexión2"]
        }}
    ]
}}""",
            question_type=QuestionType.SYNTHESIS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            requires_multi_chunk=True
        )
        
        # Causal/Reasoning Questions Template
        templates['causal_advanced'] = PromptTemplate(
            name="Causal Advanced",
            system_prompt="""Eres un experto en crear preguntas que requieren razonamiento causal y análisis profundo.
Tu tarea es generar preguntas sobre causas, efectos, y relaciones causales basándote en el contenido.
Las respuestas deben explicar claramente las cadenas causales y el razonamiento.""",
            user_prompt="""Analiza el siguiente contenido y genera 2-3 preguntas que requieran razonamiento causal:

Contenido:
{content}

Contexto adicional:
{context}

Genera preguntas sobre causas y efectos:
{{
    "questions": [
        {{
            "question": "¿Por qué...? o ¿Cuáles son las consecuencias de...?",
            "answer": "Explicación detallada de la cadena causal...",
            "causal_chain": ["causa1 → efecto1", "efecto1 → consecuencia"],
            "reasoning_steps": ["paso1", "paso2", "paso3"]
        }}
    ]
}}""",
            question_type=QuestionType.CAUSAL,
            difficulty=DifficultyLevel.ADVANCED
        )
        
        # Application Questions Template
        templates['application_intermediate'] = PromptTemplate(
            name="Application Intermediate",
            system_prompt="""Eres un experto en crear preguntas de aplicación práctica.
Tu tarea es generar preguntas que requieran aplicar conceptos a situaciones nuevas o resolver problemas.
Las respuestas deben demostrar cómo aplicar el conocimiento de manera práctica.""",
            user_prompt="""Basándote en el contenido, genera 2-3 preguntas de aplicación práctica:

Contenido principal:
{content}

Genera preguntas que requieran aplicar estos conceptos:
{{
    "questions": [
        {{
            "question": "¿Cómo aplicarías [concepto] para resolver [situación]?",
            "answer": "Explicación paso a paso de la aplicación...",
            "application_steps": ["paso1", "paso2"],
            "required_concepts": ["concepto1", "concepto2"]
        }}
    ]
}}""",
            question_type=QuestionType.APPLICATION,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # Comparison Questions Template
        templates['comparison_intermediate'] = PromptTemplate(
            name="Comparison Intermediate",
            system_prompt="""Eres un experto en crear preguntas comparativas.
Tu tarea es generar preguntas que requieran comparar y contrastar conceptos, métodos o elementos.
Las respuestas deben destacar similitudes y diferencias clave.""",
            user_prompt="""Analiza el contenido y genera 2-3 preguntas de comparación:

Contenido:
{content}

Genera preguntas comparativas:
{{
    "questions": [
        {{
            "question": "¿Cuáles son las diferencias principales entre [A] y [B]?",
            "answer": "Análisis comparativo detallado...",
            "similarities": ["similitud1", "similitud2"],
            "differences": ["diferencia1", "diferencia2"],
            "comparison_criteria": ["criterio1", "criterio2"]
        }}
    ]
}}""",
            question_type=QuestionType.COMPARISON,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # Analysis Questions Template
        templates['analysis_advanced'] = PromptTemplate(
            name="Analysis Advanced",
            system_prompt="""Eres un experto en crear preguntas analíticas profundas.
Tu tarea es generar preguntas que requieran descomponer conceptos complejos, evaluar componentes y 
entender relaciones profundas entre elementos.""",
            user_prompt="""Estudia el contenido y genera 2-3 preguntas de análisis profundo:

Contenido principal:
{content}

Contexto extendido:
{context}

Genera preguntas analíticas:
{{
    "questions": [
        {{
            "question": "Analiza los componentes de [sistema/concepto] y explica cómo interactúan",
            "answer": "Análisis detallado de componentes e interacciones...",
            "components": ["componente1", "componente2"],
            "interactions": ["interacción1", "interacción2"],
            "analysis_framework": "framework utilizado"
        }}
    ]
}}""",
            question_type=QuestionType.ANALYSIS,
            difficulty=DifficultyLevel.ADVANCED
        )
        
        return templates
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific template by name"""
        return self.templates.get(template_name)
    
    def get_templates_by_type(self, question_type: QuestionType) -> List[PromptTemplate]:
        """Get all templates of a specific question type"""
        return [t for t in self.templates.values() if t.question_type == question_type]
    
    def get_templates_by_difficulty(self, difficulty: DifficultyLevel) -> List[PromptTemplate]:
        """Get all templates of a specific difficulty"""
        return [t for t in self.templates.values() if t.difficulty == difficulty]
    
    def get_multi_chunk_templates(self) -> List[PromptTemplate]:
        """Get templates that require multiple chunks"""
        return [t for t in self.templates.values() if t.requires_multi_chunk]
    
    def get_variation_prompt(self) -> str:
        """Get prompt for generating question variations"""
        return """Dada la siguiente pregunta original, genera 3-5 variaciones que mantengan el mismo significado
pero usen diferentes formulaciones, estructuras o enfoques:

Pregunta original: {original_question}

Genera las variaciones en formato JSON:
{{
    "variations": [
        "¿Variación 1...?",
        "¿Variación 2...?",
        "¿Variación 3...?"
    ]
}}

Las variaciones deben:
- Mantener el mismo significado esencial
- Usar diferentes palabras o estructuras
- Variar en longitud y complejidad
- Ser naturales y fluidas en español"""
    
    def get_quality_check_prompt(self) -> str:
        """Get prompt for quality evaluation"""
        return """Evalúa la calidad de la siguiente pregunta y respuesta:

Pregunta: {question}
Respuesta: {answer}
Tipo: {question_type}
Dificultad: {difficulty}

Evalúa según estos criterios:
1. Claridad de la pregunta (0-1)
2. Completitud de la respuesta (0-1)
3. Precisión técnica (0-1)
4. Relevancia al contenido fuente (0-1)
5. Nivel de dificultad apropiado (0-1)

Responde en formato JSON:
{{
    "clarity_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.95,
    "relevance_score": 0.9,
    "difficulty_appropriate": 0.85,
    "overall_quality": 0.88,
    "feedback": "Comentarios específicos sobre mejoras...",
    "issues": ["problema1", "problema2"]
}}"""
    
    def format_system_message(self, domain: str, context: str) -> str:
        """Format the system message for fine-tuning"""
        return f"""Eres un asistente experto en {domain}. Tienes acceso al siguiente contexto: {context}"""