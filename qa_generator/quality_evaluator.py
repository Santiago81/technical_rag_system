"""
Quality evaluation system for generated Q&A pairs
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Quality metrics for evaluation"""
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    DIFFICULTY_MATCH = "difficulty_match"
    LANGUAGE_QUALITY = "language_quality"

@dataclass
class QualityScore:
    """Quality score for a Q&A pair"""
    clarity: float
    completeness: float
    accuracy: float
    relevance: float
    difficulty_match: float
    language_quality: float
    overall: float
    issues: List[str]
    suggestions: List[str]
    
    def to_dict(self) -> dict:
        return {
            'clarity': self.clarity,
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'relevance': self.relevance,
            'difficulty_match': self.difficulty_match,
            'language_quality': self.language_quality,
            'overall': self.overall,
            'issues': self.issues,
            'suggestions': self.suggestions
        }
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if quality meets threshold"""
        return self.overall >= threshold

class QualityEvaluator:
    """Evaluates quality of generated Q&A pairs"""
    
    def __init__(self):
        # Common quality issues patterns
        self.quality_patterns = {
            'vague_terms': r'\b(algo|cosa|eso|esto|aquello)\b',
            'excessive_length': 500,  # characters
            'minimum_length': 20,  # characters
            'repetition': r'\b(\w+)\b.*\b\1\b.*\b\1\b',  # word repeated 3+ times
            'incomplete_sentence': r'[^.!?]$',
            'yes_no_only': r'^(Sí|No|Si)\.?$',
            'numbered_list': r'^\d+\.',
            'proper_punctuation': r'[.!?]$'
        }
        
        # Difficulty indicators
        self.difficulty_indicators = {
            'basic': ['qué es', 'cuál es', 'define', 'nombra', 'lista'],
            'intermediate': ['explica', 'describe', 'cómo', 'por qué', 'compara'],
            'advanced': ['analiza', 'evalúa', 'sintetiza', 'critica', 'diseña', 'propón']
        }
    
    def evaluate_qa_pair(self, question: str, answer: str, 
                        source_content: str, 
                        expected_difficulty: str = None,
                        question_type: str = None) -> QualityScore:
        """Evaluate a Q&A pair comprehensively"""
        
        scores = {}
        issues = []
        suggestions = []
        
        # Evaluate clarity
        clarity_score, clarity_issues = self._evaluate_clarity(question, answer)
        scores['clarity'] = clarity_score
        issues.extend(clarity_issues)
        
        # Evaluate completeness
        completeness_score, completeness_issues = self._evaluate_completeness(answer, question)
        scores['completeness'] = completeness_score
        issues.extend(completeness_issues)
        
        # Evaluate accuracy/relevance
        relevance_score = self._evaluate_relevance(question, answer, source_content)
        scores['relevance'] = relevance_score
        scores['accuracy'] = relevance_score  # Simplified for now
        
        # Evaluate difficulty match
        if expected_difficulty:
            difficulty_score = self._evaluate_difficulty_match(question, expected_difficulty)
            scores['difficulty_match'] = difficulty_score
        else:
            scores['difficulty_match'] = 1.0
        
        # Evaluate language quality
        language_score, language_issues = self._evaluate_language_quality(question, answer)
        scores['language_quality'] = language_score
        issues.extend(language_issues)
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(scores, issues)
        
        return QualityScore(
            clarity=scores['clarity'],
            completeness=scores['completeness'],
            accuracy=scores['accuracy'],
            relevance=scores['relevance'],
            difficulty_match=scores['difficulty_match'],
            language_quality=scores['language_quality'],
            overall=overall_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _evaluate_clarity(self, question: str, answer: str) -> Tuple[float, List[str]]:
        """Evaluate clarity of Q&A"""
        score = 1.0
        issues = []
        
        # Check for vague terms
        if re.search(self.quality_patterns['vague_terms'], question.lower()):
            score -= 0.2
            issues.append("Question contains vague terms")
        
        if re.search(self.quality_patterns['vague_terms'], answer.lower()):
            score -= 0.1
            issues.append("Answer contains vague terms")
        
        # Check question structure
        if not question.strip().endswith('?'):
            score -= 0.1
            issues.append("Question missing proper punctuation")
        
        # Check for clear question words
        question_words = ['qué', 'cómo', 'cuándo', 'dónde', 'por qué', 'cuál', 'quién']
        has_question_word = any(word in question.lower() for word in question_words)
        if not has_question_word:
            score -= 0.1
            issues.append("Question lacks clear interrogative word")
        
        return max(0, score), issues
    
    def _evaluate_completeness(self, answer: str, question: str) -> Tuple[float, List[str]]:
        """Evaluate completeness of answer"""
        score = 1.0
        issues = []
        
        # Check length
        if len(answer) < self.quality_patterns['minimum_length']:
            score -= 0.3
            issues.append("Answer too short")
        elif len(answer) > self.quality_patterns['excessive_length']:
            score -= 0.1
            issues.append("Answer might be too verbose")
        
        # Check for yes/no only answers
        if re.match(self.quality_patterns['yes_no_only'], answer.strip()):
            score -= 0.5
            issues.append("Answer is only yes/no without explanation")
        
        # Check if answer addresses the question type
        if 'por qué' in question.lower() and 'porque' not in answer.lower():
            score -= 0.2
            issues.append("'Why' question not properly answered with reasoning")
        
        if 'cómo' in question.lower() and not any(word in answer.lower() 
                                                  for word in ['primero', 'luego', 'después', 'finalmente', 'paso']):
            score -= 0.1
            issues.append("'How' question might lack procedural explanation")
        
        return max(0, score), issues
    
    def _evaluate_relevance(self, question: str, answer: str, source_content: str) -> float:
        """Evaluate relevance to source content"""
        score = 1.0
        
        # Simple keyword overlap check
        source_words = set(source_content.lower().split())
        qa_words = set((question + " " + answer).lower().split())
        
        # Remove common words
        common_words = {'el', 'la', 'de', 'en', 'y', 'a', 'que', 'es', 'un', 'una', 'los', 'las'}
        source_words -= common_words
        qa_words -= common_words
        
        # Calculate overlap
        overlap = len(source_words.intersection(qa_words))
        overlap_ratio = overlap / max(len(source_words), 1)
        
        # Adjust score based on overlap
        if overlap_ratio < 0.1:
            score = 0.3
        elif overlap_ratio < 0.2:
            score = 0.6
        elif overlap_ratio < 0.3:
            score = 0.8
        
        return score
    
    def _evaluate_difficulty_match(self, question: str, expected_difficulty: str) -> float:
        """Evaluate if question matches expected difficulty"""
        question_lower = question.lower()
        
        # Count indicators for each difficulty level
        indicator_counts = {}
        for level, indicators in self.difficulty_indicators.items():
            count = sum(1 for indicator in indicators if indicator in question_lower)
            indicator_counts[level] = count
        
        # Determine detected difficulty
        detected_difficulty = max(indicator_counts, key=indicator_counts.get)
        
        # Score based on match
        if detected_difficulty == expected_difficulty:
            return 1.0
        elif (expected_difficulty == 'intermediate' and 
              detected_difficulty in ['basic', 'advanced']):
            return 0.7
        else:
            return 0.4
    
    def _evaluate_language_quality(self, question: str, answer: str) -> Tuple[float, List[str]]:
        """Evaluate language quality"""
        score = 1.0
        issues = []
        
        # Check punctuation
        if not re.search(self.quality_patterns['proper_punctuation'], answer):
            score -= 0.1
            issues.append("Answer missing proper ending punctuation")
        
        # Check for repetition
        if re.search(self.quality_patterns['repetition'], answer):
            score -= 0.2
            issues.append("Answer contains excessive repetition")
        
        # Check for complete sentences
        sentences = answer.split('.')
        incomplete = sum(1 for s in sentences[:-1] if len(s.strip()) < 10)
        if incomplete > 0:
            score -= 0.1 * incomplete
            issues.append(f"{incomplete} potentially incomplete sentences")
        
        return max(0, score), issues
    
    def _generate_suggestions(self, scores: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if scores['clarity'] < 0.8:
            suggestions.append("Reformulate question with clearer, more specific terms")
        
        if scores['completeness'] < 0.8:
            suggestions.append("Expand answer with more detail and explanation")
        
        if scores['relevance'] < 0.8:
            suggestions.append("Ensure Q&A directly relates to source content")
        
        if scores['difficulty_match'] < 0.8:
            suggestions.append("Adjust question complexity to match target difficulty")
        
        if scores['language_quality'] < 0.8:
            suggestions.append("Review grammar, punctuation, and sentence structure")
        
        return suggestions
    
    def evaluate_batch(self, qa_pairs: List[Dict], source_contents: List[str]) -> List[QualityScore]:
        """Evaluate a batch of Q&A pairs"""
        results = []
        
        for qa, source in zip(qa_pairs, source_contents):
            score = self.evaluate_qa_pair(
                qa['question'],
                qa['answer'],
                source,
                qa.get('difficulty'),
                qa.get('question_type')
            )
            results.append(score)
        
        return results
    
    def filter_by_quality(self, qa_pairs: List[Dict], 
                         scores: List[QualityScore], 
                         threshold: float = 0.7) -> List[Dict]:
        """Filter Q&A pairs by quality threshold"""
        filtered = []
        
        for qa, score in zip(qa_pairs, scores):
            if score.is_acceptable(threshold):
                qa['quality_score'] = score.overall
                filtered.append(qa)
            else:
                logger.info(f"Filtered out Q&A with score {score.overall}: {qa['question'][:50]}...")
        
        return filtered
    
    def get_quality_statistics(self, scores: List[QualityScore]) -> Dict:
        """Get statistics about quality scores"""
        if not scores:
            return {}
        
        metrics = ['clarity', 'completeness', 'accuracy', 'relevance', 
                  'difficulty_match', 'language_quality', 'overall']
        
        stats = {}
        for metric in metrics:
            values = [getattr(score, metric) for score in scores]
            stats[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'above_threshold': sum(1 for v in values if v >= 0.7) / len(values)
            }
        
        # Common issues
        all_issues = []
        for score in scores:
            all_issues.extend(score.issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        stats['common_issues'] = sorted(issue_counts.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:5]
        
        return stats
    
    async def evaluate_single(self, question: str, answer: str, source_content: str) -> Tuple[bool, float]:
        """Simple async evaluation method for single Q&A pair
        
        Returns:
            Tuple of (is_valid, quality_score)
        """
        try:
            # Basic validation
            if not question or not answer:
                return False, 0.0
                
            if len(question) < 10 or len(answer) < 20:
                return False, 0.0
            
            # Check for generic patterns that indicate low quality
            generic_patterns = [
                "what is shown", "what can you tell", "describe the", 
                "what information", "based on the text", "according to",
                "qué se muestra", "qué puedes decir", "describe el",
                "qué información", "basado en el texto", "según"
            ]
            
            question_lower = question.lower()
            if any(pattern in question_lower for pattern in generic_patterns):
                return False, 0.0
            
            # Perform full evaluation
            score = self.evaluate_qa_pair(question, answer, source_content)
            
            # Return validation result
            is_valid = score.overall >= 0.6
            return is_valid, score.overall
            
        except Exception as e:
            logger.error(f"Error evaluating Q&A: {e}")
            return False, 0.0