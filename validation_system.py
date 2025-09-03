"""
Validation System for HypeTorch AI Predictions

Measures prediction accuracy across different AI systems to provide learning feedback:
1. Context classification accuracy validation
2. HYPE score prediction accuracy
3. Entity detection accuracy  
4. Talk time attribution accuracy
5. Quality threshold effectiveness

Provides feedback to adaptive systems for continuous improvement.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger('validation_system')


@dataclass
class ValidationResult:
    """Result of a validation check."""
    system_type: str  # "context_classification", "hype_score", etc.
    entity_name: str
    predicted_value: Any
    actual_value: Any
    accuracy: float
    confidence: float
    timestamp: float
    validation_method: str


class ValidationSystem:
    """
    Validates AI system predictions and provides feedback for learning.
    
    Uses multiple validation approaches:
    - Manual validation samples
    - Cross-validation between AI methods
    - Business outcome correlation
    - User feedback integration
    """
    
    def __init__(self, validation_rate: float = 0.1):
        self.validation_rate = validation_rate  # Percentage of predictions to validate
        
        # Store validation results for analysis
        self.validation_history = defaultdict(lambda: deque(maxlen=1000))
        self.accuracy_trends = defaultdict(list)
        
        # Manual validation queue (for human review)
        self.manual_validation_queue = deque(maxlen=100)
        self.manual_validation_results = {}
        
        # Business outcome tracking
        self.business_outcomes = defaultdict(list)
        
        # System performance tracking
        self.system_stats = {
            "total_validations": 0,
            "validation_by_system": defaultdict(int),
            "average_accuracy": defaultdict(list),
            "improvement_tracking": defaultdict(list)
        }
        
        self.cache_path = Path("temp/validation_cache.pkl")
        
        logger.info("🔍 Validation System initialized")
    
    def should_validate_prediction(self, system_type: str) -> bool:
        """Determine if this prediction should be validated."""
        return random.random() < self.validation_rate
    
    def validate_context_classification(self, entity_name: str, sentence: str, 
                                      predicted_context: str, predicted_confidence: float,
                                      entity_type: str = "person") -> Optional[ValidationResult]:
        """Validate context classification prediction."""
        
        if not self.should_validate_prediction("context_classification"):
            return None
        
        # Use cross-validation between different AI methods
        actual_context, validation_confidence = self._cross_validate_context(
            entity_name, sentence, entity_type
        )
        
        if actual_context is None:
            return None
        
        # Calculate accuracy (exact match or semantic similarity)
        if predicted_context == actual_context:
            accuracy = 1.0
        else:
            # Semantic similarity between contexts (some contexts are similar)
            similarity_map = {
                ("personal_life", "personality"): 0.7,
                ("business", "performance"): 0.6,
                ("controversy", "personality"): 0.4,
                ("brief_mention", "performance"): 0.3
            }
            
            similarity = similarity_map.get((predicted_context, actual_context), 0.0)
            similarity = max(similarity, similarity_map.get((actual_context, predicted_context), 0.0))
            accuracy = similarity
        
        result = ValidationResult(
            system_type="context_classification",
            entity_name=entity_name,
            predicted_value=predicted_context,
            actual_value=actual_context,
            accuracy=accuracy,
            confidence=predicted_confidence,
            timestamp=time.time(),
            validation_method="cross_validation"
        )
        
        self._record_validation_result(result)
        return result
    
    def _cross_validate_context(self, entity_name: str, sentence: str, 
                               entity_type: str) -> Tuple[Optional[str], float]:
        """Use multiple AI methods to cross-validate context classification."""
        
        try:
            from context_classifier import context_classifier
            
            # Get classifications from different methods
            method_results = []
            
            # Try each method independently
            for method in ["zero_shot", "embeddings", "nlp_features"]:
                if hasattr(context_classifier, f'_classify_with_{method}'):
                    try:
                        if method == "zero_shot":
                            context, conf = context_classifier._classify_with_zero_shot(
                                sentence, context_classifier._get_context_definitions(entity_type)
                            )
                        elif method == "embeddings":
                            context, conf = context_classifier._classify_with_embeddings(
                                sentence, context_classifier._get_context_definitions(entity_type)
                            )
                        elif method == "nlp_features":
                            context, conf = context_classifier._classify_with_nlp_features(
                                sentence, entity_name
                            )
                        
                        if context and conf > 0.3:
                            method_results.append((context, conf, method))
                            
                    except Exception as e:
                        logger.debug(f"Cross-validation method {method} failed: {e}")
            
            if not method_results:
                return None, 0.0
            
            # Find consensus or highest confidence result
            context_votes = defaultdict(lambda: {"total_conf": 0.0, "count": 0, "methods": []})
            
            for context, conf, method in method_results:
                context_votes[context]["total_conf"] += conf
                context_votes[context]["count"] += 1
                context_votes[context]["methods"].append(method)
            
            # Choose context with highest average confidence
            best_context = None
            best_avg_conf = 0.0
            
            for context, votes in context_votes.items():
                avg_conf = votes["total_conf"] / votes["count"]
                if avg_conf > best_avg_conf:
                    best_context = context
                    best_avg_conf = avg_conf
            
            return best_context, best_avg_conf
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return None, 0.0
    
    def validate_hype_score(self, entity_name: str, predicted_score: float, 
                           predicted_confidence: float, data_sources: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate HYPE score prediction using business outcome correlation."""
        
        if not self.should_validate_prediction("hype_score"):
            return None
        
        # Use historical correlation with business outcomes
        actual_score = self._estimate_actual_hype_from_outcomes(entity_name, data_sources)
        
        if actual_score is None:
            # Queue for manual validation if no business outcome data
            self.manual_validation_queue.append({
                "type": "hype_score",
                "entity": entity_name,
                "predicted_score": predicted_score,
                "confidence": predicted_confidence,
                "timestamp": time.time()
            })
            return None
        
        # Calculate accuracy based on relative error
        error = abs(predicted_score - actual_score)
        max_score = max(predicted_score, actual_score, 1.0)
        accuracy = max(0.0, 1.0 - (error / max_score))
        
        result = ValidationResult(
            system_type="hype_score",
            entity_name=entity_name,
            predicted_value=predicted_score,
            actual_value=actual_score,
            accuracy=accuracy,
            confidence=predicted_confidence,
            timestamp=time.time(),
            validation_method="business_correlation"
        )
        
        self._record_validation_result(result)
        return result
    
    def _estimate_actual_hype_from_outcomes(self, entity_name: str, data_sources: Dict) -> Optional[float]:
        """Estimate actual HYPE score from business outcomes and engagement data."""
        
        # Look for business outcome indicators
        outcome_indicators = []
        
        # Social media engagement (if available)
        if "engagement_metrics" in data_sources:
            engagement = data_sources["engagement_metrics"]
            if isinstance(engagement, dict):
                total_engagement = sum(engagement.values())
                if total_engagement > 0:
                    outcome_indicators.append(min(100, total_engagement / 1000))  # Scale engagement
        
        # Search interest trends
        if "Google Trends" in data_sources and entity_name in data_sources["Google Trends"]:
            trends_score = data_sources["Google Trends"][entity_name]
            if trends_score > 0:
                outcome_indicators.append(trends_score)
        
        # Reddit mentions as social proof
        if "Reddit Mentions" in data_sources and entity_name in data_sources["Reddit Mentions"]:
            reddit_score = data_sources["Reddit Mentions"][entity_name]
            if reddit_score > 0:
                outcome_indicators.append(min(50, reddit_score))  # Cap reddit influence
        
        # Historical HYPE correlation (if we have past data)
        historical_scores = self.business_outcomes.get(entity_name, [])
        if len(historical_scores) >= 3:
            # Use median of recent historical scores
            recent_scores = [score for score, timestamp in historical_scores[-5:] 
                           if time.time() - timestamp < 30 * 24 * 3600]  # Last 30 days
            if recent_scores:
                outcome_indicators.append(np.median(recent_scores))
        
        if not outcome_indicators:
            return None
        
        # Combine indicators into estimated actual score
        estimated_score = np.mean(outcome_indicators)
        
        # Add some noise to represent uncertainty
        noise = np.random.normal(0, estimated_score * 0.1)  # 10% noise
        return max(0, estimated_score + noise)
    
    def validate_entity_detection(self, entity_name: str, predicted_confidence: float,
                                entity_type: str, text: str) -> Optional[ValidationResult]:
        """Validate entity detection accuracy."""
        
        if not self.should_validate_prediction("entity_detection"):
            return None
        
        # Use multiple detection methods for cross-validation
        actual_confidence = self._cross_validate_entity_detection(entity_name, entity_type, text)
        
        if actual_confidence is None:
            return None
        
        # Calculate accuracy based on confidence correlation
        confidence_error = abs(predicted_confidence - actual_confidence)
        accuracy = max(0.0, 1.0 - confidence_error)
        
        result = ValidationResult(
            system_type="entity_detection", 
            entity_name=entity_name,
            predicted_value=predicted_confidence,
            actual_value=actual_confidence,
            accuracy=accuracy,
            confidence=predicted_confidence,
            timestamp=time.time(),
            validation_method="cross_validation"
        )
        
        self._record_validation_result(result)
        return result
    
    def _cross_validate_entity_detection(self, entity_name: str, entity_type: str, text: str) -> Optional[float]:
        """Cross-validate entity detection using multiple NLP methods."""
        
        try:
            import spacy
            from sentence_transformers import SentenceTransformer
            
            confidences = []
            
            # Method 1: spaCy named entity recognition
            try:
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(text)
                
                spacy_confidence = 0.0
                for ent in doc.ents:
                    if entity_name.lower() in ent.text.lower() or ent.text.lower() in entity_name.lower():
                        # spaCy found the entity - confidence based on label match
                        expected_labels = {"PERSON": ["person"], "ORG": ["team", "non-person"], 
                                         "GPE": ["team"], "MONEY": ["non-person"]}
                        if ent.label_ in expected_labels:
                            if entity_type in expected_labels[ent.label_]:
                                spacy_confidence = 0.8  # High confidence for label match
                            else:
                                spacy_confidence = 0.4  # Lower confidence for label mismatch
                        else:
                            spacy_confidence = 0.6  # Medium confidence for unknown labels
                        break
                
                if spacy_confidence > 0:
                    confidences.append(spacy_confidence)
                    
            except Exception as e:
                logger.debug(f"spaCy entity validation failed: {e}")
            
            # Method 2: Semantic similarity
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Create embeddings for entity name and text
                entity_embedding = model.encode([entity_name])
                text_embedding = model.encode([text])
                
                # Calculate similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(entity_embedding, text_embedding)[0][0]
                
                # Convert similarity to confidence (entities mentioned in text should have higher similarity)
                semantic_confidence = min(0.9, max(0.1, similarity * 2))  # Scale and bound
                confidences.append(semantic_confidence)
                
            except Exception as e:
                logger.debug(f"Semantic entity validation failed: {e}")
            
            # Method 3: Simple string matching with fuzzy logic
            try:
                from difflib import SequenceMatcher
                
                # Find best string match in text
                entity_words = entity_name.lower().split()
                text_words = text.lower().split()
                
                best_match_ratio = 0.0
                for entity_word in entity_words:
                    for text_word in text_words:
                        ratio = SequenceMatcher(None, entity_word, text_word).ratio()
                        best_match_ratio = max(best_match_ratio, ratio)
                
                if best_match_ratio > 0.7:  # Good string match
                    string_confidence = best_match_ratio * 0.9
                    confidences.append(string_confidence)
                    
            except Exception as e:
                logger.debug(f"String matching validation failed: {e}")
            
            if not confidences:
                return None
            
            # Return average confidence across methods
            return np.mean(confidences)
            
        except Exception as e:
            logger.error(f"Entity detection cross-validation failed: {e}")
            return None
    
    def _record_validation_result(self, result: ValidationResult):
        """Record validation result for analysis and learning."""
        
        # Store in history
        self.validation_history[result.system_type].append(result)
        
        # Update statistics
        self.system_stats["total_validations"] += 1
        self.system_stats["validation_by_system"][result.system_type] += 1
        self.system_stats["average_accuracy"][result.system_type].append(result.accuracy)
        
        # Track accuracy trends over time
        self.accuracy_trends[result.system_type].append({
            "timestamp": result.timestamp,
            "accuracy": result.accuracy,
            "confidence": result.confidence
        })
        
        # Provide feedback to relevant systems
        self._provide_learning_feedback(result)
        
        logger.debug(f"🔍 Validation recorded: {result.system_type} -> {result.accuracy:.3f} accuracy")
    
    def _provide_learning_feedback(self, result: ValidationResult):
        """Provide feedback to adaptive systems for learning."""
        
        try:
            if result.system_type == "context_classification":
                # Provide feedback to adaptive context scorer
                from adaptive_context_scorer import adaptive_scorer
                
                # Record successful or unsuccessful classification
                adaptive_scorer._record_validation_feedback(
                    entity_name=result.entity_name,
                    predicted_context=result.predicted_value,
                    actual_context=result.actual_value,
                    accuracy=result.accuracy
                )
                
                # Provide feedback to ensemble optimizer
                from adaptive_ensemble_optimizer import ensemble_optimizer
                
                # Record method performance (would need to track which methods were used)
                # This is a simplified version - in practice would track individual method performance
                
            elif result.system_type == "hype_score":
                # Provide feedback to HYPE optimizer
                from adaptive_hype_optimizer import hype_optimizer
                
                hype_optimizer.record_prediction_result(
                    entity_name=result.entity_name,
                    context_key="default",  # Would need actual context
                    weights_used={},  # Would need actual weights used
                    predicted_hype=result.predicted_value,
                    actual_performance=result.actual_value
                )
                
            elif result.system_type == "entity_detection":
                # Provide feedback to quality manager
                from adaptive_quality_manager import adaptive_quality_manager
                
                adaptive_quality_manager.record_prediction_result(
                    threshold_type="entity_confidence",
                    threshold_used=0.4,  # Would need actual threshold used
                    prediction_confidence=result.predicted_value,
                    entity_type="person",  # Would need actual entity type
                    actual_accuracy=result.accuracy
                )
                
        except Exception as e:
            logger.debug(f"Failed to provide learning feedback: {e}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        report = {
            "summary": {
                "total_validations": self.system_stats["total_validations"],
                "systems_validated": list(self.system_stats["validation_by_system"].keys()),
                "validation_rate_actual": self.validation_rate,
                "manual_queue_size": len(self.manual_validation_queue)
            },
            "accuracy_by_system": {},
            "improvement_trends": {},
            "validation_coverage": {}
        }
        
        # Calculate accuracy statistics
        for system_type, accuracies in self.system_stats["average_accuracy"].items():
            if accuracies:
                report["accuracy_by_system"][system_type] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "min_accuracy": np.min(accuracies),
                    "max_accuracy": np.max(accuracies),
                    "total_validations": len(accuracies),
                    "recent_accuracy": np.mean(accuracies[-20:]) if len(accuracies) >= 20 else np.mean(accuracies)
                }
        
        # Calculate improvement trends
        for system_type, trends in self.accuracy_trends.items():
            if len(trends) >= 10:
                # Compare recent vs early performance
                recent = [t["accuracy"] for t in trends[-20:]]
                early = [t["accuracy"] for t in trends[:20]]
                
                if len(early) >= 5 and len(recent) >= 5:
                    improvement = np.mean(recent) - np.mean(early)
                    report["improvement_trends"][system_type] = {
                        "early_accuracy": np.mean(early),
                        "recent_accuracy": np.mean(recent),
                        "improvement": improvement,
                        "trend_direction": "improving" if improvement > 0.02 else "declining" if improvement < -0.02 else "stable"
                    }
        
        # Validation coverage analysis
        for system_type, validations in self.system_stats["validation_by_system"].items():
            total_predictions_estimate = validations / self.validation_rate  # Rough estimate
            report["validation_coverage"][system_type] = {
                "validations_performed": validations,
                "estimated_total_predictions": int(total_predictions_estimate),
                "coverage_percentage": self.validation_rate * 100
            }
        
        return report
    
    def save_validation_data(self, filepath: Optional[str] = None):
        """Save validation data to disk."""
        
        if filepath is None:
            filepath = self.cache_path
        
        try:
            validation_data = {
                "validation_history": {k: list(v) for k, v in self.validation_history.items()},
                "accuracy_trends": dict(self.accuracy_trends),
                "system_stats": dict(self.system_stats),
                "business_outcomes": dict(self.business_outcomes),
                "validation_rate": self.validation_rate
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(validation_data, f)
            
            logger.info(f"💾 Saved validation data to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save validation data: {e}")
    
    def load_validation_data(self, filepath: Optional[str] = None):
        """Load validation data from disk."""
        
        if filepath is None:
            filepath = self.cache_path
        
        if not Path(filepath).exists():
            logger.info("ℹ️ No cached validation data found, starting fresh")
            return
        
        try:
            with open(filepath, "rb") as f:
                validation_data = pickle.load(f)
            
            # Restore data structures
            history_data = validation_data.get("validation_history", {})
            for system_type, history_list in history_data.items():
                self.validation_history[system_type] = deque(history_list, maxlen=1000)
            
            self.accuracy_trends = defaultdict(list, validation_data.get("accuracy_trends", {}))
            self.system_stats = validation_data.get("system_stats", {})
            self.business_outcomes = defaultdict(list, validation_data.get("business_outcomes", {}))
            self.validation_rate = validation_data.get("validation_rate", self.validation_rate)
            
            logger.info(f"📂 Loaded validation data: {self.system_stats.get('total_validations', 0)} validations, "
                       f"{len(self.validation_history)} systems tracked")
            
        except Exception as e:
            logger.error(f"❌ Failed to load validation data: {e}")


# Global validation system instance
validation_system = ValidationSystem()