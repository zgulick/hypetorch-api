"""
Adaptive Context Scoring System for HypeTorch

Replaces hardcoded scoring rules with AI-driven semantic pattern learning.
Uses transformer embeddings and similarity scoring to dynamically classify contexts.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

logger = logging.getLogger('adaptive_context')


class AdaptiveContextScorer:
    """
    AI-driven context scoring that learns from examples and adapts over time.
    
    Replaces hardcoded scoring rules with:
    1. Semantic pattern learning from training examples
    2. Dynamic threshold adaptation based on accuracy
    3. Context-specific confidence calibration
    4. Entity-type aware scoring
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.context_patterns = {}
        self.adaptive_thresholds = {}
        self.performance_history = defaultdict(list)
        self.entity_type_modifiers = {}
        self.cache_path = Path("temp/adaptive_context_cache.pkl")
        
        # Initialize with sensible defaults that will be overridden by learning
        self.base_confidence = 0.5
        self.learning_rate = 0.1
        
        logger.info("🧠 Initializing Adaptive Context Scorer")
        
    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                logger.info(f"🔄 Loading semantic model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("✅ Semantic model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load semantic model: {e}")
                raise
    
    def learn_context_patterns(self, context_definitions: Dict[str, Dict]) -> None:
        """
        Learn semantic patterns from context definition examples.
        
        Args:
            context_definitions: Dictionary with context types and their training examples
        """
        self._ensure_model_loaded()
        
        logger.info("🎓 Learning semantic patterns from training examples...")
        
        for context_type, definition in context_definitions.items():
            examples = definition.get("training_examples", definition.get("examples", []))
            if not examples:
                logger.warning(f"⚠️ No examples found for context: {context_type}")
                continue
                
            # Generate embeddings for all examples
            try:
                embeddings = self.model.encode(examples, convert_to_tensor=False)
                
                # Create semantic centroid for this context
                centroid = np.mean(embeddings, axis=0)
                
                # Calculate intra-context variance for threshold adaptation
                similarities = cosine_similarity([centroid], embeddings)[0]
                variance = np.var(similarities)
                
                self.context_patterns[context_type] = {
                    "centroid": centroid,
                    "examples": examples,
                    "variance": variance,
                    "example_count": len(examples),
                    "base_threshold": max(0.3, 0.8 - variance)  # Adaptive threshold
                }
                
                logger.info(f"✅ Learned patterns for '{context_type}': {len(examples)} examples, variance: {variance:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to learn patterns for {context_type}: {e}")
                continue
        
        # Learn entity-type modifiers from training data structure
        self._learn_entity_type_patterns()
        
        logger.info(f"🎯 Pattern learning complete: {len(self.context_patterns)} contexts learned")
    
    def _learn_entity_type_patterns(self):
        """Learn how different entity types should modify context scoring."""
        # Analyze patterns across entity types to learn modifiers
        entity_type_contexts = {
            "person": ["personal_life", "personality", "performance", "business", "controversy"],
            "non-person": ["performance", "business", "technology", "market_sentiment"],
            "team": ["performance", "business", "roster_changes", "management"]
        }
        
        # Learn relative importance weights for each entity type
        for entity_type, relevant_contexts in entity_type_contexts.items():
            modifiers = {}
            
            for context in relevant_contexts:
                if context in self.context_patterns:
                    # Base modifier on pattern variance and example quality
                    pattern_quality = 1.0 - self.context_patterns[context]["variance"]
                    example_strength = min(1.0, self.context_patterns[context]["example_count"] / 10)
                    
                    # Entity-specific context relevance learning
                    if entity_type == "person":
                        if context in ["personal_life", "personality"]:
                            modifiers[context] = 1.2 + (pattern_quality * 0.3)  # Boost personal contexts
                        elif context == "performance":
                            modifiers[context] = 0.9 + (pattern_quality * 0.2)  # Slight reduction
                        else:
                            modifiers[context] = 1.0 + (pattern_quality * 0.1)
                    
                    elif entity_type == "non-person":
                        if context == "performance":
                            modifiers[context] = 1.3 + (pattern_quality * 0.4)  # Strong boost for assets
                        elif context == "personal_life":
                            modifiers[context] = 0.1  # Minimal for non-persons
                        else:
                            modifiers[context] = 1.0 + (pattern_quality * 0.2)
                    
                    else:  # team
                        modifiers[context] = 1.0 + (pattern_quality * 0.15)
            
            self.entity_type_modifiers[entity_type] = modifiers
        
        logger.info(f"🎯 Learned entity-type modifiers for {len(self.entity_type_modifiers)} types")
    
    def classify_context_adaptive(self, text: str, entity_name: str, entity_type: str = "person") -> Tuple[str, float]:
        """
        Adaptively classify context using learned semantic patterns.
        
        Args:
            text: Input text to classify
            entity_name: Entity being analyzed  
            entity_type: Type of entity (person, non-person, team)
            
        Returns:
            Tuple of (context_type, confidence_score)
        """
        if not self.context_patterns:
            logger.warning("⚠️ No patterns learned yet, using fallback")
            return "brief_mention", 0.3
        
        self._ensure_model_loaded()
        
        try:
            # Generate embedding for input text
            text_embedding = self.model.encode([text], convert_to_tensor=False)[0]
            
            # Calculate similarity to each learned pattern
            context_scores = {}
            
            for context_type, pattern in self.context_patterns.items():
                # Semantic similarity to learned centroid
                similarity = cosine_similarity([text_embedding], [pattern["centroid"]])[0][0]
                
                # Adaptive threshold based on pattern variance
                threshold = pattern["base_threshold"]
                
                # Apply entity-type modifier
                entity_modifier = self.entity_type_modifiers.get(entity_type, {}).get(context_type, 1.0)
                adjusted_similarity = similarity * entity_modifier
                
                # Confidence based on how much above threshold
                if adjusted_similarity > threshold:
                    confidence = min(0.95, threshold + ((adjusted_similarity - threshold) * 2))
                    context_scores[context_type] = {
                        "score": adjusted_similarity,
                        "confidence": confidence,
                        "threshold": threshold
                    }
            
            # Select best matching context
            if context_scores:
                best_context = max(context_scores.keys(), 
                                 key=lambda c: context_scores[c]["score"])
                
                best_score = context_scores[best_context]
                
                # Log the decision for learning
                self._log_classification_decision(text, entity_name, entity_type, 
                                                best_context, best_score["confidence"])
                
                return best_context, best_score["confidence"]
            
            else:
                # No patterns matched above threshold
                return "brief_mention", 0.4
                
        except Exception as e:
            logger.error(f"❌ Adaptive classification error: {e}")
            return "brief_mention", 0.3
    
    def _log_classification_decision(self, text: str, entity: str, entity_type: str, 
                                   predicted_context: str, confidence: float):
        """Log classification decisions for future learning and validation."""
        decision = {
            "timestamp": time.time(),
            "text": text[:100],  # Truncate for privacy
            "entity": entity,
            "entity_type": entity_type,
            "predicted_context": predicted_context,
            "confidence": confidence
        }
        
        # Store for potential reinforcement learning
        self.performance_history[predicted_context].append(decision)
        
        # Limit history size for memory management
        if len(self.performance_history[predicted_context]) > 1000:
            self.performance_history[predicted_context] = self.performance_history[predicted_context][-500:]
    
    def update_patterns_from_feedback(self, text: str, true_context: str, predicted_context: str, 
                                    confidence: float):
        """
        Update learned patterns based on feedback/corrections.
        
        This enables the system to improve over time with human feedback.
        """
        self._ensure_model_loaded()
        
        try:
            text_embedding = self.model.encode([text], convert_to_tensor=False)[0]
            
            if true_context != predicted_context:
                # Misclassification - update patterns
                
                # Strengthen correct context pattern
                if true_context in self.context_patterns:
                    current_centroid = self.context_patterns[true_context]["centroid"]
                    # Move centroid slightly toward this example
                    updated_centroid = current_centroid + (self.learning_rate * (text_embedding - current_centroid))
                    self.context_patterns[true_context]["centroid"] = updated_centroid
                
                # Weaken incorrect context pattern if confidence was high
                if predicted_context in self.context_patterns and confidence > 0.7:
                    current_centroid = self.context_patterns[predicted_context]["centroid"]
                    # Move centroid slightly away from this example
                    updated_centroid = current_centroid - (self.learning_rate * 0.5 * (text_embedding - current_centroid))
                    self.context_patterns[predicted_context]["centroid"] = updated_centroid
                
                logger.info(f"🎓 Updated patterns: {predicted_context} -> {true_context} (confidence: {confidence:.2f})")
                
        except Exception as e:
            logger.error(f"❌ Pattern update error: {e}")
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns for monitoring."""
        stats = {
            "total_patterns": len(self.context_patterns),
            "entity_type_modifiers": len(self.entity_type_modifiers),
            "classification_history": {
                context: len(decisions) 
                for context, decisions in self.performance_history.items()
            }
        }
        
        # Add pattern quality metrics
        if self.context_patterns:
            pattern_qualities = [p["variance"] for p in self.context_patterns.values()]
            stats["pattern_quality"] = {
                "avg_variance": float(np.mean(pattern_qualities)),
                "min_variance": float(np.min(pattern_qualities)),
                "max_variance": float(np.max(pattern_qualities))
            }
        
        return stats
    
    def save_patterns(self, filepath: Optional[str] = None):
        """Save learned patterns to disk for persistence."""
        if filepath is None:
            filepath = self.cache_path
            
        try:
            cache_data = {
                "context_patterns": self.context_patterns,
                "entity_type_modifiers": self.entity_type_modifiers,
                "performance_history": dict(self.performance_history),
                "model_name": self.model_name
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"💾 Saved adaptive patterns to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save patterns: {e}")
    
    def load_patterns(self, filepath: Optional[str] = None):
        """Load previously learned patterns from disk."""
        if filepath is None:
            filepath = self.cache_path
            
        if not Path(filepath).exists():
            logger.info("ℹ️ No cached patterns found, will learn from scratch")
            return
            
        try:
            with open(filepath, "rb") as f:
                cache_data = pickle.load(f)
            
            self.context_patterns = cache_data.get("context_patterns", {})
            self.entity_type_modifiers = cache_data.get("entity_type_modifiers", {})
            self.performance_history = defaultdict(list, cache_data.get("performance_history", {}))
            
            logger.info(f"📂 Loaded {len(self.context_patterns)} adaptive patterns from cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to load patterns: {e}")


# Global instance for use throughout the system
adaptive_scorer = AdaptiveContextScorer()