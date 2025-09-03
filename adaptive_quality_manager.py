"""
Adaptive Quality Manager for HypeTorch

Replaces hardcoded quality thresholds with learning system that adapts based on 
historical accuracy and prediction performance. Different thresholds for different 
entity types and contexts.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from pathlib import Path
import pickle
import time
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger('adaptive_quality')


@dataclass
class QualityMetrics:
    """Quality metrics for tracking threshold performance."""
    threshold_used: float
    prediction_confidence: float
    actual_accuracy: Optional[float]
    entity_type: str
    context_type: str
    timestamp: float
    

class AdaptiveQualityManager:
    """
    Learns optimal quality thresholds based on historical accuracy data.
    
    Replaces hardcoded thresholds with adaptive thresholds that:
    1. Learn from prediction accuracy over time
    2. Adapt to different entity types (person vs non-person vs team)
    3. Adjust for different contexts (personal_life vs performance vs business)
    4. Balance precision vs recall based on business needs
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        
        # Context-specific thresholds (start with reasonable defaults)
        self.adaptive_thresholds = {
            # Entity detection thresholds by type
            "entity_confidence": {
                "person": 0.4,
                "non-person": 0.45,  # Slightly higher for crypto/assets
                "team": 0.35,        # Slightly lower for team mentions
                "default": 0.4
            },
            # Context classification thresholds by context
            "context_confidence": {
                "personal_life": 0.3,    # Lower threshold - personal content valuable
                "performance": 0.4,      # Standard threshold
                "business": 0.45,        # Higher threshold - business impact important
                "controversy": 0.5,      # High threshold - controversy needs confidence
                "personality": 0.35,     # Medium threshold
                "brief_mention": 0.2,    # Low threshold - brief mentions are low risk
                "default": 0.3
            },
            # Talk time thresholds by entity type
            "talk_time_confidence": {
                "person": 0.2,
                "non-person": 0.3,      # Higher for non-person entities
                "team": 0.25,
                "default": 0.2
            },
            # HYPE score thresholds by context dominance
            "hype_score_confidence": {
                "personal_heavy": 0.4,   # Personal content = lower business risk
                "performance_heavy": 0.5,  # Performance content = standard
                "business_heavy": 0.6,   # Business content = higher stakes
                "controversy_heavy": 0.7,  # Controversial content = highest stakes
                "default": 0.5
            }
        }
        
        # Performance tracking for threshold adaptation
        self.threshold_performance = defaultdict(lambda: deque(maxlen=1000))
        self.accuracy_history = defaultdict(list)
        
        # Threshold adaptation statistics
        self.adaptation_stats = {
            "total_adaptations": 0,
            "improvements_made": 0,
            "performance_gains": {}
        }
        
        self.cache_path = Path("temp/adaptive_quality_cache.pkl")
        
        logger.info("📊 Initializing Adaptive Quality Manager")
    
    def get_adaptive_threshold(self, threshold_type: str, entity_type: str = None, 
                             context_type: str = None, context_distribution: Dict[str, float] = None) -> float:
        """
        Get optimal threshold for a specific scenario.
        
        Args:
            threshold_type: Type of threshold (entity_confidence, context_confidence, etc.)
            entity_type: Type of entity (person, non-person, team)
            context_type: Primary context type
            context_distribution: Distribution of contexts for HYPE threshold
            
        Returns:
            Optimal threshold for this scenario
        """
        
        if threshold_type not in self.adaptive_thresholds:
            logger.warning(f"Unknown threshold type: {threshold_type}")
            return 0.5  # Default fallback
        
        threshold_config = self.adaptive_thresholds[threshold_type]
        
        # For HYPE thresholds, determine context dominance
        if threshold_type == "hype_score_confidence" and context_distribution:
            context_key = self._determine_context_dominance(context_distribution)
            return threshold_config.get(context_key, threshold_config["default"])
        
        # For other thresholds, use entity type or context type
        lookup_key = entity_type or context_type or "default"
        return threshold_config.get(lookup_key, threshold_config["default"])
    
    def _determine_context_dominance(self, context_distribution: Dict[str, float]) -> str:
        """Determine which context dominates for threshold selection."""
        
        if not context_distribution:
            return "default"
        
        # Find dominant context
        dominant_context = max(context_distribution.keys(), key=context_distribution.get)
        dominant_weight = context_distribution[dominant_context]
        
        # Only consider dominant if >40% of content
        if dominant_weight < 0.4:
            return "default"
        
        # Map contexts to threshold categories
        context_mapping = {
            "personal_life": "personal_heavy",
            "personality": "personal_heavy", 
            "performance": "performance_heavy",
            "business": "business_heavy",
            "controversy": "controversy_heavy"
        }
        
        return context_mapping.get(dominant_context, "default")
    
    def record_prediction_result(self, threshold_type: str, threshold_used: float, 
                               prediction_confidence: float, entity_type: str,
                               context_type: str = None, actual_accuracy: float = None):
        """
        Record the result of using a specific threshold.
        
        Args:
            threshold_type: Type of threshold used
            threshold_used: The threshold value that was used
            prediction_confidence: Confidence of the prediction made
            entity_type: Type of entity
            context_type: Context type (if applicable)
            actual_accuracy: Measured accuracy (if available)
        """
        
        metrics = QualityMetrics(
            threshold_used=threshold_used,
            prediction_confidence=prediction_confidence,
            actual_accuracy=actual_accuracy,
            entity_type=entity_type,
            context_type=context_type or "unknown",
            timestamp=time.time()
        )
        
        # Create key for this threshold scenario
        scenario_key = f"{threshold_type}_{entity_type}_{context_type or 'default'}"
        
        # Store metrics
        self.threshold_performance[scenario_key].append(metrics)
        
        # If we have actual accuracy, track for learning
        if actual_accuracy is not None:
            self.accuracy_history[scenario_key].append({
                "threshold": threshold_used,
                "accuracy": actual_accuracy,
                "confidence": prediction_confidence,
                "timestamp": time.time()
            })
            
            # Trigger adaptation if we have enough samples
            self._maybe_adapt_threshold(threshold_type, entity_type, context_type)
        
        logger.debug(f"📊 Recorded threshold performance: {scenario_key} -> "
                    f"threshold={threshold_used:.2f}, confidence={prediction_confidence:.2f}, "
                    f"accuracy={actual_accuracy}")
    
    def _maybe_adapt_threshold(self, threshold_type: str, entity_type: str, context_type: str = None):
        """Adapt threshold if we have enough data and detect improvement opportunity."""
        
        from config import QUALITY_ADAPTATION_MIN_SAMPLES, ENABLE_ADAPTIVE_QUALITY_THRESHOLDS
        
        if not ENABLE_ADAPTIVE_QUALITY_THRESHOLDS:
            return
        
        scenario_key = f"{threshold_type}_{entity_type}_{context_type or 'default'}"
        history = self.accuracy_history.get(scenario_key, [])
        
        if len(history) < QUALITY_ADAPTATION_MIN_SAMPLES:
            return  # Need more samples
        
        # Analyze recent performance vs current threshold
        recent_data = history[-QUALITY_ADAPTATION_MIN_SAMPLES:]
        current_threshold = self.get_adaptive_threshold(threshold_type, entity_type, context_type)
        
        # Calculate performance at current threshold
        current_performance = self._calculate_threshold_performance(recent_data, current_threshold)
        
        # Try slightly higher and lower thresholds
        higher_threshold = min(0.95, current_threshold + 0.05)
        lower_threshold = max(0.05, current_threshold - 0.05)
        
        higher_performance = self._calculate_threshold_performance(recent_data, higher_threshold)
        lower_performance = self._calculate_threshold_performance(recent_data, lower_threshold)
        
        # Choose best threshold
        performances = [
            (current_threshold, current_performance),
            (higher_threshold, higher_performance),
            (lower_threshold, lower_performance)
        ]
        
        best_threshold, best_performance = max(performances, key=lambda x: x[1]["f1_score"])
        
        # Only adapt if improvement is significant (>2% F1 score improvement)
        if best_performance["f1_score"] > current_performance["f1_score"] + 0.02:
            
            # Update threshold
            lookup_key = entity_type or context_type or "default"
            if threshold_type in self.adaptive_thresholds:
                old_threshold = self.adaptive_thresholds[threshold_type][lookup_key]
                
                # Gradual adaptation using learning rate
                new_threshold = old_threshold + (self.learning_rate * (best_threshold - old_threshold))
                self.adaptive_thresholds[threshold_type][lookup_key] = new_threshold
                
                # Track adaptation
                self.adaptation_stats["total_adaptations"] += 1
                if best_performance["f1_score"] > current_performance["f1_score"]:
                    self.adaptation_stats["improvements_made"] += 1
                
                improvement = best_performance["f1_score"] - current_performance["f1_score"]
                self.adaptation_stats["performance_gains"][scenario_key] = improvement
                
                logger.info(f"🎓 Adapted threshold for {scenario_key}: {old_threshold:.3f} -> {new_threshold:.3f} "
                           f"(F1: {current_performance['f1_score']:.3f} -> {best_performance['f1_score']:.3f})")
    
    def _calculate_threshold_performance(self, data: List[Dict], threshold: float) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score for a given threshold."""
        
        if not data:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Simulate predictions at this threshold
        true_positives = 0
        false_positives = 0  
        false_negatives = 0
        
        for item in data:
            predicted_positive = item["confidence"] >= threshold
            actual_positive = item["accuracy"] > 0.7  # Consider >70% accuracy as "correct"
            
            if predicted_positive and actual_positive:
                true_positives += 1
            elif predicted_positive and not actual_positive:
                false_positives += 1
            elif not predicted_positive and actual_positive:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1_score": f1_score
        }
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get statistics about quality threshold adaptation."""
        
        stats = {
            "adaptive_thresholds": self.adaptive_thresholds,
            "adaptation_stats": self.adaptation_stats,
            "scenario_count": len(self.threshold_performance),
            "total_predictions": sum(len(perf) for perf in self.threshold_performance.values())
        }
        
        # Calculate average performance improvements
        if self.adaptation_stats["performance_gains"]:
            avg_improvement = np.mean(list(self.adaptation_stats["performance_gains"].values()))
            stats["average_performance_gain"] = avg_improvement
        
        return stats
    
    def filter_entities_by_adaptive_quality(self, entity_data: Dict[str, Any], 
                                          entity_type: str = "person") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Filter entities using adaptive thresholds instead of hardcoded ones.
        
        Args:
            entity_data: Dictionary of entity data with confidence scores
            entity_type: Type of entities being filtered
            
        Returns:
            Tuple of (passed_entities, filtered_entities)
        """
        
        # Get adaptive threshold for this entity type
        threshold = self.get_adaptive_threshold("entity_confidence", entity_type)
        
        passed = {}
        filtered = {}
        
        for entity, data in entity_data.items():
            if isinstance(data, dict):
                confidence = data.get("confidence", 0.0)
                
                if confidence >= threshold:
                    passed[entity] = data
                    
                    # Record successful filtering decision
                    self.record_prediction_result(
                        threshold_type="entity_confidence",
                        threshold_used=threshold,
                        prediction_confidence=confidence,
                        entity_type=entity_type,
                        actual_accuracy=None  # Would need external validation
                    )
                else:
                    filtered[entity] = data
            else:
                # Handle legacy format
                if data >= threshold:
                    passed[entity] = data
                else:
                    filtered[entity] = data
        
        logger.debug(f"📊 Adaptive filtering: {len(passed)}/{len(entity_data)} entities passed "
                    f"(threshold: {threshold:.2f})")
        
        return passed, filtered
    
    def save_adaptation_data(self, filepath: Optional[str] = None):
        """Save learned threshold adaptations to disk."""
        
        if filepath is None:
            filepath = self.cache_path
        
        try:
            adaptation_data = {
                "adaptive_thresholds": self.adaptive_thresholds,
                "accuracy_history": dict(self.accuracy_history),
                "adaptation_stats": self.adaptation_stats,
                "learning_rate": self.learning_rate
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(adaptation_data, f)
            
            logger.info(f"💾 Saved adaptive quality thresholds to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save adaptation data: {e}")
    
    def load_adaptation_data(self, filepath: Optional[str] = None):
        """Load previously learned threshold adaptations."""
        
        if filepath is None:
            filepath = self.cache_path
        
        if not Path(filepath).exists():
            logger.info("ℹ️ No cached adaptation data found, starting with defaults")
            return
        
        try:
            with open(filepath, "rb") as f:
                adaptation_data = pickle.load(f)
            
            self.adaptive_thresholds = adaptation_data.get("adaptive_thresholds", {})
            self.accuracy_history = defaultdict(list, adaptation_data.get("accuracy_history", {}))
            self.adaptation_stats = adaptation_data.get("adaptation_stats", {})
            self.learning_rate = adaptation_data.get("learning_rate", self.learning_rate)
            
            logger.info(f"📂 Loaded adaptive quality data: "
                       f"{self.adaptation_stats.get('total_adaptations', 0)} adaptations made, "
                       f"{len(self.accuracy_history)} scenarios tracked")
            
        except Exception as e:
            logger.error(f"❌ Failed to load adaptation data: {e}")


# Global instance for use throughout the system
adaptive_quality_manager = AdaptiveQualityManager()