"""
Adaptive Ensemble Weight Optimizer for HypeTorch AI Methods

Replaces hardcoded AI method weights (zero_shot: 0.4, embeddings: 0.3, etc.) with:
1. Multi-armed bandit optimization based on method accuracy for different context types
2. Context-aware method selection (some methods better for performance vs personal_life)
3. Entity-type specific optimization (methods perform differently for persons vs non-persons)
4. Real-time performance feedback and continuous learning
5. Method confidence boosting for consistently accurate methods
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

logger = logging.getLogger('ensemble_optimizer')


@dataclass
class MethodPerformance:
    """Performance metrics for an AI method."""
    accuracy: float
    confidence: float
    context_type: str
    entity_type: str
    timestamp: float
    processing_time: float
    

class AdaptiveEnsembleOptimizer:
    """
    Self-optimizing AI ensemble weight system using multi-armed bandit algorithms.
    
    Continuously learns optimal weights for AI methods:
    - zero_shot: BART zero-shot classification (baseline ~0.4)
    - embeddings: Sentence transformer embeddings (baseline ~0.3) 
    - nlp_features: spaCy linguistic analysis (baseline ~0.2)
    - openai: OpenAI GPT premium method (baseline ~0.5)
    
    Adapts weights based on:
    - Context type (personal_life vs performance vs business)
    - Entity type (person, non-person, team)
    - Method accuracy over time
    - Processing speed and availability
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        
        # AI method names and baseline weights
        self.methods = ["zero_shot", "embeddings", "nlp_features", "openai"]
        self.baseline_weights = {
            "zero_shot": 0.4,      # BART zero-shot - reliable, fast
            "embeddings": 0.3,     # Sentence transformers - good semantic understanding  
            "nlp_features": 0.2,   # spaCy - grammatical analysis
            "openai": 0.5          # OpenAI - highest quality when available
        }
        
        # Bounds for method weights
        self.weight_bounds = (0.05, 0.8)  # Min/max weight for any method
        self.exploration_rate = 0.05       # Epsilon for epsilon-greedy exploration
        
        # Context and entity-specific weight storage
        self.optimized_weights = {
            # Default weights
            "default": self.baseline_weights.copy(),
            
            # Context-specific weights
            "personal_life": self.baseline_weights.copy(),
            "performance": self.baseline_weights.copy(), 
            "business": self.baseline_weights.copy(),
            "controversy": self.baseline_weights.copy(),
            "personality": self.baseline_weights.copy(),
            "brief_mention": self.baseline_weights.copy(),
            
            # Entity-specific weights
            "person": self.baseline_weights.copy(),
            "non_person": self.baseline_weights.copy(),
            "team": self.baseline_weights.copy(),
            
            # Combined context-entity weights for fine-grained optimization
            "person_personal": self.baseline_weights.copy(),
            "person_performance": self.baseline_weights.copy(),
            "non_person_performance": self.baseline_weights.copy(),
            "non_person_business": self.baseline_weights.copy()
        }
        
        # Performance tracking for each method
        self.method_performance = defaultdict(lambda: deque(maxlen=1000))
        self.accuracy_history = defaultdict(list)
        self.method_availability = defaultdict(lambda: True)  # Track if methods are working
        
        # Confidence boosting for consistently accurate methods
        self.confidence_boosts = defaultdict(lambda: 1.0)  # Multiplier for method confidence
        
        # Adaptation statistics
        self.adaptation_stats = {
            "total_optimizations": 0,
            "improvements_made": 0,
            "method_performance_gains": {},
            "confidence_boosts_applied": 0
        }
        
        self.cache_path = Path("temp/ensemble_optimizer_cache.pkl")
        
        logger.info("🤖 Initializing Adaptive Ensemble Optimizer")
    
    def get_optimal_weights(self, context_type: str = None, entity_type: str = None,
                          method_availability: Dict[str, bool] = None) -> Dict[str, float]:
        """
        Get optimal weights for AI methods in a specific scenario.
        
        Args:
            context_type: Primary context type (personal_life, performance, etc.)
            entity_type: Entity type (person, non-person, team)
            method_availability: Which methods are currently available
            
        Returns:
            Optimized weights dictionary for ensemble voting
        """
        # Determine weight context key
        context_key = self._determine_context_key(context_type, entity_type)
        
        # Get base optimized weights for this context
        base_weights = self.optimized_weights.get(context_key, self.optimized_weights["default"]).copy()
        
        # Adjust for method availability
        if method_availability:
            adapted_weights = self._adapt_for_availability(base_weights, method_availability)
        else:
            adapted_weights = base_weights
        
        # Apply confidence boosting for consistently accurate methods
        boosted_weights = self._apply_confidence_boosting(adapted_weights, context_key)
        
        # Ensure weights sum to 1.0 and respect bounds
        final_weights = self._normalize_weights(boosted_weights)
        
        logger.debug(f"🎯 Optimal ensemble weights for {context_key}: {final_weights}")
        return final_weights
    
    def _determine_context_key(self, context_type: str, entity_type: str) -> str:
        """Determine which weight context to use based on scenario."""
        
        # Try combined context-entity key first (most specific)
        if context_type and entity_type:
            combined_key = f"{entity_type}_{context_type}"
            if combined_key in self.optimized_weights:
                return combined_key
        
        # Fall back to context-specific
        if context_type and context_type in self.optimized_weights:
            return context_type
        
        # Fall back to entity-specific
        if entity_type and entity_type in self.optimized_weights:
            return entity_type
        
        # Default fallback
        return "default"
    
    def _adapt_for_availability(self, base_weights: Dict[str, float],
                              method_availability: Dict[str, bool]) -> Dict[str, float]:
        """Adapt weights based on which methods are currently available."""
        adapted_weights = base_weights.copy()
        
        # Zero out weights for unavailable methods
        unavailable_weight = 0.0
        for method in self.methods:
            if not method_availability.get(method, True):
                unavailable_weight += adapted_weights[method]
                adapted_weights[method] = 0.0
                logger.debug(f"Method {method} unavailable, weight set to 0")
        
        # Redistribute unavailable weight to available methods proportionally
        if unavailable_weight > 0:
            available_methods = [m for m in self.methods if method_availability.get(m, True)]
            if available_methods:
                available_total = sum(adapted_weights[m] for m in available_methods)
                if available_total > 0:
                    for method in available_methods:
                        proportion = adapted_weights[method] / available_total
                        adapted_weights[method] += unavailable_weight * proportion
        
        return adapted_weights
    
    def _apply_confidence_boosting(self, weights: Dict[str, float], context_key: str) -> Dict[str, float]:
        """Apply confidence boosting for consistently accurate methods."""
        boosted_weights = weights.copy()
        
        for method in self.methods:
            boost_factor = self.confidence_boosts[f"{context_key}_{method}"]
            if boost_factor > 1.0:
                # Boost weight for consistently accurate methods
                boosted_weights[method] *= boost_factor
                logger.debug(f"Applied {boost_factor:.2f}x boost to {method} for {context_key}")
        
        return boosted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights sum to 1.0 and respect bounds."""
        
        # Apply bounds
        bounded_weights = {}
        for method, weight in weights.items():
            bounded_weights[method] = max(self.weight_bounds[0], min(self.weight_bounds[1], weight))
        
        # Normalize to sum to 1.0
        total = sum(bounded_weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in bounded_weights.items()}
        else:
            # Fallback to baseline if all weights are zero
            normalized_weights = self.baseline_weights.copy()
        
        return normalized_weights
    
    def record_method_performance(self, method: str, context_type: str, entity_type: str,
                                prediction_confidence: float, actual_accuracy: float,
                                processing_time: float = None):
        """
        Record the performance of a specific AI method for learning.
        
        Args:
            method: AI method name (zero_shot, embeddings, nlp_features, openai)
            context_type: Context being classified
            entity_type: Entity type being processed  
            prediction_confidence: Confidence the method reported
            actual_accuracy: Measured accuracy (0.0 to 1.0)
            processing_time: How long the method took to process
        """
        
        performance = MethodPerformance(
            accuracy=actual_accuracy,
            confidence=prediction_confidence,
            context_type=context_type,
            entity_type=entity_type,
            timestamp=time.time(),
            processing_time=processing_time or 0.0
        )
        
        # Create performance tracking key
        context_key = self._determine_context_key(context_type, entity_type)
        method_key = f"{context_key}_{method}"
        
        # Store performance data
        self.method_performance[method_key].append(performance)
        self.accuracy_history[method_key].append(actual_accuracy)
        
        # Update method availability
        if processing_time is not None and processing_time < 30.0:  # Method responded within 30s
            self.method_availability[method] = True
        elif processing_time is None:  # No processing time = method worked
            self.method_availability[method] = True
        
        # Trigger optimization if we have enough samples
        self._maybe_optimize_weights(context_key)
        
        # Check for confidence boosting opportunity
        self._maybe_update_confidence_boost(method_key, actual_accuracy, prediction_confidence)
        
        logger.debug(f"📊 Recorded {method} performance: {context_key} -> "
                    f"accuracy={actual_accuracy:.2f}, confidence={prediction_confidence:.2f}")
    
    def _maybe_optimize_weights(self, context_key: str):
        """Optimize weights if we have enough performance data."""
        
        from config import ENSEMBLE_MIN_PREDICTIONS, ENABLE_ADAPTIVE_ENSEMBLE_WEIGHTS
        
        if not ENABLE_ADAPTIVE_ENSEMBLE_WEIGHTS:
            return
        
        # Check if we have enough data for each method in this context
        method_data_counts = {}
        for method in self.methods:
            method_key = f"{context_key}_{method}"
            method_data_counts[method] = len(self.accuracy_history.get(method_key, []))
        
        # Need minimum samples for each method to optimize
        if all(count >= ENSEMBLE_MIN_PREDICTIONS for count in method_data_counts.values()):
            self._optimize_weights_for_context(context_key)
    
    def _optimize_weights_for_context(self, context_key: str):
        """Optimize weights for a specific context based on recent performance."""
        
        # Calculate recent performance for each method
        method_performances = {}
        for method in self.methods:
            method_key = f"{context_key}_{method}"
            recent_accuracies = self.accuracy_history.get(method_key, [])[-50:]  # Last 50 predictions
            
            if recent_accuracies:
                avg_accuracy = np.mean(recent_accuracies)
                consistency = 1.0 - np.std(recent_accuracies)  # Lower std = more consistent
                method_performances[method] = avg_accuracy * 0.7 + consistency * 0.3
            else:
                method_performances[method] = 0.5  # Default performance
        
        # Current weights
        current_weights = self.optimized_weights.get(context_key, self.baseline_weights.copy())
        
        # Calculate optimal weights using softmax of performance scores
        performance_scores = np.array([method_performances[method] for method in self.methods])
        
        # Apply temperature scaling (higher temperature = more exploration)
        temperature = 2.0
        softmax_weights = np.exp(performance_scores / temperature) / np.sum(np.exp(performance_scores / temperature))
        
        # New optimal weights
        optimal_weights = {method: weight for method, weight in zip(self.methods, softmax_weights)}
        
        # Gradual adaptation using learning rate
        updated_weights = {}
        for method in self.methods:
            current = current_weights.get(method, self.baseline_weights[method])
            optimal = optimal_weights[method]
            updated_weights[method] = current + (self.learning_rate * (optimal - current))
        
        # Normalize and store
        self.optimized_weights[context_key] = self._normalize_weights(updated_weights)
        
        # Track adaptation
        self.adaptation_stats["total_optimizations"] += 1
        
        # Check if this improved overall performance
        old_avg_performance = np.mean([method_performances[m] * current_weights.get(m, 0.25) for m in self.methods])
        new_avg_performance = np.mean([method_performances[m] * updated_weights[m] for m in self.methods])
        
        if new_avg_performance > old_avg_performance:
            self.adaptation_stats["improvements_made"] += 1
            improvement = new_avg_performance - old_avg_performance
            self.adaptation_stats["method_performance_gains"][context_key] = improvement
            
            logger.info(f"🎓 Optimized ensemble weights for {context_key}: "
                       f"performance {old_avg_performance:.3f} -> {new_avg_performance:.3f} "
                       f"(+{improvement:.3f})")
        else:
            logger.debug(f"🎯 Updated ensemble weights for {context_key} (exploratory)")
    
    def _maybe_update_confidence_boost(self, method_key: str, accuracy: float, confidence: float):
        """Update confidence boosting for consistently accurate methods."""
        
        from config import CONFIDENCE_BOOST_THRESHOLD
        
        recent_accuracies = self.accuracy_history[method_key][-20:]  # Last 20 predictions
        
        if len(recent_accuracies) >= 10:
            avg_recent_accuracy = np.mean(recent_accuracies)
            
            # Boost confidence for consistently high-performing methods
            if avg_recent_accuracy > CONFIDENCE_BOOST_THRESHOLD:
                old_boost = self.confidence_boosts[method_key]
                # Gradual boosting based on consistent accuracy
                new_boost = old_boost + (self.learning_rate * 0.1 * (avg_recent_accuracy - 0.7))
                new_boost = min(1.5, max(1.0, new_boost))  # Boost between 1.0x and 1.5x
                
                if new_boost > old_boost:
                    self.confidence_boosts[method_key] = new_boost
                    self.adaptation_stats["confidence_boosts_applied"] += 1
                    
                    logger.info(f"🚀 Applied confidence boost to {method_key}: "
                               f"{old_boost:.2f}x -> {new_boost:.2f}x "
                               f"(avg accuracy: {avg_recent_accuracy:.3f})")
    
    def get_method_rankings(self, context_type: str = None, entity_type: str = None) -> Dict[str, Any]:
        """Get current method rankings for a specific scenario."""
        
        context_key = self._determine_context_key(context_type, entity_type)
        
        rankings = {}
        for method in self.methods:
            method_key = f"{context_key}_{method}"
            recent_accuracies = self.accuracy_history.get(method_key, [])
            
            if recent_accuracies:
                rankings[method] = {
                    "avg_accuracy": np.mean(recent_accuracies[-50:]),  # Recent performance
                    "consistency": 1.0 - np.std(recent_accuracies[-50:]),
                    "total_predictions": len(recent_accuracies),
                    "confidence_boost": self.confidence_boosts[method_key],
                    "current_weight": self.optimized_weights.get(context_key, {}).get(method, 0.25)
                }
            else:
                rankings[method] = {
                    "avg_accuracy": 0.5,
                    "consistency": 0.5, 
                    "total_predictions": 0,
                    "confidence_boost": 1.0,
                    "current_weight": self.baseline_weights[method]
                }
        
        # Sort by combined score
        for method, stats in rankings.items():
            combined_score = (stats["avg_accuracy"] * 0.6 + 
                            stats["consistency"] * 0.3 + 
                            (min(stats["total_predictions"], 100) / 100) * 0.1)
            stats["combined_score"] = combined_score
        
        return dict(sorted(rankings.items(), key=lambda x: x[1]["combined_score"], reverse=True))
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about ensemble optimization performance."""
        
        stats = {
            "contexts_optimized": list(self.optimized_weights.keys()),
            "total_optimizations": self.adaptation_stats["total_optimizations"],
            "improvements_made": self.adaptation_stats["improvements_made"],
            "confidence_boosts_applied": self.adaptation_stats["confidence_boosts_applied"],
            "method_availability": dict(self.method_availability),
            "total_predictions": sum(len(history) for history in self.method_performance.values())
        }
        
        # Performance improvements by context
        context_improvements = {}
        for context in self.optimized_weights.keys():
            method_accuracies = {}
            for method in self.methods:
                method_key = f"{context}_{method}"
                accuracies = self.accuracy_history.get(method_key, [])
                if len(accuracies) >= 20:
                    recent_avg = np.mean(accuracies[-20:])
                    early_avg = np.mean(accuracies[:20])
                    improvement = recent_avg - early_avg
                    method_accuracies[method] = {
                        "early_accuracy": early_avg,
                        "recent_accuracy": recent_avg,
                        "improvement": improvement
                    }
            
            if method_accuracies:
                context_improvements[context] = method_accuracies
        
        stats["performance_improvements"] = context_improvements
        
        # Current vs baseline weight comparisons
        weight_changes = {}
        for context, weights in self.optimized_weights.items():
            if context != "default":
                changes = {}
                for method in self.methods:
                    baseline = self.baseline_weights[method]
                    current = weights.get(method, baseline)
                    changes[method] = {
                        "baseline": baseline,
                        "current": current,
                        "change": current - baseline,
                        "change_percent": ((current - baseline) / baseline) * 100 if baseline > 0 else 0
                    }
                weight_changes[context] = changes
        
        stats["weight_changes"] = weight_changes
        
        return stats
    
    def save_optimization_data(self, filepath: Optional[str] = None):
        """Save ensemble optimization data to disk."""
        
        if filepath is None:
            filepath = self.cache_path
        
        try:
            optimization_data = {
                "optimized_weights": self.optimized_weights,
                "method_performance": {k: list(v) for k, v in self.method_performance.items()},
                "accuracy_history": dict(self.accuracy_history),
                "confidence_boosts": dict(self.confidence_boosts),
                "method_availability": dict(self.method_availability),
                "adaptation_stats": self.adaptation_stats,
                "baseline_weights": self.baseline_weights
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(optimization_data, f)
            
            logger.info(f"💾 Saved ensemble optimization data to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save ensemble optimization data: {e}")
    
    def load_optimization_data(self, filepath: Optional[str] = None):
        """Load previously optimized ensemble data from disk."""
        
        if filepath is None:
            filepath = self.cache_path
        
        if not Path(filepath).exists():
            logger.info("ℹ️ No cached ensemble optimization data found, starting fresh")
            return
        
        try:
            with open(filepath, "rb") as f:
                optimization_data = pickle.load(f)
            
            self.optimized_weights = optimization_data.get("optimized_weights", {})
            self.confidence_boosts = defaultdict(lambda: 1.0, optimization_data.get("confidence_boosts", {}))
            self.method_availability = defaultdict(lambda: True, optimization_data.get("method_availability", {}))
            self.adaptation_stats = optimization_data.get("adaptation_stats", {})
            
            # Restore performance history
            performance_data = optimization_data.get("method_performance", {})
            for method_key, performance_list in performance_data.items():
                self.method_performance[method_key] = deque(performance_list, maxlen=1000)
            
            self.accuracy_history = defaultdict(list, optimization_data.get("accuracy_history", {}))
            
            logger.info(f"📂 Loaded ensemble optimization data: {len(self.optimized_weights)} contexts, "
                       f"{sum(len(h) for h in self.method_performance.values())} method performances tracked")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ensemble optimization data: {e}")


# Global instance for use throughout the system
ensemble_optimizer = AdaptiveEnsembleOptimizer()