"""
Adaptive HYPE Score Weight Optimizer for HypeTorch

Replaces hardcoded HYPE component weights (30%/25%/20%/15%/10%) with:
1. Multi-armed bandit optimization based on prediction accuracy
2. Context-aware weight adaptation (different weights for different scenarios)
3. Entity-type specific optimization (athletes vs cryptocurrencies)
4. Real-time performance feedback and continuous learning
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger('hype_optimizer')


class AdaptiveHypeOptimizer:
    """
    Self-optimizing HYPE score weight system using multi-armed bandit algorithms.
    
    Continuously learns optimal weights for:
    - Talk Time (baseline ~30%)
    - Mentions (baseline ~25%)
    - Google Trends (baseline ~20%)
    - Reddit Mentions (baseline ~15%)
    - Wikipedia Views (baseline ~10%)
    
    Adapts weights based on:
    - Entity type (person, non-person, team)
    - Context distribution (personal_life heavy vs performance heavy)
    - Time period (trending vs established entities)
    - Data completeness (which sources are available)
    """
    
    def __init__(self):
        # Component names and baseline weights (starting point)
        self.components = ["talk_time", "mentions", "google_trends", "reddit_mentions", "wikipedia_views"]
        self.baseline_weights = {
            "talk_time": 0.30,
            "mentions": 0.25, 
            "google_trends": 0.20,
            "reddit_mentions": 0.15,
            "wikipedia_views": 0.10
        }
        
        # Multi-armed bandit parameters
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy
        self.learning_rate = 0.05
        self.weight_bounds = (0.05, 0.60)  # Min/max weight for any component
        
        # Context-specific weight storage
        self.optimized_weights = {
            "default": self.baseline_weights.copy(),
            "person_personal": self.baseline_weights.copy(),
            "person_performance": self.baseline_weights.copy(), 
            "non_person_performance": self.baseline_weights.copy(),
            "non_person_business": self.baseline_weights.copy(),
            "team_performance": self.baseline_weights.copy(),
            "trending": self.baseline_weights.copy(),
            "established": self.baseline_weights.copy()
        }
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.weight_performance = defaultdict(lambda: defaultdict(list))
        self.prediction_accuracy = defaultdict(list)
        
        # A/B testing framework
        self.active_experiments = {}
        self.experiment_results = defaultdict(list)
        
        self.cache_path = Path("temp/hype_optimizer_cache.pkl")
        
        logger.info("🎯 Initializing Adaptive HYPE Weight Optimizer")
    
    def get_optimal_weights(self, entity_type: str = "person", context_distribution: Dict[str, float] = None,
                          is_trending: bool = False, data_completeness: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get optimal weights for a specific scenario.
        
        Args:
            entity_type: Type of entity (person, non-person, team)
            context_distribution: Distribution of contexts (e.g., {"personal_life": 0.6, "performance": 0.4})
            is_trending: Whether this is a currently trending entity
            data_completeness: Which data sources are available (e.g., {"mentions": 1.0, "trends": 0.8})
            
        Returns:
            Optimized weights dictionary
        """
        # Determine context key for weight lookup
        context_key = self._determine_context_key(entity_type, context_distribution, is_trending)
        
        # Get base optimized weights for this context
        base_weights = self.optimized_weights.get(context_key, self.optimized_weights["default"]).copy()
        
        # Adapt weights based on data completeness
        if data_completeness:
            adapted_weights = self._adapt_for_data_completeness(base_weights, data_completeness)
        else:
            adapted_weights = base_weights
        
        # Ensure weights sum to 1.0
        adapted_weights = self._normalize_weights(adapted_weights)
        
        logger.debug(f"🎯 Optimal weights for {context_key}: {adapted_weights}")
        return adapted_weights
    
    def _determine_context_key(self, entity_type: str, context_distribution: Dict[str, float], 
                              is_trending: bool) -> str:
        """Determine which weight context to use based on scenario."""
        
        if is_trending:
            return "trending"
        
        if not context_distribution:
            return "default"
        
        # Find dominant context
        dominant_context = max(context_distribution.keys(), key=context_distribution.get)
        dominant_weight = context_distribution[dominant_context]
        
        # Only use specific contexts if they're dominant (>40%)
        if dominant_weight < 0.4:
            return "default"
        
        # Map to weight context keys
        if entity_type == "person":
            if dominant_context in ["personal_life", "personality"]:
                return "person_personal"
            elif dominant_context == "performance":
                return "person_performance"
        elif entity_type == "non-person":
            if dominant_context == "performance":
                return "non_person_performance"
            elif dominant_context == "business":
                return "non_person_business"
        elif entity_type == "team":
            return "team_performance"
        
        return "default"
    
    def _adapt_for_data_completeness(self, base_weights: Dict[str, float], 
                                   data_completeness: Dict[str, float]) -> Dict[str, float]:
        """Adapt weights based on which data sources are actually available."""
        adapted_weights = base_weights.copy()
        
        # Reduce weights for components with low data completeness
        for component in self.components:
            completeness = data_completeness.get(component, 1.0)
            
            # Reduce weight proportionally to missing data
            if completeness < 0.5:  # Less than 50% data available
                reduction_factor = 0.3 + (completeness * 0.7)  # 30-100% of original weight
                adapted_weights[component] *= reduction_factor
        
        return adapted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights sum to 1.0 and respect bounds."""
        
        # Apply bounds
        bounded_weights = {}
        for component, weight in weights.items():
            bounded_weights[component] = max(self.weight_bounds[0], min(self.weight_bounds[1], weight))
        
        # Normalize to sum to 1.0
        total = sum(bounded_weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in bounded_weights.items()}
        else:
            # Fallback to baseline if all weights are zero
            normalized_weights = self.baseline_weights.copy()
        
        return normalized_weights
    
    def record_prediction_result(self, entity_name: str, context_key: str, weights_used: Dict[str, float],
                               predicted_hype: float, actual_performance: float = None,
                               engagement_metrics: Dict[str, float] = None):
        """
        Record the result of a HYPE score prediction for learning.
        
        Args:
            entity_name: Entity that was analyzed
            context_key: Context used for weight selection
            weights_used: The weights that were used
            predicted_hype: The HYPE score that was predicted
            actual_performance: Actual performance metric if available (engagement, views, etc.)
            engagement_metrics: Additional metrics for validation
        """
        
        timestamp = time.time()
        
        # Record prediction
        prediction_record = {
            "timestamp": timestamp,
            "entity": entity_name,
            "context_key": context_key,
            "weights_used": weights_used.copy(),
            "predicted_hype": predicted_hype,
            "actual_performance": actual_performance,
            "engagement_metrics": engagement_metrics or {}
        }
        
        self.performance_history[context_key].append(prediction_record)
        
        # If we have actual performance, calculate accuracy and update weights
        if actual_performance is not None:
            error = abs(predicted_hype - actual_performance)
            accuracy = max(0, 1 - (error / max(actual_performance, predicted_hype, 1)))
            
            self.prediction_accuracy[context_key].append(accuracy)
            
            # Update weights based on performance
            self._update_weights_from_feedback(context_key, weights_used, accuracy)
            
            logger.debug(f"📊 Recorded prediction: {entity_name} -> {predicted_hype:.1f} "
                        f"(actual: {actual_performance:.1f}, accuracy: {accuracy:.2f})")
    
    def _update_weights_from_feedback(self, context_key: str, weights_used: Dict[str, float], 
                                    accuracy: float):
        """Update weight optimization based on prediction accuracy."""
        
        if context_key not in self.optimized_weights:
            self.optimized_weights[context_key] = self.baseline_weights.copy()
        
        current_weights = self.optimized_weights[context_key]
        
        # Gradient-based update toward better performance
        if accuracy > 0.7:  # Good prediction - reinforce these weights
            for component in self.components:
                # Move current weights slightly toward weights that worked well
                difference = weights_used[component] - current_weights[component]
                update = self.learning_rate * difference * accuracy
                current_weights[component] += update
        
        elif accuracy < 0.3:  # Poor prediction - move away from these weights  
            for component in self.components:
                # Move current weights away from weights that worked poorly
                difference = weights_used[component] - current_weights[component]
                update = self.learning_rate * difference * (1 - accuracy) * -0.5
                current_weights[component] += update
        
        # Normalize updated weights
        self.optimized_weights[context_key] = self._normalize_weights(current_weights)
        
        logger.debug(f"🎓 Updated weights for {context_key} based on accuracy {accuracy:.2f}")
    
    def run_ab_test(self, experiment_name: str, variant_weights: Dict[str, Dict[str, float]], 
                   duration_hours: int = 24) -> str:
        """
        Run A/B test comparing different weight configurations.
        
        Args:
            experiment_name: Name for this experiment
            variant_weights: Dict of variant_name -> weights_dict
            duration_hours: How long to run the experiment
            
        Returns:
            Experiment ID for tracking
        """
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        self.active_experiments[experiment_id] = {
            "name": experiment_name,
            "variants": variant_weights,
            "start_time": time.time(),
            "end_time": time.time() + (duration_hours * 3600),
            "results": {variant: [] for variant in variant_weights.keys()}
        }
        
        logger.info(f"🧪 Started A/B test '{experiment_name}' with {len(variant_weights)} variants "
                   f"for {duration_hours} hours")
        
        return experiment_id
    
    def get_ab_test_variant(self, experiment_id: str) -> Optional[Tuple[str, Dict[str, float]]]:
        """Get a variant assignment for an active A/B test."""
        
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        # Check if experiment is still active
        if time.time() > experiment["end_time"]:
            return None
        
        # Random variant assignment
        variant_names = list(experiment["variants"].keys())
        selected_variant = np.random.choice(variant_names)
        weights = experiment["variants"][selected_variant]
        
        return selected_variant, weights
    
    def record_ab_test_result(self, experiment_id: str, variant: str, accuracy: float):
        """Record a result for an A/B test."""
        
        if experiment_id in self.active_experiments:
            self.active_experiments[experiment_id]["results"][variant].append(accuracy)
    
    def analyze_ab_test_results(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze completed A/B test results."""
        
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        results = {}
        
        for variant, accuracies in experiment["results"].items():
            if accuracies:
                results[variant] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "sample_size": len(accuracies),
                    "confidence_interval": stats.t.interval(0.95, len(accuracies)-1, 
                                                          loc=np.mean(accuracies), 
                                                          scale=stats.sem(accuracies)) if len(accuracies) > 1 else (0, 0)
                }
        
        # Statistical significance testing
        if len(results) >= 2:
            variant_names = list(results.keys())
            accuracies_a = experiment["results"][variant_names[0]]
            accuracies_b = experiment["results"][variant_names[1]]
            
            if len(accuracies_a) > 1 and len(accuracies_b) > 1:
                t_stat, p_value = stats.ttest_ind(accuracies_a, accuracies_b)
                results["statistical_test"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        return {
            "experiment_name": experiment["name"],
            "results": results,
            "duration_hours": (experiment["end_time"] - experiment["start_time"]) / 3600,
            "total_samples": sum(len(acc) for acc in experiment["results"].values())
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about weight optimization performance."""
        
        stats = {
            "contexts_optimized": list(self.optimized_weights.keys()),
            "active_experiments": len(self.active_experiments),
            "total_predictions": sum(len(history) for history in self.performance_history.values())
        }
        
        # Calculate performance improvements
        context_improvements = {}
        for context, accuracies in self.prediction_accuracy.items():
            if len(accuracies) >= 10:  # Need sufficient data
                # Compare recent vs early performance
                recent = accuracies[-100:]  # Last 100 predictions
                early = accuracies[:100]    # First 100 predictions
                
                if len(early) >= 10:
                    improvement = np.mean(recent) - np.mean(early)
                    context_improvements[context] = {
                        "early_accuracy": np.mean(early),
                        "recent_accuracy": np.mean(recent), 
                        "improvement": improvement,
                        "total_predictions": len(accuracies)
                    }
        
        stats["performance_improvements"] = context_improvements
        
        # Current vs baseline weight comparisons
        weight_changes = {}
        for context, weights in self.optimized_weights.items():
            if context != "default":
                changes = {}
                for component in self.components:
                    baseline = self.baseline_weights[component]
                    current = weights[component]
                    changes[component] = {
                        "baseline": baseline,
                        "current": current,
                        "change": current - baseline,
                        "change_percent": ((current - baseline) / baseline) * 100
                    }
                weight_changes[context] = changes
        
        stats["weight_changes"] = weight_changes
        
        return stats
    
    def save_optimization_data(self, filepath: Optional[str] = None):
        """Save optimization data to disk for persistence."""
        
        if filepath is None:
            filepath = self.cache_path
        
        try:
            optimization_data = {
                "optimized_weights": self.optimized_weights,
                "performance_history": {k: list(v) for k, v in self.performance_history.items()},
                "prediction_accuracy": dict(self.prediction_accuracy),
                "active_experiments": self.active_experiments,
                "baseline_weights": self.baseline_weights
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(optimization_data, f)
            
            logger.info(f"💾 Saved optimization data to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save optimization data: {e}")
    
    def load_optimization_data(self, filepath: Optional[str] = None):
        """Load previously optimized data from disk."""
        
        if filepath is None:
            filepath = self.cache_path
        
        if not Path(filepath).exists():
            logger.info("ℹ️ No cached optimization data found, starting fresh")
            return
        
        try:
            with open(filepath, "rb") as f:
                optimization_data = pickle.load(f)
            
            self.optimized_weights = optimization_data.get("optimized_weights", {})
            
            # Restore performance history as deques
            history_data = optimization_data.get("performance_history", {})
            for context, history_list in history_data.items():
                self.performance_history[context] = deque(history_list, maxlen=1000)
            
            self.prediction_accuracy = defaultdict(list, optimization_data.get("prediction_accuracy", {}))
            self.active_experiments = optimization_data.get("active_experiments", {})
            
            # Clean up expired experiments
            current_time = time.time()
            expired_experiments = [exp_id for exp_id, exp in self.active_experiments.items() 
                                 if current_time > exp["end_time"]]
            for exp_id in expired_experiments:
                del self.active_experiments[exp_id]
            
            logger.info(f"📂 Loaded optimization data: {len(self.optimized_weights)} contexts, "
                       f"{sum(len(h) for h in self.performance_history.values())} predictions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load optimization data: {e}")


# Global instance for use throughout the system  
hype_optimizer = AdaptiveHypeOptimizer()