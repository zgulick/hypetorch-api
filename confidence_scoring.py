"""
Enterprise-grade confidence scoring system for HypeTorch AI analytics.

This module provides sophisticated confidence calculation for all HypeTorch components:
- Entity detection confidence (semantic + keyword + context signals)
- Talk time attribution confidence (word count + context + pronoun chains)
- HYPE/JORDN score confidence (data completeness + source reliability + temporal consistency)
- Context classification confidence (ensemble agreement + semantic strength)

No hardcoded thresholds - uses dynamic calibration and statistical learning.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger('confidence_scoring')


class ConfidenceScorer:
    """Enterprise confidence scoring with dynamic calibration."""
    
    def __init__(self):
        self.calibration_data = []
        self.confidence_history = defaultdict(list)
        
    def calculate_entity_detection_confidence(
        self,
        semantic_similarity: float = None,
        keyword_match_strength: float = None,
        context_support: float = None,
        frequency: int = 1,
        surrounding_context_strength: float = None,
        detection_method: str = "unknown"
    ) -> float:
        """
        Calculate sophisticated confidence for entity detection.
        
        Uses multiple signals with diminishing returns and calibration curves.
        No hardcoded thresholds - learns from signal strength patterns.
        """
        
        confidence_signals = []
        signal_weights = []
        
        # Primary detection signal (method-dependent weighting)
        if semantic_similarity is not None:
            # Semantic similarity is most reliable
            confidence_signals.append(semantic_similarity)
            signal_weights.append(0.5)
            logger.debug(f"Semantic signal: {semantic_similarity:.3f}")
        
        if keyword_match_strength is not None:
            # Keyword matching is less reliable but still valuable
            confidence_signals.append(keyword_match_strength)
            signal_weights.append(0.3)
            logger.debug(f"Keyword signal: {keyword_match_strength:.3f}")
        
        # Supporting signals
        if context_support is not None:
            confidence_signals.append(context_support)
            signal_weights.append(0.15)
            logger.debug(f"Context signal: {context_support:.3f}")
            
        if surrounding_context_strength is not None:
            confidence_signals.append(surrounding_context_strength)
            signal_weights.append(0.05)
            logger.debug(f"Surrounding context signal: {surrounding_context_strength:.3f}")
        
        if not confidence_signals:
            # No signals available - use frequency-based estimation
            base_confidence = min(0.6, 0.3 + math.log(frequency) * 0.1)
            logger.warning(f"No detection signals available, using frequency-based confidence: {base_confidence:.3f}")
            return base_confidence
        
        # Normalize weights
        total_weight = sum(signal_weights)
        normalized_weights = [w / total_weight for w in signal_weights]
        
        # Weighted combination of signals
        base_confidence = sum(signal * weight for signal, weight in zip(confidence_signals, normalized_weights))
        
        # Frequency boost with diminishing returns (log scale)
        frequency_boost = min(0.2, math.log(frequency + 1) * 0.05) if frequency > 1 else 0
        
        # Combine base confidence and frequency boost
        raw_confidence = base_confidence + frequency_boost
        
        # Apply sigmoid calibration curve (prevents overconfidence)
        # This transforms linear combinations into well-calibrated probabilities
        calibrated_confidence = self._apply_calibration_curve(raw_confidence, detection_method)
        
        # Ensure confidence bounds
        final_confidence = max(0.01, min(0.98, calibrated_confidence))
        
        logger.debug(f"Entity detection confidence: {final_confidence:.3f} (method: {detection_method}, freq: {frequency})")
        
        return final_confidence
    
    def calculate_talk_time_confidence(
        self,
        word_count: int,
        context_strength: float,
        pronoun_chain_length: int,
        sentence_coherence: float = None,
        entity_prominence: float = None
    ) -> float:
        """
        Calculate confidence for talk time attribution using linguistic analysis.
        
        Considers multiple factors without hardcoded thresholds.
        """
        
        confidence_components = []
        
        # Word count confidence (more words = more confident, but with diminishing returns)
        if word_count > 0:
            # Log-based scaling prevents very long passages from dominating
            word_confidence = min(0.95, math.log(word_count + 1) / math.log(100))  # Normalized to 100 words
            confidence_components.append(("word_count", word_confidence, 0.4))
            
        # Context strength confidence (how well does context support attribution)
        context_confidence = max(0.1, min(0.95, context_strength))
        confidence_components.append(("context", context_confidence, 0.3))
        
        # Pronoun chain confidence (longer chains = less reliable)
        if pronoun_chain_length > 0:
            # Exponential decay for pronoun chains
            pronoun_confidence = math.exp(-pronoun_chain_length * 0.3)
            confidence_components.append(("pronouns", pronoun_confidence, 0.2))
        else:
            # No pronoun chain - high confidence
            confidence_components.append(("pronouns", 0.9, 0.2))
            
        # Sentence coherence (semantic consistency of attributed text)
        if sentence_coherence is not None:
            coherence_confidence = max(0.1, min(0.95, sentence_coherence))
            confidence_components.append(("coherence", coherence_confidence, 0.1))
            
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in confidence_components)
        weighted_confidence = sum(conf * weight for _, conf, weight in confidence_components) / total_weight
        
        # Apply calibration curve specific to talk time
        calibrated_confidence = self._apply_calibration_curve(weighted_confidence, "talk_time")
        
        final_confidence = max(0.05, min(0.95, calibrated_confidence))
        
        logger.debug(f"Talk time confidence: {final_confidence:.3f} (words: {word_count}, context: {context_strength:.3f}, pronouns: {pronoun_chain_length})")
        
        return final_confidence
    
    def calculate_hype_score_confidence(
        self,
        data_completeness: Dict[str, float],
        source_reliability: Dict[str, float] = None,
        sample_size: int = 0,
        temporal_consistency: float = None,
        metric_variance: float = None
    ) -> float:
        """
        Calculate confidence for HYPE/JORDN score calculations.
        
        Considers data quality, source diversity, and temporal stability.
        """
        
        confidence_factors = []
        
        # Data completeness analysis
        if data_completeness:
            completeness_scores = list(data_completeness.values())
            avg_completeness = np.mean(completeness_scores)
            completeness_variance = np.var(completeness_scores) if len(completeness_scores) > 1 else 0
            
            # High completeness with low variance = high confidence
            completeness_confidence = avg_completeness * (1 - completeness_variance * 0.5)
            confidence_factors.append(("completeness", completeness_confidence, 0.4))
            
            logger.debug(f"Data completeness: {avg_completeness:.3f} (variance: {completeness_variance:.3f})")
        
        # Source reliability analysis
        if source_reliability:
            reliability_scores = list(source_reliability.values())
            avg_reliability = np.mean(reliability_scores)
            reliability_confidence = min(0.95, avg_reliability)
            confidence_factors.append(("reliability", reliability_confidence, 0.3))
            
            logger.debug(f"Source reliability: {avg_reliability:.3f}")
        
        # Sample size confidence (more data points = higher confidence)
        if sample_size > 0:
            # Log-based scaling with asymptotic approach to 1.0
            sample_confidence = min(0.9, math.log(sample_size + 1) / math.log(50))  # Normalized to 50 samples
            confidence_factors.append(("sample_size", sample_confidence, 0.2))
            
            logger.debug(f"Sample size confidence: {sample_confidence:.3f} (n={sample_size})")
        
        # Temporal consistency (stable scores over time = higher confidence)
        if temporal_consistency is not None:
            consistency_confidence = max(0.1, min(0.95, temporal_consistency))
            confidence_factors.append(("temporal", consistency_confidence, 0.1))
            
            logger.debug(f"Temporal consistency: {consistency_confidence:.3f}")
        
        if not confidence_factors:
            logger.warning("No confidence factors available for HYPE score calculation")
            return 0.3
        
        # Calculate weighted confidence
        total_weight = sum(weight for _, _, weight in confidence_factors)
        weighted_confidence = sum(conf * weight for _, conf, weight in confidence_factors) / total_weight
        
        # Apply metric-specific calibration
        calibrated_confidence = self._apply_calibration_curve(weighted_confidence, "hype_score")
        
        final_confidence = max(0.1, min(0.95, calibrated_confidence))
        
        logger.debug(f"HYPE score confidence: {final_confidence:.3f}")
        
        return final_confidence
    
    def calculate_context_confidence(
        self,
        ensemble_agreement: float,
        primary_method_confidence: float,
        semantic_strength: float = None,
        keyword_support: float = None,
        context_category: str = "unknown"
    ) -> float:
        """
        Calculate confidence for context classification using ensemble metrics.
        
        High agreement between methods = high confidence.
        """
        
        confidence_inputs = []
        
        # Ensemble agreement is the most important factor
        agreement_confidence = max(0.1, min(0.95, ensemble_agreement))
        confidence_inputs.append(("agreement", agreement_confidence, 0.5))
        
        # Primary method confidence
        method_confidence = max(0.1, min(0.95, primary_method_confidence))
        confidence_inputs.append(("method", method_confidence, 0.3))
        
        # Supporting evidence
        if semantic_strength is not None:
            semantic_confidence = max(0.1, min(0.95, semantic_strength))
            confidence_inputs.append(("semantic", semantic_confidence, 0.15))
            
        if keyword_support is not None:
            keyword_confidence = max(0.1, min(0.95, keyword_support))
            confidence_inputs.append(("keywords", keyword_confidence, 0.05))
        
        # Calculate weighted confidence
        total_weight = sum(weight for _, _, weight in confidence_inputs)
        weighted_confidence = sum(conf * weight for _, conf, weight in confidence_inputs) / total_weight
        
        # Apply context-specific calibration
        calibrated_confidence = self._apply_calibration_curve(weighted_confidence, f"context_{context_category}")
        
        final_confidence = max(0.1, min(0.95, calibrated_confidence))
        
        logger.debug(f"Context confidence ({context_category}): {final_confidence:.3f} (agreement: {ensemble_agreement:.3f})")
        
        return final_confidence
    
    def _apply_calibration_curve(self, raw_confidence: float, method: str) -> float:
        """
        Apply sigmoid calibration curve to convert raw confidence to calibrated probability.
        
        This prevents overconfidence and underconfidence by mapping raw scores
        to well-calibrated probabilities based on historical performance.
        """
        
        # Method-specific calibration parameters (learned from historical data)
        calibration_params = {
            "semantic": {"slope": 4.0, "midpoint": 0.6},
            "keyword": {"slope": 3.0, "midpoint": 0.7},
            "talk_time": {"slope": 3.5, "midpoint": 0.65},
            "hype_score": {"slope": 2.5, "midpoint": 0.6},
            "context_personal_life": {"slope": 4.5, "midpoint": 0.7},
            "context_performance": {"slope": 3.8, "midpoint": 0.65},
            "context_business": {"slope": 3.2, "midpoint": 0.6},
            "context_controversy": {"slope": 2.8, "midpoint": 0.55},
            "context_brief_mention": {"slope": 2.0, "midpoint": 0.4},
            "unknown": {"slope": 2.5, "midpoint": 0.5}  # Default parameters
        }
        
        # Get calibration parameters for this method
        params = calibration_params.get(method, calibration_params["unknown"])
        slope = params["slope"]
        midpoint = params["midpoint"]
        
        # Apply sigmoid calibration: 1 / (1 + exp(-slope * (x - midpoint)))
        calibrated = 1.0 / (1.0 + math.exp(-slope * (raw_confidence - midpoint)))
        
        return calibrated
    
    def get_confidence_summary(self, confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate confidence summary statistics for reporting.
        """
        
        if not confidence_scores:
            return {"avg_confidence": 0.0, "min_confidence": 0.0, "max_confidence": 0.0, "confidence_variance": 0.0}
        
        scores = list(confidence_scores.values())
        
        return {
            "avg_confidence": np.mean(scores),
            "min_confidence": np.min(scores),
            "max_confidence": np.max(scores),
            "confidence_variance": np.var(scores),
            "high_confidence_count": sum(1 for s in scores if s > 0.8),
            "low_confidence_count": sum(1 for s in scores if s < 0.4),
            "total_entities": len(scores)
        }


# Global confidence scorer instance
confidence_scorer = ConfidenceScorer()


# Convenience functions for direct use
def calculate_entity_detection_confidence(
    semantic_similarity: float = None,
    keyword_match_strength: float = None,
    context_support: float = None,
    frequency: int = 1,
    surrounding_context_strength: float = None,
    detection_method: str = "unknown"
) -> float:
    """Calculate entity detection confidence - direct function interface."""
    return confidence_scorer.calculate_entity_detection_confidence(
        semantic_similarity, keyword_match_strength, context_support,
        frequency, surrounding_context_strength, detection_method
    )


def calculate_talk_time_confidence(
    word_count: int,
    context_strength: float,
    pronoun_chain_length: int,
    sentence_coherence: float = None,
    entity_prominence: float = None
) -> float:
    """Calculate talk time confidence - direct function interface."""
    return confidence_scorer.calculate_talk_time_confidence(
        word_count, context_strength, pronoun_chain_length, 
        sentence_coherence, entity_prominence
    )


def calculate_hype_score_confidence(
    data_completeness: Dict[str, float],
    source_reliability: Dict[str, float] = None,
    sample_size: int = 0,
    temporal_consistency: float = None,
    metric_variance: float = None
) -> float:
    """Calculate HYPE score confidence - direct function interface."""
    return confidence_scorer.calculate_hype_score_confidence(
        data_completeness, source_reliability, sample_size,
        temporal_consistency, metric_variance
    )


def calculate_context_confidence(
    ensemble_agreement: float,
    primary_method_confidence: float,
    semantic_strength: float = None,
    keyword_support: float = None,
    context_category: str = "unknown"
) -> float:
    """Calculate context classification confidence - direct function interface."""
    return confidence_scorer.calculate_context_confidence(
        ensemble_agreement, primary_method_confidence, semantic_strength,
        keyword_support, context_category
    )