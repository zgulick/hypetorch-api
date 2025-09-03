"""
Quality filtering and validation system for HypeTorch analytics.
Ensures only high-confidence, reliable data reaches licensing partners.

Integrates with our existing enterprise AI system (context_classifier.py) 
rather than duplicating functionality.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from adaptive_quality_manager import adaptive_quality_manager

logger = logging.getLogger('quality_gates')


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for the entire pipeline."""
    
    # Entity Detection Quality
    min_entity_confidence: float = 0.4
    min_mentions_for_analysis: int = 2
    max_entity_results: int = 100
    
    # Talk Time Quality
    min_talk_time_confidence: float = 0.3
    min_talk_time_seconds: float = 5.0
    max_talk_time_minutes: float = 120.0
    
    # HYPE Score Quality
    min_hype_confidence: float = 0.5
    min_data_sources_for_hype: int = 2
    max_score_change_percent: float = 300.0
    
    # Context Classification Quality (integrates with our AI system)
    min_context_confidence: float = 0.3
    require_context_for_weighting: bool = True
    
    # Transcript Quality
    min_transcript_words: int = 100
    max_repetition_ratio: float = 0.40
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "entity_detection": {
                "min_confidence": self.min_entity_confidence,
                "min_mentions": self.min_mentions_for_analysis,
                "max_results": self.max_entity_results
            },
            "talk_time": {
                "min_confidence": self.min_talk_time_confidence,
                "min_seconds": self.min_talk_time_seconds,
                "max_minutes": self.max_talk_time_minutes
            },
            "hype_score": {
                "min_confidence": self.min_hype_confidence,
                "min_data_sources": self.min_data_sources_for_hype,
                "max_change_percent": self.max_score_change_percent
            },
            "context": {
                "min_confidence": self.min_context_confidence,
                "require_for_weighting": self.require_context_for_weighting
            },
            "transcript": {
                "min_words": self.min_transcript_words,
                "max_repetition_ratio": self.max_repetition_ratio
            }
        }


# Global instance
QUALITY_CONFIG = QualityThresholds()


class QualityGateManager:
    """
    Manages quality filtering throughout the pipeline.
    Integrates with our existing AI-enhanced data structures.
    """
    
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QUALITY_CONFIG
        self.filtered_entities = []
        self.quality_metrics = defaultdict(list)
        self.processing_stats = {
            "entities_processed": 0,
            "entities_passed": 0,
            "talk_time_processed": 0,
            "talk_time_passed": 0,
            "hype_scores_processed": 0,
            "hype_scores_validated": 0
        }
        
    def filter_entities_by_confidence(
        self, 
        entity_data: Dict[str, Any], 
        threshold: float = None,
        entity_type: str = "person"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Filter entities based on adaptive confidence threshold.
        Uses adaptive quality manager for intelligent thresholds.
        Returns (passed_entities, filtered_entities)
        """
        # Use adaptive quality manager instead of hardcoded threshold
        if threshold is None:
            threshold = adaptive_quality_manager.get_adaptive_threshold(
                "entity_confidence", entity_type=entity_type
            )
        
        passed = {}
        filtered = {}
        
        for entity, data in entity_data.items():
            self.processing_stats["entities_processed"] += 1
            
            # Handle both legacy and AI-enhanced formats
            if isinstance(data, dict):
                confidence = data.get('confidence', 0.5)
                quality_passed = data.get('quality_passed', True)
                ai_enhanced = data.get('ai_enhanced', False)
                
                # Special handling for AI-enhanced entities
                if ai_enhanced:
                    # Use weighted_count confidence if available
                    raw_count = data.get('raw_count', 0)
                    weighted_count = data.get('weighted_count', raw_count)
                    
                    # Boost confidence for well-weighted entities
                    if weighted_count > raw_count * 1.5:  # High-value context detected
                        confidence = min(0.95, confidence * 1.1)
                    
            else:
                # Legacy format - assume medium confidence
                confidence = 0.5
                quality_passed = True
                data = {'count': data, 'confidence': confidence}
            
            # Apply quality filters
            filter_reasons = []
            
            if confidence < threshold:
                filter_reasons.append(f"Low confidence: {confidence:.3f}")
            
            if not quality_passed:
                filter_reasons.append("Failed quality checks")
            
            # Check minimum mentions if count available
            count = data.get('count', data.get('raw_count', data.get('weighted_count', 0)))
            if count < self.thresholds.min_mentions_for_analysis:
                filter_reasons.append(f"Too few mentions: {count}")
            
            if not filter_reasons:
                passed[entity] = data
                self.processing_stats["entities_passed"] += 1
                self.quality_metrics['entities_passed'].append(entity)
            else:
                filtered[entity] = data
                self.filtered_entities.append({
                    'entity': entity,
                    'confidence': confidence,
                    'reason': '; '.join(filter_reasons),
                    'data_type': 'AI-enhanced' if isinstance(data, dict) and data.get('ai_enhanced') else 'legacy'
                })
                
        logger.info(f"✅ Entity filtering: {len(passed)}/{len(entity_data)} passed quality gates (threshold: {threshold:.2f})")
        
        return passed, filtered
    
    def filter_talk_time_by_quality(
        self,
        talk_time_data: Dict[str, Any],
        threshold: float = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Filter talk time data based on quality criteria.
        Returns (passed_data, filtered_data)
        """
        threshold = threshold or self.thresholds.min_talk_time_confidence
        min_seconds = self.thresholds.min_talk_time_seconds
        max_minutes = self.thresholds.max_talk_time_minutes
        
        passed = {}
        filtered = {}
        
        for entity, data in talk_time_data.items():
            self.processing_stats["talk_time_processed"] += 1
            
            if isinstance(data, dict):
                confidence = data.get('confidence', 0.5)
                minutes = data.get('minutes', 0)
            else:
                # Legacy format
                confidence = 0.5
                minutes = data
                data = {'minutes': minutes, 'confidence': confidence}
            
            seconds = minutes * 60
            
            # Apply multiple quality checks
            quality_issues = []
            
            if confidence < threshold:
                quality_issues.append(f"Low confidence: {confidence:.3f}")
            
            if seconds < min_seconds:
                quality_issues.append(f"Too brief: {seconds:.1f}s")
            
            if minutes > max_minutes:
                quality_issues.append(f"Unrealistic duration: {minutes:.1f}m")
            
            # Check for negative values
            if minutes < 0:
                quality_issues.append(f"Negative duration: {minutes:.2f}m")
            
            if not quality_issues:
                passed[entity] = data
                self.processing_stats["talk_time_passed"] += 1
                self.quality_metrics['talk_time_passed'].append(entity)
            else:
                filtered[entity] = data
                self.filtered_entities.append({
                    'entity': entity,
                    'confidence': confidence,
                    'reason': ', '.join(quality_issues),
                    'data_type': 'talk_time'
                })
        
        logger.info(f"⏱️ Talk time filtering: {len(passed)}/{len(talk_time_data)} passed quality gates")
        
        return passed, filtered
    
    def validate_hype_score_quality(
        self,
        hype_scores: Dict[str, float],
        confidence_scores: Dict[str, float],
        data_sources: Dict[str, Any],
        entity_context_distribution: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Validate HYPE score quality and add quality indicators.
        Integrates with our AI-enhanced confidence system.
        """
        validated_scores = {}
        
        for entity, score in hype_scores.items():
            self.processing_stats["hype_scores_processed"] += 1
            
            confidence = confidence_scores.get(entity, 0.5)
            
            # Get adaptive threshold based on entity's context distribution
            context_distribution = entity_context_distribution.get(entity) if entity_context_distribution else None
            adaptive_threshold = adaptive_quality_manager.get_adaptive_threshold(
                "hype_score_confidence", context_distribution=context_distribution
            )
            
            # Count available data sources
            source_count = 0
            for source_name, source_data in data_sources.items():
                if entity in source_data and source_data[entity]:
                    source_count += 1
            
            # Check for AI enhancement boost
            ai_enhanced = False
            if "Mentions" in data_sources:
                mention_data = data_sources["Mentions"].get(entity, {})
                if isinstance(mention_data, dict):
                    ai_enhanced = mention_data.get("ai_enhanced", False)
                    if ai_enhanced:
                        # Boost confidence for AI-enhanced mentions
                        confidence = min(0.95, confidence * 1.05)
            
            # Determine quality level
            if confidence >= 0.8 and source_count >= 4:
                quality_level = "high"
            elif confidence >= 0.6 and source_count >= 3:
                quality_level = "medium" 
            elif confidence >= 0.4 and source_count >= 2:
                quality_level = "acceptable"
            else:
                quality_level = "low"
            
            # Additional quality checks
            quality_flags = []
            
            if score < 0:
                quality_flags.append("negative_score")
            if score > 1000:  # Unrealistically high
                quality_flags.append("extreme_score")
            if source_count < self.thresholds.min_data_sources_for_hype:
                quality_flags.append("insufficient_data")
            
            validated_scores[entity] = {
                "score": score,
                "confidence": confidence,
                "quality": quality_level,
                "data_sources": source_count,
                "ai_enhanced": ai_enhanced,
                "quality_flags": quality_flags,
                "validated": True
            }
            
            if quality_level in ["high", "medium"]:
                self.processing_stats["hype_scores_validated"] += 1
                self.quality_metrics['high_quality_scores'].append(entity)
        
        high_quality_count = len([v for v in validated_scores.values() if v['quality'] in ['high', 'medium']])
        logger.info(f"🏆 HYPE validation: {high_quality_count}/{len(hype_scores)} high/medium quality")
        
        return validated_scores
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for the processing run.
        Includes AI-enhancement statistics and business metrics.
        """
        
        # Calculate pass rates
        entity_pass_rate = (
            self.processing_stats["entities_passed"] / max(self.processing_stats["entities_processed"], 1)
        )
        talk_time_pass_rate = (
            self.processing_stats["talk_time_passed"] / max(self.processing_stats["talk_time_processed"], 1)
        )
        hype_validation_rate = (
            self.processing_stats["hype_scores_validated"] / max(self.processing_stats["hype_scores_processed"], 1)
        )
        
        # Aggregate confidence scores
        all_confidences = []
        for item in self.filtered_entities:
            all_confidences.append(item['confidence'])
        for entity in self.quality_metrics['entities_passed']:
            all_confidences.append(0.7)  # Assume passed entities have decent confidence
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        # Check for AI enhancement usage
        ai_enhanced_count = len([item for item in self.filtered_entities if item.get('data_type') == 'AI-enhanced'])
        ai_enhancement_rate = ai_enhanced_count / max(len(self.filtered_entities), 1) if self.filtered_entities else 0
        
        # Calculate off-court percentage if context classifier available
        off_court_percentage = None
        try:
            from context_classifier import context_classifier
            off_court_percentage = context_classifier.get_off_court_percentage()
        except ImportError:
            pass
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "entities_processed": self.processing_stats["entities_processed"],
                "entities_passed": self.processing_stats["entities_passed"],
                "talk_time_processed": self.processing_stats["talk_time_processed"], 
                "talk_time_passed": self.processing_stats["talk_time_passed"],
                "hype_scores_processed": self.processing_stats["hype_scores_processed"],
                "hype_scores_validated": self.processing_stats["hype_scores_validated"],
                "entity_pass_rate": round(entity_pass_rate, 3),
                "talk_time_pass_rate": round(talk_time_pass_rate, 3),
                "hype_validation_rate": round(hype_validation_rate, 3),
                "average_confidence": round(avg_confidence, 3)
            },
            "ai_enhancement_stats": {
                "ai_enhanced_entities": ai_enhanced_count,
                "ai_enhancement_rate": round(ai_enhancement_rate, 3),
                "context_classification_available": off_court_percentage is not None,
                "off_court_percentage": off_court_percentage
            },
            "quality_distribution": {
                "high_quality_scores": len(self.quality_metrics['high_quality_scores']),
                "entities_passed": len(self.quality_metrics['entities_passed']),
                "talk_time_passed": len(self.quality_metrics['talk_time_passed']),
                "filtered_reasons": self._aggregate_filter_reasons()
            },
            "business_metrics": {
                "licensing_readiness": self._assess_licensing_readiness(avg_confidence, off_court_percentage),
                "data_quality_grade": self._calculate_quality_grade(entity_pass_rate, avg_confidence),
                "investor_readiness": "ready" if avg_confidence > 0.7 and entity_pass_rate > 0.8 else "needs_improvement"
            },
            "thresholds_used": self.thresholds.to_dict(),
            "recommendations": self._generate_recommendations(entity_pass_rate, avg_confidence, off_court_percentage)
        }
        
        return report
    
    def _aggregate_filter_reasons(self) -> Dict[str, int]:
        """Aggregate reasons for filtering."""
        reasons = defaultdict(int)
        for item in self.filtered_entities:
            # Extract main reason (before first colon or semicolon)
            main_reason = item['reason'].split(':')[0].split(';')[0].strip()
            reasons[main_reason] += 1
        return dict(reasons)
    
    def _assess_licensing_readiness(self, avg_confidence: float, off_court_pct: Optional[float]) -> str:
        """Assess readiness for licensing partners."""
        if avg_confidence > 0.8 and (off_court_pct is None or off_court_pct > 60):
            return "ready"
        elif avg_confidence > 0.6 and (off_court_pct is None or off_court_pct > 40):
            return "almost_ready"
        else:
            return "needs_improvement"
    
    def _calculate_quality_grade(self, pass_rate: float, confidence: float) -> str:
        """Calculate overall quality grade."""
        score = (pass_rate * 0.4 + confidence * 0.6)  # Weight confidence more heavily
        
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        else:
            return "D"
    
    def _generate_recommendations(
        self, 
        pass_rate: float, 
        avg_confidence: float, 
        off_court_pct: Optional[float]
    ) -> List[str]:
        """Generate actionable recommendations based on quality metrics."""
        recommendations = []
        
        # Pass rate recommendations
        if pass_rate < 0.5:
            recommendations.append("Consider lowering confidence thresholds - too many entities filtered")
        elif pass_rate > 0.95:
            recommendations.append("Consider raising confidence thresholds - may be too permissive")
        
        # Confidence recommendations
        if avg_confidence < 0.5:
            recommendations.append("Low overall confidence - check data quality or detection methods")
        elif avg_confidence > 0.85:
            recommendations.append("High confidence achieved - system performing excellently")
        
        # AI enhancement recommendations
        ai_enhanced_rate = len([item for item in self.filtered_entities if item.get('data_type') == 'AI-enhanced']) / max(len(self.filtered_entities), 1)
        if ai_enhanced_rate < 0.3:
            recommendations.append("Consider enabling AI-enhanced processing for better accuracy")
        
        # Business model recommendations
        if off_court_pct is not None:
            if off_court_pct < 40:
                recommendations.append("Off-court content percentage low - may not align with licensing strategy")
            elif off_court_pct > 80:
                recommendations.append("Off-court content very high - excellent for athlete influencer licensing")
        
        # Data source recommendations
        if len(self.quality_metrics['high_quality_scores']) < 5:
            recommendations.append("Few high-quality scores - may need more data sources or better AI classification")
        
        return recommendations


# Convenience functions for backward compatibility
def filter_entities_by_confidence(entity_data: Dict, threshold: float = None) -> Dict:
    """Legacy function - filter entities by confidence."""
    manager = QualityGateManager()
    passed, _ = manager.filter_entities_by_confidence(entity_data, threshold)
    return passed

def filter_talk_time_by_quality(talk_time_data: Dict, threshold: float = None) -> Dict:
    """Legacy function - filter talk time by quality."""
    manager = QualityGateManager()
    passed, _ = manager.filter_talk_time_by_quality(talk_time_data, threshold)
    return passed

def validate_hype_score_quality(hype_scores: Dict, confidence: Dict, data: Dict) -> Dict:
    """Legacy function - validate HYPE scores.""" 
    manager = QualityGateManager()
    return manager.validate_hype_score_quality(hype_scores, confidence, data)

def generate_quality_report() -> Dict:
    """Legacy function - generate quality report."""
    manager = QualityGateManager()
    return manager.generate_quality_report()