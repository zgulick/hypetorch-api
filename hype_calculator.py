import numpy as np
import math
import time
from datetime import datetime
from config import AUDIENCE_METRICS_CONFIG, BASE_JORDN_WEIGHTS, ENABLE_DYNAMIC_WEIGHTS, ENABLE_INTERACTION_EFFECTS, ENABLE_INFORMATION_GAIN
from adaptive_hype_optimizer import hype_optimizer

def calculate_weighted_audience_score(metrics, config=None):
    """Calculate a weighted audience score based on available metrics and configuration."""
    if config is None:
        config = AUDIENCE_METRICS_CONFIG
    
    base_score = 0
    
    # Calculate weighted sum of available metrics
    if "subscribers" in metrics:
        base_score += metrics["subscribers"] * config["subscribers_weight"]
    if "views" in metrics:
        base_score += metrics["views"] * config["views_weight"]
    if "downloads" in metrics:
        base_score += metrics["downloads"] * config["downloads_weight"]
    if "likes" in metrics:
        base_score += metrics["likes"] * config["likes_weight"]
    if "comments" in metrics:
        base_score += metrics["comments"] * config["comments_weight"]
    
    # If no metrics are available, return a default small value
    if base_score == 0:
        return 1  # Default minimum score
    
    # Apply platform-specific multiplier
    platform = metrics.get("platform", "default")
    platform_config = config["platforms"].get(platform, 
                                           config["platforms"].get("default", {"base_multiplier": 1.0}))
    base_score *= platform_config["base_multiplier"]
    
    # Apply audience scaling (prevents massive audiences from dominating)
    if config["audience_scaling"] == "logarithmic":
        return math.log1p(base_score)  # log(1+x) to handle zero values
    elif config["audience_scaling"] == "square_root":
        return math.sqrt(base_score)
    else:  # linear is default
        return base_score

def apply_time_decay(score, publication_date, config=None):
    """Apply time decay to scores based on age of content."""
    if config is None:
        config = AUDIENCE_METRICS_CONFIG
        
    if not publication_date:
        return score
    
    # Convert string date to datetime if needed
    if isinstance(publication_date, str):
        try:
            publication_date = datetime.strptime(publication_date.split("T")[0], '%Y-%m-%d')
        except (ValueError, TypeError):
            return score  # Return original score if date parsing fails
    
    days_old = (datetime.now() - publication_date).days
    if days_old < 0:  # Future dates shouldn't get a boost
        days_old = 0
        
    platform = config.get("platform", "default")
    decay_factor = config["platforms"].get(platform, {}).get("time_decay_factor", 0.9)
    
    return score * (decay_factor ** days_old)

def apply_exponential_decay(values, timestamps, half_life_days=14):
    """Applies exponential decay to values based on their timestamps."""
    now = time.time()  # Get the current timestamp
    half_life = half_life_days * 86400  # Convert days to seconds

    decayed_values = {
        key: value * math.exp(- (now - timestamps.get(key, now)) / half_life)
        for key, value in values.items()
    }
    return decayed_values

def apply_audience_weight(talk_time, audience_size):
    """Weight talk time by audience size using log scaling."""
    return {
        player: round(value * math.log(1 + audience_size.get(player, 1)), 2)  # log(1 + audience)
        for player, value in talk_time.items()
    }

def z_score_normalization(data):
    """Applies Z-score normalization to prevent extreme values from dominating."""
    values = list(data.values())
    if not values:
        return {}  # Return empty dict if no values
        
    mean = sum(values) / len(values) if values else 0
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)) if values else 1
    
    # Avoid division by zero
    std = std if std > 0 else 1  
    
    normalized = {key: (value - mean) / std for key, value in data.items()}
    return normalized

def calculate_dynamic_weights(data, base_weights):
    """Calculate dynamic weights based on data quality and information gain."""
    if not ENABLE_DYNAMIC_WEIGHTS:
        return base_weights
    
    dynamic_weights = base_weights.copy()
    
    # 1. Data Quality Adjustment - Zero out weights for missing metrics
    metric_mapping = {
        'talk_time': "Talk Time (Minutes)",
        'mentions': "Mentions", 
        'google_trends': "Google Trends",
        'reddit_mentions': "Reddit Mentions",
        'wikipedia_views': "Wikipedia Views"
    }
    
    total_weight_removed = 0
    active_metrics = []
    
    for weight_key, data_key in metric_mapping.items():
        metric_data = data.get(data_key, {})
        if not metric_data or all(v == 0 for v in metric_data.values()):
            # No data for this metric - remove its weight
            total_weight_removed += dynamic_weights[weight_key]
            dynamic_weights[weight_key] = 0
        else:
            active_metrics.append(weight_key)
    
    # 2. Redistribute removed weight to active metrics
    if total_weight_removed > 0 and active_metrics:
        redistribution = total_weight_removed / len(active_metrics)
        for metric in active_metrics:
            dynamic_weights[metric] += redistribution
    
    # 3. Information Gain Adjustment (if enabled)
    if ENABLE_INFORMATION_GAIN and active_metrics:
        for weight_key in active_metrics:
            data_key = metric_mapping[weight_key]
            metric_values = list(data.get(data_key, {}).values())
            if metric_values:
                # Calculate coefficient of variation as surprise factor
                mean_val = sum(metric_values) / len(metric_values)
                if mean_val > 0:
                    std_val = math.sqrt(sum((x - mean_val) ** 2 for x in metric_values) / len(metric_values))
                    cv = std_val / mean_val  # Coefficient of variation
                    # Higher variation = more informative = higher weight
                    info_gain = min(cv * 0.5, 0.3)  # Cap at 30% boost
                    dynamic_weights[weight_key] *= (1 + info_gain)
    
    # 4. Normalize weights to sum to 1.0
    weight_sum = sum(dynamic_weights.values())
    if weight_sum > 0:
        dynamic_weights = {k: v / weight_sum for k, v in dynamic_weights.items()}
    
    return dynamic_weights

def calculate_interaction_effects(normalized_data):
    """Calculate interaction effect multipliers for viral moments and breakthroughs."""
    if not ENABLE_INTERACTION_EFFECTS:
        return {}
    
    interaction_multipliers = {}
    
    for entity in normalized_data.get("Mentions", {}):
        multiplier = 1.0
        
        talk_time = normalized_data.get("Talk Time (Minutes)", {}).get(entity, 0)
        mentions = normalized_data.get("Mentions", {}).get(entity, 0)
        google_trends = normalized_data.get("Google Trends", {}).get(entity, 0)
        reddit = normalized_data.get("Reddit Mentions", {}).get(entity, 0)
        
        # Viral moment: High talk time + high mentions
        if talk_time > 1.0 and mentions > 1.0:  # Both above average (z-score > 1)
            multiplier *= 1.3
            
        # Mainstream breakthrough: High Google trends + low talk time  
        elif google_trends > 1.0 and talk_time < 0:  # Trending but not talked about much
            multiplier *= 1.2
            
        # Community debate: High Reddit mentions
        if reddit > 1.0:
            multiplier *= 1.15
        
        interaction_multipliers[entity] = multiplier
    
    return interaction_multipliers

def extract_mention_counts(mentions_data):
    """
    Extract mention counts from either legacy format or AI-enhanced format.
    
    Args:
        mentions_data: Dict containing mention data in legacy or AI format
        
    Returns:
        Dict with entity -> count mapping for downstream processing
    """
    
    extracted_counts = {}
    
    if not mentions_data:
        return extracted_counts
    
    for entity, mention_info in mentions_data.items():
        if isinstance(mention_info, (int, float)):
            # Legacy format - simple count
            extracted_counts[entity] = mention_info
        elif isinstance(mention_info, dict):
            # AI-enhanced format - use weighted count if available
            if "weighted_count" in mention_info:
                extracted_counts[entity] = mention_info["weighted_count"]
                print(f"🤖 AI-weighted mentions for {entity}: {mention_info.get('raw_count', 0)} → {mention_info['weighted_count']}")
            elif "raw_count" in mention_info:
                extracted_counts[entity] = mention_info["raw_count"] 
            elif "count" in mention_info:
                extracted_counts[entity] = mention_info["count"]
            else:
                # Fallback - assume it's a count if it's a number
                extracted_counts[entity] = 1
        else:
            # Unknown format - conservative fallback
            extracted_counts[entity] = 1
    
    return extracted_counts


def calculate_hype_scores_with_confidence(data, audience_size=None, timestamps=None):
    """
    Enhanced HYPE score calculation with AI context weighting and confidence tracking.
    
    Supports both legacy mention format and AI-enhanced mention format with context weighting.
    """
    from confidence_scoring import calculate_hype_score_confidence
    
    if audience_size is None:
        audience_size = {}
    if timestamps is None:
        timestamps = {}
    
    print("🚀 Calculating AI-enhanced HYPE scores...")
    
    # Extract mention counts from AI-enhanced or legacy format
    raw_mentions_data = data.get("Mentions", {})
    mention_counts = extract_mention_counts(raw_mentions_data)
    
    # Check if we have AI-enhanced mentions data
    ai_enhanced = any(isinstance(v, dict) and v.get("ai_enhanced", False) for v in raw_mentions_data.values())
    quality_filtered_count = sum(1 for v in raw_mentions_data.values() if isinstance(v, dict) and v.get("quality_filtered", False))
    
    if ai_enhanced:
        print(f"🤖 Processing AI-enhanced mentions: {len(mention_counts)} entities, {quality_filtered_count} quality-filtered")
    else:
        print(f"📊 Processing legacy mentions: {len(mention_counts)} entities")
    
    # Build data structure for processing (replacing Mentions with extracted counts)
    processed_data = data.copy()
    processed_data["Mentions"] = mention_counts
    
    # Apply transformations
    weighted_talk_time = apply_audience_weight(processed_data.get("Talk Time (Minutes)", {}), audience_size)
    
    # Apply exponential decay before normalization if timestamps are provided
    decayed_talk_time = apply_exponential_decay(processed_data.get("Talk Time (Minutes)", {}), timestamps)
    decayed_mentions = apply_exponential_decay(mention_counts, timestamps)
    
    # Now normalize the decayed values
    normalized_talk_time = z_score_normalization(decayed_talk_time if timestamps else processed_data.get("Talk Time (Minutes)", {}))
    normalized_mentions = z_score_normalization(decayed_mentions if timestamps else mention_counts)
    normalized_google_trends = z_score_normalization(processed_data.get("Google Trends", {}))
    normalized_reddit_mentions = z_score_normalization(processed_data.get("Reddit Mentions", {}))
    normalized_google_news = z_score_normalization(processed_data.get("Google News Mentions", {}))
    normalized_wikipedia_views = z_score_normalization(processed_data.get("Wikipedia Views", {}))
    
    # Print debugging info
    print(f"🔍 DEBUG: Normalized Google Trends - {len(normalized_google_trends)} entities")
    print(f"🔍 DEBUG: Normalized Talk Time - {len(normalized_talk_time)} entities")
    print(f"🔍 DEBUG: Normalized Mentions (AI-weighted) - {len(normalized_mentions)} entities")
    
    # 🧠 SMART AI: Get optimal weights using machine learning optimization
    try:
        # Determine entity characteristics for intelligent weight selection
        entity_types = {}
        context_distributions = {}
        trending_entities = set()
        
        # Analyze entities to determine optimal weight context
        for entity, mentions_data in raw_mentions_data.items():
            if isinstance(mentions_data, dict):
                # Extract entity characteristics from AI-enhanced data
                entity_types[entity] = mentions_data.get("entity_type", "person")
                context_distributions[entity] = mentions_data.get("context_breakdown", {})
                
                # Determine if trending (high mention velocity)
                raw_count = mentions_data.get("raw_count", 0)
                if raw_count > 10:  # Arbitrary threshold for "trending"
                    trending_entities.add(entity)
        
        # Calculate data completeness for each component
        data_completeness = {}
        total_entities = len(mention_counts)
        if total_entities > 0:
            data_completeness["mentions"] = len([e for e in mention_counts.values() if e > 0]) / total_entities
            data_completeness["talk_time"] = len([e for e in processed_data.get("Talk Time (Minutes)", {}).values() if e > 0]) / total_entities
            data_completeness["google_trends"] = len([e for e in processed_data.get("Google Trends", {}).values() if e > 0]) / total_entities
            data_completeness["reddit_mentions"] = len([e for e in processed_data.get("Reddit Mentions", {}).values() if e > 0]) / total_entities
            data_completeness["wikipedia_views"] = len([e for e in processed_data.get("Wikipedia Views", {}).values() if e > 0]) / total_entities
        
        # Get smart optimized weights (this replaces hardcoded BASE_JORDN_WEIGHTS!)
        smart_weights = {}
        for entity in mention_counts:
            entity_type = entity_types.get(entity, "person")
            context_dist = context_distributions.get(entity, {})
            is_trending = entity in trending_entities
            
            optimal_weights = hype_optimizer.get_optimal_weights(
                entity_type=entity_type,
                context_distribution=context_dist,
                is_trending=is_trending,
                data_completeness=data_completeness
            )
            smart_weights[entity] = optimal_weights
        
        # Use most common optimal weights as default, or fallback to baseline
        if smart_weights:
            # Average the weights across all entities for consistency
            avg_weights = {}
            for component in ["talk_time", "mentions", "google_trends", "reddit_mentions", "wikipedia_views"]:
                weights_for_component = [w.get(component, BASE_JORDN_WEIGHTS.get(component, 0.2)) 
                                       for w in smart_weights.values()]
                avg_weights[component] = sum(weights_for_component) / len(weights_for_component)
            
            weights = avg_weights
            print(f"🧠 Using AI-optimized weights: Talk:{weights['talk_time']:.2f}, "
                  f"Mentions:{weights['mentions']:.2f}, Trends:{weights['google_trends']:.2f}")
        else:
            # Fallback to original dynamic calculation
            weights = calculate_dynamic_weights(processed_data, BASE_JORDN_WEIGHTS)
            print("⚠️ Using fallback dynamic weights calculation")
            
    except Exception as e:
        # Fallback to original dynamic calculation if smart optimization fails
        weights = calculate_dynamic_weights(processed_data, BASE_JORDN_WEIGHTS)
        print(f"⚠️ Smart weight optimization failed ({e}), using fallback dynamic weights")
    
    # Create normalized data structure for interaction effects
    normalized_data = {
        "Talk Time (Minutes)": normalized_talk_time,
        "Mentions": normalized_mentions,
        "Google Trends": normalized_google_trends,
        "Reddit Mentions": normalized_reddit_mentions,
        "Wikipedia Views": normalized_wikipedia_views
    }
    
    # Calculate interaction effect multipliers
    interaction_multipliers = calculate_interaction_effects(normalized_data)
    
    # Compute Hype Score with dynamic weights and confidence tracking
    hype_scores = {}
    confidence_scores = {}
    
    for player in mention_counts:
        base_score = (
            normalized_talk_time.get(player, 0) * weights['talk_time'] +
            normalized_mentions.get(player, 0) * weights['mentions'] +
            normalized_google_trends.get(player, 0) * weights['google_trends'] +
            normalized_reddit_mentions.get(player, 0) * weights['reddit_mentions'] +
            normalized_wikipedia_views.get(player, 0) * weights['wikipedia_views']
        )
        
        # Apply interaction effects multiplier
        multiplier = interaction_multipliers.get(player, 1.0)
        hype_scores[player] = base_score * multiplier
        
        # Calculate confidence for this HYPE score
        data_completeness = {}
        if normalized_talk_time.get(player, 0) != 0:
            data_completeness["talk_time"] = 1.0
        if normalized_mentions.get(player, 0) != 0:
            data_completeness["mentions"] = 1.0
        if normalized_google_trends.get(player, 0) != 0:
            data_completeness["google_trends"] = 1.0
        if normalized_reddit_mentions.get(player, 0) != 0:
            data_completeness["reddit"] = 1.0
        if normalized_wikipedia_views.get(player, 0) != 0:
            data_completeness["wikipedia"] = 1.0
        
        # Include AI confidence if available
        ai_confidence = None
        if isinstance(raw_mentions_data.get(player), dict):
            ai_confidence = raw_mentions_data[player].get("confidence", 0.5)
        
        sample_size = len(data_completeness)
        hype_confidence = calculate_hype_score_confidence(
            data_completeness=data_completeness,
            sample_size=sample_size,
            temporal_consistency=0.7  # Default consistency
        )
        
        # Boost confidence if AI enhanced the mentions
        if ai_confidence is not None:
            hype_confidence = min(0.95, hype_confidence * (1 + ai_confidence * 0.2))
        
        confidence_scores[player] = hype_confidence
    
    # Fix scaling so that the minimum score is above 0 and average is 100
    min_score = min(hype_scores.values()) if hype_scores else 0
    adjusted_scores = {player: score - min_score + 1 for player, score in hype_scores.items()}
    
    average_score = sum(adjusted_scores.values()) / len(adjusted_scores) if adjusted_scores else 1
    scaled_hype_scores = {player: round((score / average_score) * 100, 2) for player, score in adjusted_scores.items()}
    
    # Log final statistics
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
    high_confidence_count = sum(1 for c in confidence_scores.values() if c > 0.8)
    
    # 🎓 LEARNING: Record results for weight optimization (if we used smart weights)
    try:
        if 'smart_weights' in locals() and smart_weights:
            for entity, hype_score in scaled_hype_scores.items():
                entity_weights = smart_weights.get(entity, weights)
                entity_type = entity_types.get(entity, "person")
                context_dist = context_distributions.get(entity, {})
                is_trending = entity in trending_entities
                
                # Determine context key for learning
                context_key = hype_optimizer._determine_context_key(entity_type, context_dist, is_trending)
                
                # Record prediction for future learning (no actual performance yet)
                hype_optimizer.record_prediction_result(
                    entity_name=entity,
                    context_key=context_key,
                    weights_used=entity_weights,
                    predicted_hype=hype_score,
                    actual_performance=None,  # Would need external metrics to validate
                    engagement_metrics={"confidence": confidence_scores.get(entity, 0.5)}
                )
            
            print(f"🎓 Recorded {len(scaled_hype_scores)} predictions for weight optimization learning")
    except Exception as e:
        print(f"⚠️ Could not record learning data: {e}")
    
    print(f"✅ AI-Enhanced HYPE scores calculated:")
    print(f"   📊 {len(scaled_hype_scores)} entities processed")
    print(f"   🤖 AI-enhanced mentions: {'Yes' if ai_enhanced else 'No'}")
    print(f"   📈 Average confidence: {avg_confidence:.3f}")
    print(f"   🎯 High confidence entities: {high_confidence_count}/{len(confidence_scores)}")
    
    # Return enhanced results with confidence data
    return {
        "hype_scores": scaled_hype_scores,
        "confidence_scores": confidence_scores,
        "processing_stats": {
            "ai_enhanced": ai_enhanced,
            "entities_processed": len(scaled_hype_scores),
            "quality_filtered_count": quality_filtered_count,
            "avg_confidence": avg_confidence,
            "high_confidence_count": high_confidence_count
        }
    }


def calculate_hype_scores(data, audience_size=None, timestamps=None):
    """
    Legacy interface for HYPE score calculation - maintains backward compatibility.
    
    Returns simple dict for backward compatibility, but uses AI enhancements internally.
    """
    
    # Use the enhanced calculation internally
    enhanced_results = calculate_hype_scores_with_confidence(data, audience_size, timestamps)
    
    # Return just the scores for backward compatibility
    return enhanced_results["hype_scores"]