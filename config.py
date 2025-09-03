from datetime import datetime, timedelta
import os



# Set this to use a custom date range instead of predefined periods
USE_CUSTOM_DATES = True
CUSTOM_DATE_START = "20250824"  # Format: YYYYMMDD - Website Backfill: August 24-30, 2025
CUSTOM_DATE_END = "20250830"    # Format: YYYYMMDD - Website Backfill: August 24-30, 2025

# ✅ Define a Global Time Period for Data Collection
TIME_PERIOD = "last_30_days"  # Options: "last_7_days", "last_30_days", "last_6_months"

# ✅ Automatically Calculate Date Ranges
today = datetime.today()
DATE_RANGES = {
    "last_7_days": {
        "start": (today - timedelta(days=7)).strftime("%Y%m%d"),
        "end": today.strftime("%Y%m%d"),
    },
    "last_30_days": {
        "start": (today - timedelta(days=30)).strftime("%Y%m%d"),
        "end": today.strftime("%Y%m%d"),
    },
    "last_6_months": {
        "start": (today - timedelta(days=180)).strftime("%Y%m%d"),
        "end": today.strftime("%Y%m%d"),
    },
}

if USE_CUSTOM_DATES:
    # Create meaningful time period label for time-series
    TIME_PERIOD = f"week_{CUSTOM_DATE_START[:4]}_{CUSTOM_DATE_START[4:6]}_{CUSTOM_DATE_START[6:8]}"
    DATE_RANGES[TIME_PERIOD] = {
        "start": CUSTOM_DATE_START,
        "end": CUSTOM_DATE_END
    }
    print(f"🔍 Using CUSTOM date range: {CUSTOM_DATE_START} to {CUSTOM_DATE_END} (Period: {TIME_PERIOD})")

# ✅ Toggle Features for Full Production Run - ALL ENABLED
ENABLE_GOOGLE_TRENDS = True  # ✅ ENABLED for production customer report
ENABLE_WIKIPEDIA = True  # ✅ ENABLED for production customer report
ENABLE_REDDIT = True  # ✅ ENABLED for production customer report
ENABLE_GOOGLE_NEWS = True  # ✅ ENABLED for production customer report

print(f"🔍 Using TIME_PERIOD: {TIME_PERIOD} (Start: {DATE_RANGES[TIME_PERIOD]['start']}, End: {DATE_RANGES[TIME_PERIOD]['end']})")

# ✅ Select the Industry & Subcategory to Track
INDUSTRY = "Sports"  # Change to "Crypto", "Movies", etc.
SUBCATEGORY = "Unrivaled"  # Set to "Unrivaled" for website backfill (WNBA/Unrivaled only), None for all

# ✅ Auto-assign the correct entity file based on industry
ENTITY_FILES = {
    "Sports": "sports_entities.txt",
    "Movies": "movies_entities.txt",
    "Crypto": "crypto_entities.txt"
}

# Entity processing configuration
ENTITY_CONFIG_FILE = "entities.json"
ENABLE_PRONOUN_RESOLUTION = True
ENABLE_FUZZY_MATCHING = True
FUZZY_THRESHOLD = 0.8

# Entity debugging options
DEBUG_ENTITY_MATCHING = True
DEBUG_PRONOUN_RESOLUTION = True

print(f"🔧 Entity processing configuration loaded: Pronouns={ENABLE_PRONOUN_RESOLUTION}, Fuzzy={ENABLE_FUZZY_MATCHING}")

# Talk Time calculation settings
TALK_TIME_CONFIG = {
    "words_per_minute": 150,
    "words_after_entity": 50,
    "words_after_pronoun": 15,
    "context_window": 3,
    "pronoun_weight": 0.8,
    "enable_pronoun_tracking": True,
    "entity_defaults": {
        "type_detection": {
            "non_person_entities": [],
            "default_type": "person"
        },
        "gender_detection": {
            "female_default": True,
            "male_entities": [],
            "neutral_entities": []
        }
    }
}

# Audience metrics configuration for HYPE score calculation
AUDIENCE_METRICS_CONFIG = {
    "subscribers_weight": 1.0,
    "views_weight": 0.7,
    "downloads_weight": 0.8,
    "likes_weight": 1.2,
    "comments_weight": 1.5,
    "engagement_rate_min_threshold": 0.01,  # Minimum engagement rate to consider content valuable
    "audience_scaling": "logarithmic",  # Options: "linear", "logarithmic", "square_root"
    "platforms": {
        "Podcast": {
            "base_multiplier": 1.2,  # Podcasts might have more dedicated audiences
            "time_decay_factor": 0.95  # How quickly podcast content loses relevance
        },
        "YouTube": {
            "base_multiplier": 1.0,
            "time_decay_factor": 0.9
        },
        "default": {
            "base_multiplier": 1.0,
            "time_decay_factor": 0.9
        }
    }
}

print(f"📊 Audience metrics configuration loaded with {len(AUDIENCE_METRICS_CONFIG)} parameters")

# Podcast Pipeline Configuration
PODCAST_CONFIG = {
    "sources_file": "config/podcast_sources.json",
    "temp_directory": "temp/podcast_pipeline",
    "max_parallel_downloads": 2,
    "whisper_model": "base",
    "audio_quality": "192k",
    "cleanup_temp_files": True,
    "notification_email": os.environ.get("NOTIFICATION_EMAIL"),
    "enable_notifications": True,
    "max_processing_time_minutes": 60,
    "retry_attempts": 3,
    "retry_delay_seconds": 30
}

# Integration with existing TIME_PERIOD settings
PODCAST_DATE_FILTERING = {
    "use_custom_dates": USE_CUSTOM_DATES,
    "start_date": CUSTOM_DATE_START if USE_CUSTOM_DATES else None,
    "end_date": CUSTOM_DATE_END if USE_CUSTOM_DATES else None,
    "default_lookback_days": 30
}

print(f"🎙️ Podcast pipeline configuration loaded")

# ===== SEMANTIC DETECTION SETTINGS =====
# Enhanced entity detection using sentence transformers
SEMANTIC_THRESHOLD = 0.65  # Minimum similarity for entity match
SEMANTIC_CHUNK_SIZE = 100  # Words per chunk for analysis
SEMANTIC_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model
ENABLE_SEMANTIC_DETECTION = True  # Toggle for semantic vs keyword detection

# ===== DYNAMIC JORDN SETTINGS =====
# Adaptive weighting system for HYPE/JORDN scores
ENABLE_INTERACTION_EFFECTS = True  # Detect metric synergies
ENABLE_INFORMATION_GAIN = True     # Adjust weights by surprise factor
ENABLE_DYNAMIC_WEIGHTS = True      # Use adaptive vs static weights

# Base weights (will be adjusted dynamically)
BASE_JORDN_WEIGHTS = {
    'talk_time': 0.30,
    'mentions': 0.25, 
    'google_trends': 0.20,
    'reddit_mentions': 0.15,
    'wikipedia_views': 0.10
}

# ===== ENHANCED RODMN SETTINGS =====
# Advanced controversy detection components
MIN_SENTIMENTS_FOR_RODMN = 3  # Minimum data points needed
CONTROVERSY_COMPONENTS = {
    'polarization': 0.35,    # Positive vs negative distribution
    'volatility': 0.25,      # Rapid sentiment changes
    'intensity': 0.20,       # Extreme sentiment strength
    'disagreement': 0.20     # Sentiment reversals
}

# ===== AI CONTEXT CLASSIFICATION SETTINGS =====
# Enterprise AI-powered context classification system
CONTEXT_AI_CONFIG = {
    "enabled": True,
    "cache_size": 10000,
    "cache_similarity_threshold": 0.85,
    "confidence_threshold": 0.6,
    "ensemble_voting": True,
    "method_weights": {
        "zero_shot": 0.4,        # BART zero-shot classification  
        "embeddings": 0.3,       # Sentence transformer embeddings
        "nlp_features": 0.2,     # spaCy linguistic analysis
        "openai": 0.5            # OpenAI GPT (premium, highest weight)
    }
}

# Entity-type specific context weighting for business model alignment
CONTEXT_WEIGHTS_BY_TYPE = {
    # PEOPLE (Athletes, Influencers, Coaches) - "Athletes as Influencers" model
    "person": {
        "personal_life": 2.0,      # HIGH LICENSING VALUE - lifestyle, preferences, relationships
        "personality": 1.5,        # MEDIUM-HIGH VALUE - character traits, humor, opinions  
        "business": 1.2,           # MEDIUM VALUE - contracts, endorsements, career
        "controversy": 0.9,        # MEDIUM-LOW VALUE - divisive topics, criticism
        "performance": 0.8,        # LOW LICENSING VALUE - game stats, athletic achievements
        "brief_mention": 0.3       # MINIMAL VALUE - passing references
    },
    
    # NON-PERSON ENTITIES (Cryptocurrencies, Stocks, Teams) - Asset/Performance model
    "non-person": {
        "performance": 2.0,        # HIGH VALUE - price, metrics, technical performance
        "business": 1.8,           # HIGH VALUE - adoption, partnerships, institutional use
        "technology": 1.5,         # MEDIUM-HIGH VALUE - upgrades, features, innovation
        "controversy": 1.2,        # MEDIUM VALUE - regulatory issues, hacks, criticism
        "market_sentiment": 1.0,   # MEDIUM VALUE - general sentiment, speculation
        "brief_mention": 0.3,      # MINIMAL VALUE - passing references
        "personal_life": 0.1,      # NONSENSICAL - objects don't have personal lives
        "personality": 0.1         # NONSENSICAL - objects don't have personalities
    },
    
    # TEAMS/ORGANIZATIONS - Hybrid model
    "team": {
        "performance": 1.8,        # HIGH VALUE - wins, stats, achievements
        "business": 1.5,           # MEDIUM-HIGH VALUE - trades, deals, revenue
        "controversy": 1.2,        # MEDIUM VALUE - scandals, disputes
        "culture": 1.0,            # MEDIUM VALUE - team culture, management style
        "brief_mention": 0.3,      # MINIMAL VALUE - passing references
        "personal_life": 0.2,      # LOW VALUE - organizational culture aspects
        "personality": 0.2         # LOW VALUE - "team personality"
    }
}

# Backward compatibility - default to person weights
CONTEXT_WEIGHTS = CONTEXT_WEIGHTS_BY_TYPE["person"]

# Quality thresholds for AI-enhanced pipeline
QUALITY_THRESHOLDS = {
    "min_entity_confidence": 0.4,      # Minimum confidence for entity detections
    "min_context_confidence": 0.3,     # Minimum confidence for context classification
    "min_talk_time_confidence": 0.2,   # Minimum confidence for talk time attribution
    "min_hype_score_confidence": 0.5,  # Minimum confidence for final HYPE scores
    "enable_filtering": True            # Enable quality-based filtering
}

# ===== AI SYSTEM FEATURE FLAGS =====
# Control adaptive AI systems vs hardcoded fallbacks

# Semantic Context Classification
ENABLE_SEMANTIC_CONTEXT_CLASSIFICATION = True
CONTEXT_SIMILARITY_THRESHOLD = 0.5
SEMANTIC_FALLBACK_THRESHOLD = 0.4  # When to fallback to hardcoded rules

# Adaptive Quality Management
ENABLE_ADAPTIVE_QUALITY_THRESHOLDS = True
QUALITY_LEARNING_RATE = 0.1
QUALITY_ADAPTATION_MIN_SAMPLES = 20  # Min predictions before adapting thresholds

# HYPE Weight Optimization (already implemented)
ENABLE_ADAPTIVE_HYPE_WEIGHTS = True
HYPE_WEIGHT_EXPLORATION_RATE = 0.1

# AI Ensemble Optimization
ENABLE_ADAPTIVE_ENSEMBLE_WEIGHTS = True
ENSEMBLE_EXPLORATION_RATE = 0.05
ENSEMBLE_MIN_PREDICTIONS = 10  # Min predictions per method before optimization

# Integration Strategy Controls
FORCE_SEMANTIC_PRIMARY = True  # Make semantic classification primary method
HARDCODED_EMERGENCY_ONLY = True  # Use hardcoded rules only as emergency fallback
CONFIDENCE_BOOST_THRESHOLD = 0.7  # Boost confidence for consistently accurate methods

# Performance and Monitoring
ENABLE_PREDICTION_VALIDATION = True
VALIDATION_SAMPLE_RATE = 0.1  # Percentage of predictions to validate
LOG_AI_DECISIONS = True  # Log which AI system made each decision

# A/B Testing Controls  
ENABLE_AB_TESTING = True
AB_TEST_TRAFFIC_SPLIT = 0.5  # Percentage of traffic for new methods
AB_TEST_MIN_DURATION_HOURS = 24  # Minimum test duration

print("🧠 AI System Configuration Loaded:")
print(f"   Semantic Classification: {'ENABLED' if ENABLE_SEMANTIC_CONTEXT_CLASSIFICATION else 'DISABLED'}")
print(f"   Adaptive Quality Thresholds: {'ENABLED' if ENABLE_ADAPTIVE_QUALITY_THRESHOLDS else 'DISABLED'}")
print(f"   Adaptive HYPE Weights: {'ENABLED' if ENABLE_ADAPTIVE_HYPE_WEIGHTS else 'DISABLED'}")
print(f"   Ensemble Optimization: {'ENABLED' if ENABLE_ADAPTIVE_ENSEMBLE_WEIGHTS else 'DISABLED'}")