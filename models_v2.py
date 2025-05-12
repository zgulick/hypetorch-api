# models_v2.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Request models
class BulkEntitiesRequest(BaseModel):
    """Request model for bulk entity data."""
    entity_names: List[str] = Field(..., description="List of entity names to retrieve")
    include_history: bool = Field(False, description="Include historical data")
    history_days: Optional[int] = Field(30, description="Days of history to include")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to include")

# Response models
class EntityMetrics(BaseModel):
    """Current metrics for an entity."""
    hype_score: Optional[float] = None
    rodmn_score: Optional[float] = None
    talk_time: Optional[float] = None
    mentions: Optional[int] = None
    sentiment: Optional[float] = None
    wikipedia_views: Optional[int] = None
    reddit_mentions: Optional[int] = None
    google_trends: Optional[int] = None

class EntityData(BaseModel):
    """Complete entity data."""
    id: int
    name: str
    type: str = Field(default="person", description="Entity type")
    category: str
    subcategory: str
    metrics: Optional[EntityMetrics] = None
    history: Optional[Dict[str, List]] = None