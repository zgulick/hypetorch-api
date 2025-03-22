# api_models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

class EntityBase(BaseModel):
    """Base model for entity data."""
    name: str = Field(..., description="Entity name", min_length=2)
    type: str = Field("person", description="Entity type (person or non-person)")
    category: str = Field("Sports", description="Category")
    subcategory: str = Field("Unrivaled", description="Subcategory")
    
    @validator('type')
    def validate_type(cls, v):
        if v not in ['person', 'non-person']:
            raise ValueError('Type must be either "person" or "non-person"')
        return v

class EntityCreate(EntityBase):
    """Model for creating a new entity."""
    aliases: Optional[List[str]] = Field(None, description="Alternative names for the entity")
    gender: Optional[str] = Field(None, description="Gender (for person entities)")
    
    @validator('gender')
    def validate_gender(cls, v, values):
        if v and values.get('type') == 'person' and v not in ['male', 'female', 'neutral']:
            raise ValueError('Gender must be one of: male, female, neutral')
        return v

class EntityUpdate(BaseModel):
    """Model for updating an existing entity."""
    name: Optional[str] = Field(None, description="Entity name", min_length=2)
    type: Optional[str] = Field(None, description="Entity type (person or non-person)")
    category: Optional[str] = Field(None, description="Category")
    subcategory: Optional[str] = Field(None, description="Subcategory")
    aliases: Optional[List[str]] = Field(None, description="Alternative names for the entity")
    
    @validator('type')
    def validate_type(cls, v):
        if v is not None and v not in ['person', 'non-person']:
            raise ValueError('Type must be either "person" or "non-person"')
        return v

class MetricsQuery(BaseModel):
    """Query parameters for metrics endpoints."""
    start_date: Optional[datetime] = Field(None, description="Start date for metrics")
    end_date: Optional[datetime] = Field(None, description="End date for metrics")
    time_period: Optional[str] = Field(None, description="Time period (e.g., last_30_days)")
    
    @validator('time_period')
    def validate_time_period(cls, v):
        if v is not None and v not in ['last_7_days', 'last_30_days', 'last_6_months']:
            raise ValueError('Time period must be one of: last_7_days, last_30_days, last_6_months')
        return v

class BulkEntityQuery(BaseModel):
    """Parameters for bulk entity queries."""
    entities: List[str] = Field(..., description="List of entity names", min_items=1)
    metrics: Optional[List[str]] = Field(None, description="Metrics to include")
    include_history: bool = Field(False, description="Include historical data")
    history_limit: Optional[int] = Field(30, description="Max history points", ge=1, le=100)
    time_period: Optional[str] = Field(None, description="Time period")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        if v:
            valid_metrics = [
                'hype_score', 'mentions', 'talk_time', 'sentiment', 'wikipedia_views',
                'reddit_mentions', 'google_trends', 'rodmn_score'
            ]
            for metric in v:
                if metric not in valid_metrics:
                    raise ValueError(f'Invalid metric: {metric}. Valid metrics are: {", ".join(valid_metrics)}')
        return v