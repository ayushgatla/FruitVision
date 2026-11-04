from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QualityPrediction(BaseModel):
    label: str # "Good", "Damaged", "Rotten"
    confidence: float # 0.0 - 1.0
    probabilities: Dict[str, float]
    suggested_action: str
    status: str # "ACCEPTED", "SECONDARY_INSPECTION", "REJECTED"

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    status: str

class SingleInspectionResponse(BaseModel):
    id: Optional[int] = None
    predicted_class: str
    confidence: float
    status: str
    suggested_action: str
    probabilities: Dict[str, float]
    image_url: str
    annotated_image_url: Optional[str] = None
    timestamp: str
    device: str
    model_version: str

class MultiFruitInspectionResponse(BaseModel):
    id: Optional[int] = None
    total_fruits: int
    good_count: int
    damaged_count: int
    rotten_count: int
    acceptance_rate: float
    overall_status: str
    detections: List[BoundingBox]
    image_url: str
    annotated_image_url: Optional[str] = None
    timestamp: str

class ConveyorSimulationSampleResponse(BaseModel):
    item_id: str
    fruit_type: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    sorting_gate: str # "GATE_1_ACCEPTED", "GATE_2_SECONDARY", "GATE_3_REJECTED"
    action: str
    image_url: str
    timestamp: str

class AnalyticsSummaryResponse(BaseModel):
    total_inspected: int
    good_fruits: int
    damaged_fruits: int
    rotten_fruits: int
    acceptance_rate: float
    rejection_rate: float
    secondary_rate: float
    today_inspected: int
    active_alerts: int
    daily_trends: List[Dict[str, Any]]
    class_distribution: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_path: str
    using_custom_weights: bool
    classes: List[str]
    db_status: str
    version: str

