from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from .database import Base

class FruitInspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_path = Column(String(255), nullable=True)
    predicted_class = Column(String(50), nullable=False, index=True) # "Good", "Damaged", "Rotten"
    confidence = Column(Float, nullable=False)
    inspection_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    inspection_status = Column(String(50), default="COMPLETED") # "ACCEPTED", "SECONDARY_INSPECTION", "REJECTED"
    
    # Extended metrics
    fruit_count = Column(Integer, default=1)
    good_count = Column(Integer, default=0)
    damaged_count = Column(Integer, default=0)
    rotten_count = Column(Integer, default=0)
    source_type = Column(String(50), default="single_upload") # "single_upload", "multi_upload", "conveyor_simulation", "webcam"
    notes = Column(Text, nullable=True)
    evidence_path = Column(String(255), nullable=True)

class ConveyorEvent(Base):
    __tablename__ = "conveyor_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    item_id = Column(String(50), nullable=False)
    fruit_type = Column(String(50), default="Apple")
    predicted_quality = Column(String(50), nullable=False) # "Good", "Damaged", "Rotten"
    confidence = Column(Float, nullable=False)
    sorting_action = Column(String(100), nullable=False)
    gate_routed = Column(String(50), nullable=False) # "GATE_1_ACCEPTED", "GATE_2_SECONDARY", "GATE_3_REJECTED"

