from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List

from backend.database.database import get_db
from backend.database.models import FruitInspection

router = APIRouter()

@router.get("")
def get_inspection_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None), # "Good", "Damaged", "Rotten"
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Retrieve indexed inspection history records with search and category filters."""
    query = db.query(FruitInspection)

    if category and category.upper() != "ALL":
        query = query.filter(FruitInspection.predicted_class == category)

    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (FruitInspection.notes.like(search_filter)) |
            (FruitInspection.predicted_class.like(search_filter)) |
            (FruitInspection.inspection_status.like(search_filter))
        )

    total_records = query.count()
    inspections = (
        query.order_by(desc(FruitInspection.inspection_timestamp))
        .offset(offset)
        .limit(limit)
        .all()
    )

    records = []
    for ins in inspections:
        records.append({
            "id": ins.id,
            "timestamp": ins.inspection_timestamp.isoformat(),
            "predicted_class": ins.predicted_class,
            "confidence": ins.confidence,
            "inspection_status": ins.inspection_status,
            "fruit_count": ins.fruit_count,
            "good_count": ins.good_count,
            "damaged_count": ins.damaged_count,
            "rotten_count": ins.rotten_count,
            "source_type": ins.source_type,
            "image_path": ins.image_path,
            "evidence_path": ins.evidence_path or ins.image_path,
            "notes": ins.notes
        })

    return {
        "total": total_records,
        "limit": limit,
        "offset": offset,
        "records": records
    }

@router.delete("/{inspection_id}")
def delete_inspection_record(inspection_id: int, db: Session = Depends(get_db)):
    """Delete a specific inspection record from SQLite."""
    record = db.query(FruitInspection).filter(FruitInspection.id == inspection_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Inspection record not found.")

    db.delete(record)
    db.commit()
    return {"status": "success", "deleted_id": inspection_id}

@router.delete("")
def purge_all_inspections(db: Session = Depends(get_db)):
    """Purge all inspection records from SQLite."""
    deleted_count = db.query(FruitInspection).delete()
    db.commit()
    return {"status": "success", "deleted_count": deleted_count}

