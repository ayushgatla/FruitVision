import os
import sys
import random
from datetime import datetime, timedelta

# Ensure backend can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database.database import SessionLocal, engine, Base
from backend.database.models import FruitInspection, ConveyorEvent

def seed_test_inspections():
    """
    Seed realistic industrial optical inspection data starting from October 25, 2026.
    Follows rule: skip 1 to 2 days in between each inspection batch.
    """
    db = SessionLocal()
    try:
        # Clear existing inspections to replace with clean realistic trajectory
        print("[FruitVision Seeder] Purging existing cluttered test records...")
        db.query(FruitInspection).delete()
        db.commit()

        # Batch definitions starting from Oct 25, skipping 1 to 2 days in between
        # Gaps:
        # Oct 25 -> Oct 27 (1 day gap: Oct 26 skipped)
        # Oct 27 -> Oct 30 (2 days gap: Oct 28, 29 skipped)
        # Oct 30 -> Nov 01 (1 day gap: Oct 31 skipped)
        # Nov 01 -> Nov 03 (1 day gap: Nov 02 skipped)
        # Nov 03 -> Nov 05 (1 day gap: Nov 04 skipped)
        batches = [
            {
                "date": datetime(2026, 10, 25),
                "good": 36,
                "damaged": 2,
                "rotten": 0,
                "batch_id": "BATCH-20261025-A1"
            },
            {
                "date": datetime(2026, 10, 27),
                "good": 40,
                "damaged": 1,
                "rotten": 1,
                "batch_id": "BATCH-20261027-B2"
            },
            {
                "date": datetime(2026, 10, 30),
                "good": 33,
                "damaged": 2,
                "rotten": 1,
                "batch_id": "BATCH-20261030-C3"
            },
            {
                "date": datetime(2026, 11, 1),
                "good": 43,
                "damaged": 2,
                "rotten": 0,
                "batch_id": "BATCH-20261101-D4"
            },
            {
                "date": datetime(2026, 11, 3),
                "good": 38,
                "damaged": 1,
                "rotten": 1,
                "batch_id": "BATCH-20261103-E5"
            },
            {
                "date": datetime(2026, 11, 5),
                "good": 33,
                "damaged": 2,
                "rotten": 0,
                "batch_id": "BATCH-20261105-F6"
            }
        ]

        good_images = [
            "/sample_images/sample_good_apple.jpg",
            "/sample_images/sample_good_red_apple.jpg",
            "/sample_images/sample_good_orange.jpg",
            "/sample_images/sample_good_banana.jpg",
        ]
        damaged_images = [
            "/sample_images/sample_damaged_apple.jpg",
        ]
        rotten_images = [
            "/sample_images/sample_rotten_apple.jpg",
            "/sample_images/sample_real_rotten_apple.jpg",
            "/sample_rotten_banana.jpg",
        ]

        good_notes = [
            "Passed optical grading - Grade A retail packaging",
            "Conveyor belt high-speed optical verification - intact surface",
            "Grade A certification - surface integrity verified",
            "Export packaging grade - flawless skin coloration",
            "Optimal ripeness index detected - routed to Gate 1 (Accepted)"
        ]

        damaged_notes = [
            "Secondary manual grading required - minor peel abrasion",
            "Peel surface bruising detected - routed to juice/puree secondary unit",
            "Superficial mechanical scuff flagged - secondary inspection"
        ]

        rotten_notes = [
            "Fungal decay detected - automated diverter ejected to composting",
            "Necrotic tissue threshold breached - rejected from food supply chain",
            "Severe fungal lesion detected - routed to bio-waste bin"
        ]

        sources = ["conveyor_simulation", "single_upload", "conveyor_simulation", "multi_upload"]

        total_seeded = 0
        now = datetime.utcnow()

        for batch in batches:
            b_date = batch["date"]
            b_good = batch["good"]
            b_damaged = batch["damaged"]
            b_rotten = batch["rotten"]
            batch_label = batch["batch_id"]

            records = []
            # Create Good records
            for _ in range(b_good):
                records.append({
                    "class": "Good",
                    "status": "ACCEPTED",
                    "good": 1,
                    "damaged": 0,
                    "rotten": 0,
                    "img": random.choice(good_images),
                    "conf": round(random.uniform(0.92, 0.99), 3),
                    "note": f"{batch_label}: {random.choice(good_notes)}"
                })

            # Create Damaged records
            for _ in range(b_damaged):
                records.append({
                    "class": "Damaged",
                    "status": "SECONDARY_INSPECTION",
                    "good": 0,
                    "damaged": 1,
                    "rotten": 0,
                    "img": random.choice(damaged_images),
                    "conf": round(random.uniform(0.88, 0.95), 3),
                    "note": f"{batch_label}: {random.choice(damaged_notes)}"
                })

            # Create Rotten records
            for _ in range(b_rotten):
                records.append({
                    "class": "Rotten",
                    "status": "REJECTED",
                    "good": 0,
                    "damaged": 0,
                    "rotten": 1,
                    "img": random.choice(rotten_images),
                    "conf": round(random.uniform(0.92, 0.98), 3),
                    "note": f"{batch_label}: {random.choice(rotten_notes)}"
                })

            random.shuffle(records)

            is_today = (b_date.date() == now.date())
            recent_alert_assigned = False

            for idx, r in enumerate(records):
                if is_today:
                    # Distribute through today, keeping one alert in the last 30 minutes
                    if not recent_alert_assigned and r["class"] in ["Damaged", "Rotten"]:
                        rec_time = now - timedelta(minutes=18)
                        recent_alert_assigned = True
                    else:
                        hour_offset = random.randint(1, 8)
                        minute_offset = random.randint(0, 59)
                        rec_time = now - timedelta(hours=hour_offset, minutes=minute_offset)
                else:
                    hour = random.randint(8, 17)
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    rec_time = datetime(b_date.year, b_date.month, b_date.day, hour, minute, second)

                item = FruitInspection(
                    image_path=r["img"],
                    evidence_path=r["img"],
                    predicted_class=r["class"],
                    confidence=r["conf"],
                    inspection_timestamp=rec_time,
                    inspection_status=r["status"],
                    fruit_count=1,
                    good_count=r["good"],
                    damaged_count=r["damaged"],
                    rotten_count=r["rotten"],
                    source_type=random.choice(sources),
                    notes=r["note"]
                )
                db.add(item)
                total_seeded += 1

        db.commit()
        print(f"[FruitVision Seeder] Successfully seeded {total_seeded} inspection records across 6 batches starting from Oct 25!")

    except Exception as e:
        db.rollback()
        print(f"[FruitVision Seeder] Error seeding test data: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    seed_test_inspections()
