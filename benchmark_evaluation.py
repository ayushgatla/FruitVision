import os
import time
import glob
import json
from datetime import datetime
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.database import SessionLocal
from backend.database.models import FruitInspection

def run_fruitvision_benchmark():
    os.makedirs("test_results/annotated", exist_ok=True)
    
    with TestClient(app) as client:
        db = SessionLocal()

        image_files = sorted(glob.glob("sample_images/*.jpg") + glob.glob("sample_images/*.png"))
        print("\n" + "=" * 80)
        print(" FRUITVISION AI — INDUSTRIAL IMAGE INFERENCE BENCHMARK")
        print(f" Total Images in Test Suite: {len(image_files)}")
        print("=" * 80)

        results = []
        latencies = []
        total_good = 0
        total_damaged = 0
        total_rotten = 0

        for idx, img_path in enumerate(image_files, 1):
            filename = os.path.basename(img_path)
            start_t = time.time()

            with open(img_path, "rb") as f:
                if "multi" in filename:
                    response = client.post(
                        "/api/inspect/multi",
                        files={"file": (filename, f, "image/jpeg")}
                    )
                else:
                    response = client.post(
                        "/api/inspect/image",
                        files={"file": (filename, f, "image/jpeg")},
                        data={"notes": f"Benchmark {idx}"}
                    )

            latency_ms = round((time.time() - start_t) * 1000, 1)
            latencies.append(latency_ms)

            if response.status_code == 200:
                data = response.json()
                if "multi" in filename:
                    pred_class = f"Batch ({data['total_fruits']} fruits)"
                    conf = data['acceptance_rate']
                    status = data['overall_status']
                    total_good += data['good_count']
                    total_damaged += data['damaged_count']
                    total_rotten += data['rotten_count']
                else:
                    pred_class = data['predicted_class']
                    conf = round(data['confidence'] * 100, 1)
                    status = data['status']
                    if pred_class == "Good": total_good += 1
                    elif pred_class == "Damaged": total_damaged += 1
                    else: total_rotten += 1

                results.append({
                    "index": idx,
                    "file": filename,
                    "latency_ms": latency_ms,
                    "classification": pred_class,
                    "confidence_or_rate": conf,
                    "status": status
                })

                print(f"[{idx:02d}/{len(image_files):02d}] {filename:<28} | {latency_ms:>5.1f}ms | {pred_class:<15} ({conf}%) -> {status}")
            else:
                print(f"[{idx:02d}/{len(image_files):02d}] {filename:<28} | FAILED: HTTP {response.status_code}")

        # Database verification
        db_records_count = db.query(FruitInspection).count()
        avg_latency = round(sum(latencies) / max(1, len(latencies)), 1)
        total_fruits = total_good + total_damaged + total_rotten
        overall_acceptance = round((total_good / max(1, total_fruits)) * 100, 1)

        print("\n" + "=" * 80)
        print(" FRUITVISION BENCHMARK PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f" Images Processed          : {len(image_files)}")
        print(f" Total Fruits Inspected    : {total_fruits}")
        print(f" Good Fruits (Accepted)    : {total_good}")
        print(f" Damaged Fruits (Secondary): {total_damaged}")
        print(f" Rotten Fruits (Rejected)  : {total_rotten}")
        print(f" Overall Acceptance Rate   : {overall_acceptance}%")
        print(f" Average Latency per Image : {avg_latency} ms")
        print(f" Total SQLite DB Records   : {db_records_count}")
        print("=" * 80 + "\n")

        # Save benchmark report
        with open("test_results/benchmark_report.json", "w") as rf:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "images_tested": len(image_files),
                "total_fruits_inspected": total_fruits,
                "good_count": total_good,
                "damaged_count": total_damaged,
                "rotten_count": total_rotten,
                "acceptance_rate": overall_acceptance,
                "avg_latency_ms": avg_latency,
                "db_records": db_records_count,
                "results": results
            }, rf, indent=2)

        print("[FruitVision Benchmark] Benchmark report written to: test_results/benchmark_report.json")

if __name__ == "__main__":
    run_fruitvision_benchmark()

