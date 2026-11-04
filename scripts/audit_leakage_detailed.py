"""
Detailed Data Leakage and Perceptual Hash Audit
FruitVision AI — Independent Audit
"""
import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from PIL import Image
import numpy as np

def dhash(image, hash_size=8):
    resized = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS).convert('L')
    pixels = np.array(list(resized.getdata()), dtype=np.int32).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def ahash(image, hash_size=8):
    resized = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS).convert('L')
    pixels = np.array(list(resized.getdata()), dtype=np.float32)
    avg = pixels.mean()
    return (pixels > avg).flatten()

def hamming(h1, h2):
    return int(np.count_nonzero(h1 != h2))

def get_timestamp(name):
    m = re.search(r'(?:IMG)?(20\d{6})[_\-]?(\d{6})', name)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
        except:
            return None
    return None

def run_audit():
    root = Path('dataset/processed')
    splits = {}
    sha256_by_split = defaultdict(dict)
    
    for s in ['train', 'val', 'test']:
        splits[s] = []
        files = list((root / s).glob('*/*.*'))
        for f in files:
            with open(f, 'rb') as fp:
                sha = hashlib.sha256(fp.read()).hexdigest()
            sha256_by_split[s][sha] = str(f)
            with Image.open(f) as im:
                im_rgb = im.convert('RGB')
                dh = dhash(im_rgb)
                ah = ahash(im_rgb)
            splits[s].append({
                'path': str(f),
                'filename': f.name,
                'class': f.parent.name,
                'sha256': sha,
                'dhash': dh,
                'ahash': ah,
                'timestamp': get_timestamp(f.name)
            })

    print(f"Loaded samples: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # 1. Exact SHA-256 duplicate check
    exact_leak_train_val = set(sha256_by_split['train'].keys()) & set(sha256_by_split['val'].keys())
    exact_leak_train_test = set(sha256_by_split['train'].keys()) & set(sha256_by_split['test'].keys())
    exact_leak_val_test = set(sha256_by_split['val'].keys()) & set(sha256_by_split['test'].keys())

    print(f"Exact SHA-256 duplicates: Train-Val={len(exact_leak_train_val)}, Train-Test={len(exact_leak_train_test)}, Val-Test={len(exact_leak_val_test)}")

    # 2. Time-proximity cross-split analysis
    time_proximity_train_test = []
    for tr in splits['train']:
        if not tr['timestamp']: continue
        for te in splits['test']:
            if not te['timestamp']: continue
            if tr['class'] != te['class']: continue
            delta = abs((te['timestamp'] - tr['timestamp']).total_seconds())
            if delta <= 120: # <= 2 minutes
                time_proximity_train_test.append({
                    'delta_seconds': delta,
                    'class': tr['class'],
                    'train_file': tr['filename'],
                    'test_file': te['filename']
                })

    time_proximity_train_test.sort(key=lambda x: x['delta_seconds'])
    print(f"Same-class pairs between Train and Test captured within 120 seconds: {len(time_proximity_train_test)}")

    # 3. Perceptual Hash distance between Test and Train
    perceptual_near_train_test = []
    dists = []
    for te in splits['test']:
        min_dist_to_train = 64
        closest_train = None
        for tr in splits['train']:
            d = hamming(te['dhash'], tr['dhash'])
            if d < min_dist_to_train:
                min_dist_to_train = d
                closest_train = tr
            if d <= 4:
                perceptual_near_train_test.append({
                    'hamming_distance': d,
                    'test_class': te['class'],
                    'train_class': tr['class'],
                    'test_file': te['filename'],
                    'train_file': tr['filename'],
                    'same_class': (te['class'] == tr['class'])
                })
        dists.append(min_dist_to_train)

    print(f"Pairs with dHash Hamming distance <= 4 between Train and Test: {len(perceptual_near_train_test)}")
    print(f"Mean min Hamming distance from Test to Train: {np.mean(dists):.2f} bits (min: {np.min(dists)}, max: {np.max(dists)})")

    # 4. Class balance in splits
    class_balance = {}
    for s in ['train', 'val', 'test']:
        counts = defaultdict(int)
        for item in splits[s]:
            counts[item['class']] += 1
        class_balance[s] = dict(counts)

    print("Class Balance:")
    print(json.dumps(class_balance, indent=2))

    # Save findings to json
    results = {
        'total_samples': {s: len(splits[s]) for s in splits},
        'class_balance': class_balance,
        'exact_sha256_duplicates': {
            'train_val': len(exact_leak_train_val),
            'train_test': len(exact_leak_train_test),
            'val_test': len(exact_leak_val_test)
        },
        'burst_leakage_cases_within_120s': time_proximity_train_test[:30],
        'total_pairs_within_120s': len(time_proximity_train_test),
        'perceptual_near_pairs_dhash_le_4': perceptual_near_train_test[:30],
        'total_perceptual_near_le_4': len(perceptual_near_train_test),
        'min_distance_distribution': {
            'mean': float(np.mean(dists)),
            'median': float(np.median(dists)),
            'min': int(np.min(dists)),
            'max': int(np.max(dists)),
            'fraction_with_dist_le_2': float(np.mean(np.array(dists) <= 2)),
            'fraction_with_dist_le_4': float(np.mean(np.array(dists) <= 4))
        }
    }
    
    with open('reports/audit_leakage_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Audit results saved to reports/audit_leakage_results.json")

if __name__ == '__main__':
    run_audit()

