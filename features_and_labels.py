# features_and_labels.py

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import zstandard as zstd

# ---------- Low-level helpers ----------

def file_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = freq[freq > 0] / len(data)
    H = -np.sum(p * np.log2(p))
    # JS wale jaisa normalize /8
    return float(H / 8.0)

def detect_file_type_flags(data: bytes):
    """ JPEG, PNG, GIF, PDF, ZIP flags order me return karo """
    h = data[:16]
    def starts_with(sig: bytes) -> bool:
        return h.startswith(sig)

    is_jpeg = starts_with(b'\xFF\xD8')
    is_png  = starts_with(b'\x89PNG')
    is_gif  = starts_with(b'GIF')
    is_pdf  = starts_with(b'%PDF')
    is_zip  = starts_with(b'PK')

    return [
        1 if is_jpeg else 0,
        1 if is_png else 0,
        1 if is_gif else 0,
        1 if is_pdf else 0,
        1 if is_zip else 0,
    ]

def size_bins(num_bytes: int):
    size_mb = num_bytes / (1024 * 1024)
    return [
        1 if size_mb < 1 else 0,
        1 if 1 <= size_mb < 10 else 0,
        1 if size_mb >= 10 else 0,
    ]

def compute_hist16(data: bytes):
    if not data:
        return [0.0] * 16
    arr = np.frombuffer(data, dtype=np.uint8)
    # 0..255 ko 16 buckets me compress -> 0..15
    bins = arr // 16
    hist = np.bincount(bins, minlength=16).astype(np.float64)
    s = hist.sum()
    if s == 0:
        return [0.0] * 16
    return (hist / s).tolist()

def compressibility_ratio(data: bytes) -> float:
    """
    Extra feature use kar sakte ho later:
    compressed_size / original_size using zstd.
    Abhi label calculation me thoda use karunga.
    """
    if not data:
        return 1.0
    cctx = zstd.ZstdCompressor(level=5)
    comp = cctx.compress(data)
    return len(comp) / len(data)


# ---------- Feature extractor (26D, JS compatible) ----------

def extract_features_for_file(path: str):
    with open(path, 'rb') as f:
        data = f.read()

    size = len(data)
    # 4KB + 64KB windows
    sample4k = data[: min(4096, size)]
    sample64k = data[: min(65536, size)]

    entropy4k = file_entropy(sample4k)
    entropy64k = file_entropy(sample64k)
    hist16 = compute_hist16(sample64k)
    flags = detect_file_type_flags(data)
    sbins = size_bins(size)

    # EXACT ORDER (same as JS):
    features = []
    features.append(entropy4k)
    features.append(entropy64k)
    features.extend(hist16)
    features.extend(flags)
    features.extend(sbins)

    assert len(features) == 26, f"Feature length mismatch: {len(features)}"
    return features, data


# ---------- Heuristic label generator (Phase-1) ----------

def heuristic_labels(features, data: bytes):
    """
    Yahan se hum ML ke training targets banayenge.
    Later tum isko aur smart bana sakte ho by simulations.
    """

    entropy4k = features[0]
    entropy64k = features[1]
    size_bytes = len(data)
    size_mb = size_bytes / (1024 * 1024)

    # Thoda idea: low entropy ya text/pdf -> zyada bits remove kar sakte;
    # already random data (zip, encrypted) -> bahut kam.
    comp_ratio = compressibility_ratio(data)

    # ---- Corruption ratio target (0.001 .. 0.01) ----
    # comp_ratio ~ 1.0 => high-entropy; comp_ratio small => highly compressible
    # map: if comp_ratio < 0.4 => up to 0.01; if > 0.9 => near 0.002

    base_min = 0.001
    base_max = 0.01

    # invert compressibility: more compressible -> more removable
    comp_score = max(0.0, min(1.0, (1.0 - comp_ratio)))  # 0..1
    target_ratio = base_min + comp_score * (base_max - base_min)

    # size adjustment: very small file -> slightly lower; very large -> slightly lower for safety
    if size_mb < 1:
        target_ratio *= 0.7
    elif size_mb > 100:
        target_ratio *= 0.8

    # clamp
    target_ratio = float(max(0.0005, min(0.012, target_ratio)))

    # ---- Partition count target (3 .. some upper bound) ----
    # Intuition: bigger file + more complex (entropy) => more partitions.
    # Basic formula:
    base_parts = 3
    extra_by_size = int(min(100, size_mb * 2))   # 1MB -> +2, 50MB -> +100 (cap)
    extra_by_entropy = int((entropy64k - 0.3) * 50)  # entropy around 0.9 -> +30ish

    parts = base_parts + max(0, extra_by_size) + max(0, extra_by_entropy)
    # Hard clamp so abhi astronaut numbers na ho:
    parts = int(max(3, min(500, parts)))

    # ---- Pattern type (0=SCATTERED, 1=CLUSTERED, 2=HEADER-HEAVY) ----
    flags = features[18:23]
    is_zip_like = flags[4] == 1
    is_img = flags[0] == 1 or flags[1] == 1 or flags[2] == 1
    is_pdf = flags[3] == 1

    if is_zip_like:
        pattern = 1   # clustered (structure important)
    elif is_img:
        pattern = 0   # scattered
    elif is_pdf:
        pattern = 2   # header-heavy (bohot metadata)
    else:
        # entropy based fallback
        if entropy64k > 0.9:
            pattern = 0
        else:
            pattern = 1

    # ---- Base set choice target (0..N-1) ----
    # For now 3 presets:
    # 0: [12,16,20,36] balanced
    # 1: [16,20,36] more compact
    # 2: [12,16] more verbose (harder brute-force)
    if comp_ratio > 0.9:
        base_set_id = 2
    elif comp_ratio < 0.5:
        base_set_id = 1
    else:
        base_set_id = 0

    return {
        "target_ratio": target_ratio,
        "target_parts": parts,
        "target_pattern": pattern,
        "target_base_set": base_set_id,
        "size_mb": size_mb,
        "comp_ratio": comp_ratio,
        "entropy4k": entropy4k,
        "entropy64k": entropy64k,
    }


# ---------- Main dataset builder ----------

def build_dataset(folder: str, out_csv: str = "dataset.csv"):
    rows = []

    all_files = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            all_files.append(path)

    print(f"Found {len(all_files)} files")

    for path in tqdm(all_files):
        try:
            feats, data = extract_features_for_file(path)
            labels = heuristic_labels(feats, data)

            row = {
                "path": path,
            }
            # 26 features
            for i, v in enumerate(feats):
                row[f"f_{i:02d}"] = v
            # labels
            row.update(labels)
            rows.append(row)
        except Exception as e:
            print("Error processing", path, ":", e)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved dataset to", out_csv)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing training files")
    ap.add_argument("--out", default="dataset.csv", help="Output CSV path")
    args = ap.parse_args()

    build_dataset(args.folder, args.out)
