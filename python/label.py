"""
label.py — Step 1 of the CSI pose pipeline
==========================================
Runs YOLOv8-pose on every camera frame in a session.zip exported from
Chrono Vision, applies rule-based posture classification, and joins
the labels to the CSI rows by timestamp.

Usage:
    python label.py session.zip dataset.csv

Output:
    dataset.csv   — timestamp_ms, label, SC0…SCn (ready for train.py)
    label_stats.txt — class distribution summary

Requirements (pip install):
    ultralytics pillow numpy
"""

import sys, os, zipfile, csv, json, io
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── COCO keypoint indices ─────────────────────────────────────────
NOSE    = 0
L_SHOU  = 5;  R_SHOU  = 6
L_HIP   = 11; R_HIP   = 12
L_KNEE  = 13; R_KNEE  = 14
L_ANKLE = 15; R_ANKLE = 16

CONF_THRESH = 0.3   # minimum keypoint confidence to trust

# ── Rule-based classifier ─────────────────────────────────────────
def classify(kpts, prev_ankles):
    """
    kpts      : np.ndarray shape (17, 3) — x, y, conf
    prev_ankles: (l_xy, r_xy) or None from the previous frame

    Returns (label_str, curr_ankles)
    """
    def pt(idx):
        k = kpts[idx]
        return (float(k[0]), float(k[1])) if float(k[2]) > CONF_THRESH else None

    l_sho  = pt(L_SHOU);  r_sho  = pt(R_SHOU)
    l_hip  = pt(L_HIP);   r_hip  = pt(R_HIP)
    l_knee = pt(L_KNEE);  r_knee = pt(R_KNEE)
    l_ank  = pt(L_ANKLE); r_ank  = pt(R_ANKLE)

    curr_ankles = None
    if l_ank and r_ank:
        curr_ankles = (l_ank, r_ank)

    # --- Fall: torso more horizontal than vertical ---
    sho = l_sho or r_sho
    hip = l_hip or r_hip
    if sho and hip:
        dx = abs(hip[0] - sho[0])
        dy = abs(hip[1] - sho[1])
        if dx > dy * 1.2:          # torso tilted past ~50° from vertical
            return 'falling', curr_ankles

    # --- Sitting: knee and hip at similar height, small ankle–knee gap ---
    if l_hip and l_knee and l_ank:
        hip_knee_dy  = l_knee[1] - l_hip[1]   # positive = knee below hip
        knee_ank_dy  = l_ank[1]  - l_knee[1]  # positive = ankle below knee
        if hip_knee_dy > 10 and knee_ank_dy < hip_knee_dy * 0.6:
            return 'sitting', curr_ankles

    # --- Walking: cumulative ankle displacement between frames ---
    if prev_ankles and curr_ankles:
        dl = np.linalg.norm(np.array(curr_ankles[0]) - np.array(prev_ankles[0]))
        dr = np.linalg.norm(np.array(curr_ankles[1]) - np.array(prev_ankles[1]))
        if dl + dr > 18:           # ~18 px movement in 150 ms = walking pace
            return 'walking', curr_ankles

    return 'standing', curr_ankles


# ── Main ─────────────────────────────────────────────────────────
def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else 'session.zip'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'dataset.csv'

    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading YOLOv8-pose (downloads ~6 MB on first run)…")
    model = YOLO('yolov8n-pose.pt')

    csi_data   = {}   # timestamp_ms -> [SC0…SCn]
    frame_list = []   # [(frame_idx, timestamp_ms, filename)]

    with zipfile.ZipFile(zip_path, 'r') as z:
        # ── Load CSI rows ──────────────────────────────────────────
        with z.open('csi.csv') as f:
            reader = csv.DictReader(f.read().decode('utf-8').splitlines())
            for row in reader:
                ts   = int(row['timestamp_ms'])
                sc_keys = sorted([k for k in row if k.startswith('SC')],
                                 key=lambda k: int(k[2:]))
                amps = [float(row[k]) for k in sc_keys]
                csi_data[ts] = amps

        # ── Collect frame filenames ────────────────────────────────
        for name in sorted(z.namelist()):
            if not (name.startswith('frames/') and name.endswith('.jpg')):
                continue
            stem   = Path(name).stem          # frame_0000_12345
            parts  = stem.split('_')
            if len(parts) < 3:
                continue
            idx = int(parts[1])
            ts  = int(parts[2])
            frame_list.append((idx, ts, name))

    print(f"  {len(csi_data)} CSI rows, {len(frame_list)} camera frames")

    if not frame_list:
        print("WARNING: no camera frames found — only CSI rows labelled as 'unknown'.")
        rows = [{'timestamp_ms': ts, 'label': 'unknown', 'amps': a}
                for ts, a in csi_data.items()]
        _write_csv(rows, out_path)
        return

    # ── Process frames ────────────────────────────────────────────
    rows        = []
    prev_ankles = None

    with zipfile.ZipFile(zip_path, 'r') as z:
        for (idx, ts, fname) in frame_list:
            img_bytes = z.read(fname)
            img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_arr   = np.array(img)

            results = model(img_arr, verbose=False)

            label = 'empty'
            if results and len(results[0].keypoints.data) > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
                label, prev_ankles = classify(kpts, prev_ankles)
            else:
                prev_ankles = None

            # Match to nearest CSI timestamp (within 2 ticks = 300 ms)
            if csi_data:
                closest = min(csi_data.keys(), key=lambda t: abs(t - ts))
                if abs(closest - ts) <= 300:
                    rows.append({'timestamp_ms': ts,
                                 'label':        label,
                                 'amps':         csi_data[closest]})

            print(f"  frame {idx:04d} @ {ts} ms → {label}")

    _write_csv(rows, out_path)

    # ── Summary ───────────────────────────────────────────────────
    counts = {}
    for r in rows:
        counts[r['label']] = counts.get(r['label'], 0) + 1
    print(f"\nDone — {len(rows)} labeled rows → {out_path}")
    for label, n in sorted(counts.items()):
        print(f"  {label:12s}: {n:5d} ({n/len(rows)*100:.1f} %)")

    with open('label_stats.txt', 'w') as f:
        f.write(f"Total: {len(rows)}\n")
        for label, n in sorted(counts.items()):
            f.write(f"{label}: {n} ({n/len(rows)*100:.1f}%)\n")
    print("Label distribution saved to label_stats.txt")


def _write_csv(rows, out_path):
    if not rows:
        print("WARNING: no rows to write.", file=sys.stderr)
        return
    n_sc       = len(rows[0]['amps'])
    fieldnames = ['timestamp_ms', 'label'] + [f'SC{i}' for i in range(n_sc)]
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            r = {'timestamp_ms': row['timestamp_ms'], 'label': row['label']}
            for i, v in enumerate(row['amps']):
                r[f'SC{i}'] = f'{v:.4f}'
            w.writerow(r)


if __name__ == '__main__':
    main()
