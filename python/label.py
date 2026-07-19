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

NOSE=0; L_SHOU=5; R_SHOU=6; L_HIP=11; R_HIP=12
L_KNEE=13; R_KNEE=14; L_ANKLE=15; R_ANKLE=16
CONF_THRESH=0.3

def classify(kpts, prev_ankles):
    def pt(idx):
        k=kpts[idx]; return (float(k[0]),float(k[1])) if float(k[2])>CONF_THRESH else None
    l_sho=pt(L_SHOU); r_sho=pt(R_SHOU)
    l_hip=pt(L_HIP);  r_hip=pt(R_HIP)
    l_knee=pt(L_KNEE); r_knee=pt(R_KNEE)
    l_ank=pt(L_ANKLE); r_ank=pt(R_ANKLE)
    curr_ankles=None
    if l_ank and r_ank: curr_ankles=(l_ank,r_ank)
    sho=l_sho or r_sho; hip=l_hip or r_hip
    if sho and hip:
        dx=abs(hip[0]-sho[0]); dy=abs(hip[1]-sho[1])
        if dx>dy*1.2: return 'falling',curr_ankles
    if l_hip and l_knee and l_ank:
        hip_knee_dy=l_knee[1]-l_hip[1]; knee_ank_dy=l_ank[1]-l_knee[1]
        if hip_knee_dy>10 and knee_ank_dy<hip_knee_dy*0.6: return 'sitting',curr_ankles
    if prev_ankles and curr_ankles:
        dl=np.linalg.norm(np.array(curr_ankles[0])-np.array(prev_ankles[0]))
        dr=np.linalg.norm(np.array(curr_ankles[1])-np.array(prev_ankles[1]))
        if dl+dr>18: return 'walking',curr_ankles
    return 'standing',curr_ankles

def main():
    zip_path=sys.argv[1] if len(sys.argv)>1 else 'session.zip'
    out_path=sys.argv[2] if len(sys.argv)>2 else 'dataset.csv'
    if not os.path.exists(zip_path): print(f"ERROR: {zip_path} not found.",file=sys.stderr); sys.exit(1)
    print("Loading YOLOv8-pose…")
    model=YOLO('yolov8n-pose.pt')
    csi_data={}; frame_list=[]
    with zipfile.ZipFile(zip_path,'r') as z:
        with z.open('csi.csv') as f:
            reader=csv.DictReader(f.read().decode('utf-8').splitlines())
            for row in reader:
                ts=int(row['timestamp_ms'])
                sc_keys=sorted([k for k in row if k.startswith('SC')],key=lambda k:int(k[2:]))
                csi_data[ts]=[float(row[k]) for k in sc_keys]
        for name in sorted(z.namelist()):
            if not (name.startswith('frames/') and name.endswith('.jpg')): continue
            stem=Path(name).stem; parts=stem.split('_')
            if len(parts)<3: continue
            frame_list.append((int(parts[1]),int(parts[2]),name))
    print(f"  {len(csi_data)} CSI rows, {len(frame_list)} camera frames")
    rows=[]; prev_ankles=None
    with zipfile.ZipFile(zip_path,'r') as z:
        for (idx,ts,fname) in frame_list:
            img=Image.open(io.BytesIO(z.read(fname))).convert('RGB')
            results=model(np.array(img),verbose=False)
            label='empty'
            if results and len(results[0].keypoints.data)>0:
                kpts=results[0].keypoints.data[0].cpu().numpy()
                label,prev_ankles=classify(kpts,prev_ankles)
            else: prev_ankles=None
            if csi_data:
                closest=min(csi_data.keys(),key=lambda t:abs(t-ts))
                if abs(closest-ts)<=300:
                    rows.append({'timestamp_ms':ts,'label':label,'amps':csi_data[closest]})
            print(f"  frame {idx:04d} @ {ts} ms → {label}")
    _write_csv(rows,out_path)
    counts={}
    for r in rows: counts[r['label']]=counts.get(r['label'],0)+1
    print(f"\nDone — {len(rows)} labeled rows → {out_path}")
    for label,n in sorted(counts.items()): print(f"  {label:12s}: {n:5d} ({n/len(rows)*100:.1f}%)")
    with open('label_stats.txt','w') as f:
        f.write(f"Total: {len(rows)}\n")
        for label,n in sorted(counts.items()): f.write(f"{label}: {n} ({n/len(rows)*100:.1f}%)\n")
    print("Label distribution saved to label_stats.txt")

def _write_csv(rows,out_path):
    if not rows: print("WARNING: no rows to write.",file=sys.stderr); return
    n_sc=len(rows[0]['amps'])
    fieldnames=['timestamp_ms','label']+[f'SC{i}' for i in range(n_sc)]
    with open(out_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        for row in rows:
            r={'timestamp_ms':row['timestamp_ms'],'label':row['label']}
            for i,v in enumerate(row['amps']): r[f'SC{i}']=f'{v:.4f}'
            w.writerow(r)

if __name__=='__main__': main()
