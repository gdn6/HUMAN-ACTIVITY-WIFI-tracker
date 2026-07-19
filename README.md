# Human Activity WiFi Tracker

Real-time human pose estimation using Wi-Fi CSI (Channel State Information) on two ESP32-S3 Genesis Mini boards, with synchronized camera capture for training data collection.

## How it works

1. **Transmitter board** broadcasts ESP-NOW packets at 50 Hz
2. **Receiver board** captures WiFi CSI, extracts 64-subcarrier amplitudes, streams over serial at ~6.7 Hz
3. **Chrono Vision tab** records synchronized CSI + camera frames (chronophotography + slitscan + waterfall)
4. **label.py** runs YOLOv8-pose on camera frames → rule-based posture labels → `dataset.csv`
5. **train.py** trains a 2-layer LSTM on 3-second CSI windows → exports `model.onnx`
6. **Inference tab** loads `model.onnx`, reads live CSI, predicts pose in real time — no camera needed

## Pose classes
`standing` · `sitting` · `walking` · `falling` · `empty`

## Hardware
- 2× Axiometa Genesis Mini (ESP32-S3)
- Flash `firmware/sketch.ino` with `BOARD_ROLE 0` → Transmitter
- Flash with `BOARD_ROLE 1` → Receiver (keep connected to PC)

## Data collection (10 min session plan)
| Time | Activity |
|------|----------|
| 0–2 min | Room empty |
| 2–4 min | Walking back and forth |
| 4–6 min | Standing in various positions |
| 6–7.5 min | Sitting / standing repeatedly |
| 7.5–10 min | Simulated falls |

Open **Chrono Vision** in Axiometa Studio → Enable Camera → **Start Recording** → perform poses → **Stop** (auto-exports `session_<timestamp>.zip`).

## Pipeline
```bash
# 1. Label
pip install ultralytics pillow numpy
python python/label.py session_<timestamp>.zip dataset.csv

# 2. Train
pip install torch numpy scikit-learn
python python/train.py dataset.csv model.onnx

# 3. Infer
# Open Inference tab in Axiometa Studio
# Load model.onnx + model_meta.json
# Connect receiver board → live pose prediction
```

## Browser tabs (run in Axiometa Studio)
| Tab | Purpose |
|-----|---------|
| CSI Monitor | Live subcarrier bar chart, waterfall, motion meter |
| Chrono Vision | Recording: chronophotography + slitscan + CSI waterfall + auto-export |
| Inference | Live ONNX pose prediction from receiver board |
