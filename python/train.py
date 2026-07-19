"""
train.py — Step 2 of the CSI pose pipeline
==========================================
Loads dataset.csv (output of label.py), builds 20-frame sliding windows
over the CSI amplitude sequences, trains a two-layer LSTM classifier,
and exports the model as model.onnx + model_meta.json for the browser
Inference tab.

Usage:
    python train.py dataset.csv model.onnx

Output:
    model.onnx        — ONNX model (input: [batch, 20, n_sc])
    model_meta.json   — labels, window size, normalization stats

Requirements (pip install):
    torch torchvision numpy scikit-learn
"""

import sys, os, csv, json
import numpy as np

WINDOW  = 20          # frames per sequence (20 × 150 ms = 3 s)
LABELS  = ['standing', 'sitting', 'walking', 'falling', 'empty']
L2I     = {l: i for i, l in enumerate(LABELS)}
EPOCHS  = 60
BATCH   = 32
LR      = 1e-3
DROPOUT = 0.3
HIDDEN  = 128
LAYERS  = 2


# ── Load & window ────────────────────────────────────────────────
def load_windows(path):
    raw = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc_keys = sorted([k for k in row if k.startswith('SC')],
                             key=lambda k: int(k[2:]))
            amps  = [float(row[k]) for k in sc_keys]
            label = row.get('label', 'empty')
            if label not in L2I:
                continue
            raw.append((amps, L2I[label]))

    if len(raw) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} labeled rows, got {len(raw)}")

    X, y = [], []
    for i in range(len(raw) - WINDOW + 1):
        window = raw[i: i + WINDOW]
        X.append([r[0] for r in window])
        y.append(window[-1][1])            # label of the last frame in window

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Model ────────────────────────────────────────────────────────
def build_model(n_feat, n_cls):
    import torch.nn as nn
    class PoseLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, HIDDEN, LAYERS,
                                batch_first=True, dropout=DROPOUT)
            self.norm = nn.LayerNorm(HIDDEN)
            self.fc   = nn.Linear(HIDDEN, n_cls)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.norm(out[:, -1, :]))
    return PoseLSTM()


# ── Train ────────────────────────────────────────────────────────
def train(model, train_dl, val_dl, device):
    import torch, torch.nn as nn
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS)
    crit  = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
        sched.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(1)
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            acc = correct / total if total else 0
            best_acc = max(best_acc, acc)
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/len(train_dl):.4f}  val_acc={acc:.1%}")

    print(f"Best val accuracy: {best_acc:.1%}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'
    out_path     = sys.argv[2] if len(sys.argv) > 2 else 'model.onnx'

    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {dataset_path}…")
    X, y = load_windows(dataset_path)
    n_sc = X.shape[2]
    print(f"  {len(X)} windows · {n_sc} subcarriers · {len(LABELS)} classes")

    # ── Class balance check ───────────────────────────────────────
    counts = np.bincount(y, minlength=len(LABELS))
    for i, l in enumerate(LABELS):
        print(f"  {l:12s}: {counts[i]:5d} samples")
    if counts.min() < 20:
        print("WARNING: some classes have very few samples — collect more data for those poses.")

    # ── Normalize (per subcarrier, across all windows) ────────────
    flat = X.reshape(-1, n_sc)
    mu   = flat.mean(axis=0, keepdims=True)   # (1, n_sc)
    sig  = flat.std(axis=0,  keepdims=True) + 1e-8
    X    = (X - mu) / sig

    # ── Train/val split ───────────────────────────────────────────
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    ds     = TensorDataset(torch.tensor(X), torch.tensor(y))
    val_n  = max(1, int(len(ds) * 0.2))
    tr_n   = len(ds) - val_n
    tr_ds, val_ds = random_split(ds, [tr_n, val_n],
                                 generator=torch.Generator().manual_seed(42))
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True,  drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64,    shuffle=False)

    # ── Build & train ─────────────────────────────────────────────
    model = build_model(n_sc, len(LABELS)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    train(model, tr_dl, val_dl, device)

    # ── Export ONNX ───────────────────────────────────────────────
    model.eval().cpu()
    dummy = torch.zeros(1, WINDOW, n_sc)
    torch.onnx.export(
        model, dummy, out_path,
        input_names  = ['csi_window'],
        output_names = ['logits'],
        dynamic_axes = {'csi_window': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version = 12
    )
    print(f"Saved {out_path}")

    # ── Save metadata for inference tab ──────────────────────────
    meta = {
        'labels':     LABELS,
        'window':     WINDOW,
        'n_features': n_sc,
        'mu':         mu[0].tolist(),
        'sigma':      sig[0].tolist()
    }
    meta_path = out_path.replace('.onnx', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")
    print("\nNext step:")
    print(f"  Load {out_path} and {meta_path} in the Inference tab.")


if __name__ == '__main__':
    main()
