#!/usr/bin/env python3
"""
evaluate_accuracy.py
--------------------
Compiled .xmodel modelinin DPU uzerinde accuracy olcer.
deploy_img.py ile ayni runner mantigi, test klasoru uzerinde dongu.

Kullanim:
    python3 evaluate_accuracy.py --model resnet50_compiled.xmodel --data data/test
"""

import os
import sys
import argparse
import time
import datetime

import cv2
import numpy as np
import vart
import xir
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# ARGS
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet50_compiled.xmodel",
                    help="Compiled .xmodel dosyasinin yolu")
parser.add_argument("--data",  type=str, default="data/test",
                    help="Test directory (ImageFolder format: class/image.png)")
args = parser.parse_args()

# --------------------------------------------------
# LOG DOSYASI
# --------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = f"reports/image_eval_results_{timestamp}.txt"
log_file  = open(log_path, "w")

def log(msg=""):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# --------------------------------------------------
# MODEL YUKLE
# --------------------------------------------------
if not os.path.exists(args.model):
    print(f"[ERROR] Model not found: {args.model}")
    sys.exit(1)

if not os.path.exists(args.data):
    print(f"[ERROR] Test directory not found: {args.data}")
    sys.exit(1)

print(f">> Loading model: {args.model}")
graph = xir.Graph.deserialize(args.model)
root  = graph.get_root_subgraph()

dpu_subgraph = None
for s in root.get_children():
    if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
        dpu_subgraph = s
        break

if dpu_subgraph is None:
    print("[ERROR] DPU subgraph not found!")
    sys.exit(1)

runner = vart.Runner.create_runner(dpu_subgraph, "run")

input_tensors  = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()
input_dims     = tuple(input_tensors[0].dims)   # [1, H, W, C]
output_dims    = tuple(output_tensors[0].dims)   # [1, num_classes]

H, W = input_dims[1], input_dims[2]
print(f">> Input shape : {input_dims}")
print(f">> Output shape: {output_dims}")

# --------------------------------------------------
# TEST KLASORUNU OKU
# --------------------------------------------------
classes = sorted([
    d for d in os.listdir(args.data)
    if os.path.isdir(os.path.join(args.data, d))
])
print(f">> Classes: {classes}")

image_paths = []
labels      = []

for idx, cls in enumerate(classes):
    cls_dir = os.path.join(args.data, cls)
    files   = [f for f in os.listdir(cls_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    for f in files:
        image_paths.append(os.path.join(cls_dir, f))
        labels.append(idx)

total = len(image_paths)
print(f">> Total test images: {total}")
if total == 0:
    print("[ERROR] No images found in test directory.")
    sys.exit(1)

# --------------------------------------------------
# INFERENCE DONGUSU
# --------------------------------------------------
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (W, H)).astype(np.float32) / 255.0
    # 3 kanali ayni gri degerle doldur (train.py: Grayscale(num_output_channels=3))
    tensor = np.stack([resized, resized, resized], axis=-1)
    return np.expand_dims(tensor, axis=0)  # [1, H, W, 3]

y_true  = []
y_pred  = []
y_probs = []

input_buf  = np.zeros(input_dims,  dtype=np.float32)
output_buf = np.zeros(output_dims, dtype=np.float32)

print("\n" + "="*55)
print(">> Starting inference...\n")
t_start = time.time()

for i, (path, label) in enumerate(zip(image_paths, labels)):
    tensor = preprocess(path)
    if tensor is None:
        print(f"  [WARNING] Could not read, skipping: {path}")
        continue

    np.copyto(input_buf, tensor)
    inp = [np.ascontiguousarray(input_buf)]
    out = [output_buf]

    job_id = runner.execute_async(inp, out)
    runner.wait(job_id)

    probs = softmax(out[0][0])
    pred  = int(np.argmax(probs))

    y_true.append(label)
    y_pred.append(pred)
    y_probs.append(probs[1])  # positive class (sleepy) probability score

    # ilerleme
    if (i + 1) % 50 == 0 or (i + 1) == total:
        elapsed = time.time() - t_start
        print(f"  [{i+1:>4}/{total}]  elapsed: {elapsed:.1f}s  "
              f"running acc: {accuracy_score(y_true, y_pred):.4f}")

t_end = time.time()

print("\n" + "="*55)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
del runner

acc  = accuracy_score(y_true, y_pred)
cm   = confusion_matrix(y_true, y_pred)
rep  = classification_report(y_true, y_pred, target_names=classes, digits=4)

try:
    auc = roc_auc_score(y_true, y_probs)
except Exception:
    auc = float("nan")

log_file.write(f"Date/Time     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

log("\n" + "="*55)
log("  DPU ACCURACY RESULTS")
log("="*55)
log(f"  Model         : {args.model}")
log(f"  Test set      : {args.data}")
log(f"  Total images  : {len(y_true)}")
log(f"  Duration      : {t_end - t_start:.2f} s")
log(f"  Average FPS   : {len(y_true) / (t_end - t_start):.2f}")
log(f"\n  Accuracy      : {acc:.4f}  ({acc*100:.2f}%)")
log(f"  AUC-ROC       : {auc:.4f}")
log(f"\n  Confusion Matrix ({classes[0]}=0, {classes[1]}=1):")
log(f"    {cm}")
log(f"\n  Classification Report:")
log(rep)
log("="*55)
log(f"\n>> Results saved to: {log_path}")
log_file.close()
