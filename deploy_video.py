#!/usr/bin/env python3
import os
import cv2
import numpy as np
import vart
import xir
import sys
import time
from collections import deque

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "resnet50_compiled.xmodel"
CLASSES = ["AWAKE", "SLEEPY"]
DISPLAY = True  # HDMI monitor bagliysa True yap

# ---------------------------
# DPU LOAD
# ---------------------------
print(">> Model yukleniyor...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bulunamadi: {MODEL_PATH}")

graph = xir.Graph.deserialize(MODEL_PATH)
root = graph.get_root_subgraph()

dpu_subgraph = None
for s in root.get_children():
    if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
        dpu_subgraph = s
        break

if dpu_subgraph is None:
    raise Exception("DPU subgraph bulunamadi!")

runner = vart.Runner.create_runner(dpu_subgraph, "run")

input_tensors  = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

input_dims  = input_tensors[0].dims
output_dims = output_tensors[0].dims

H, W = input_dims[1], input_dims[2]
print(f">> Input: {input_dims}  Output: {output_dims}")

# Buffer reuse - loop disinda bir kere olustur
input_data  = np.zeros(input_dims,  dtype=np.float32)
output_data = np.zeros(output_dims, dtype=np.float32)

# Smoothing
history = deque(maxlen=3)

# ---------------------------
# CAMERA
# ---------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Kamera acilamadi")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(">> Basladi. Durdurmak icin Ctrl+C")

# ---------------------------
# HELPERS
# ---------------------------
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ---------------------------
# LOOP
# ---------------------------
prev_time   = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img  = cv2.resize(gray, (W, H))
        img  = img.astype(np.float32) / 255.0

        # 3 kanali ayni gri degerle doldur (train.py: Grayscale(num_output_channels=3))
        input_data[0, :, :, 0] = img
        input_data[0, :, :, 1] = img
        input_data[0, :, :, 2] = img

        inp = [np.ascontiguousarray(input_data)]
        out = [output_data]

        jid = runner.execute_async(inp, out)
        runner.wait(jid)

        probs      = softmax(out[0][0])
        idx        = int(np.argmax(probs))
        history.append(idx)
        stable_idx = max(set(history), key=history.count)
        label      = CLASSES[stable_idx]
        score      = probs[stable_idx] * 100

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        frame_count += 1

        print(f"\r[{frame_count:06d}]  {label:<8}  conf={score:5.1f}%  FPS={fps:5.1f}",
              end="", flush=True)

        if DISPLAY:
            color = (0, 255, 0) if label == "AWAKE" else (0, 0, 255)
            cv2.putText(frame, f"{label} {score:.1f}%", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("KV260 Drowsiness", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\n>> Ctrl+C alindi.")

# ---------------------------
# CLEANUP
# ---------------------------
cap.release()
if DISPLAY:
    cv2.destroyAllWindows()
del runner
print(">> Kapatildi.")
