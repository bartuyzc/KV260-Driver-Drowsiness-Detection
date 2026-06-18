#!/usr/bin/env python3
"""
video_evaluate_accuracy.py
--------------------------
Measures real system FPS by reading live from webcam.
Also computes accuracy by collecting ground truth labels.

Usage - FPS only (no labels):
    python3 video_evaluate_accuracy.py --model resnet50_compiled.xmodel

Usage - accuracy mode (first AWAKE then SLEEPY):
    python3 video_evaluate_accuracy.py --model resnet50_compiled.xmodel \
        --awake-frames 100 --sleepy-frames 100

Output:
    - Real system FPS (webcam + preprocessing + DPU + display)
    - DPU-only FPS
    - Accuracy / confusion matrix (if running in label mode)
"""

import argparse
import time
import collections
import datetime

import cv2
import numpy as np
import vart
import xir

# --------------------------------------------------
# ARGS
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str, default="resnet50_compiled.xmodel")
parser.add_argument("--camera",       type=int, default=0,
                    help="Webcam index (default 0)")
parser.add_argument("--awake-frames", type=int, default=0,
                    help="Number of AWAKE (label=0) frames to collect. 0=FPS only mode")
parser.add_argument("--sleepy-frames",type=int, default=0,
                    help="Number of SLEEPY (label=1) frames to collect.")
parser.add_argument("--warmup",       type=int, default=30,
                    help="Number of warmup frames to skip (default 30)")
args = parser.parse_args()

LABEL_MODE = (args.awake_frames > 0 or args.sleepy_frames > 0)

# --------------------------------------------------
# LOG DOSYASI
# --------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = f"reports/video_eval_results_{timestamp}.txt"
log_file  = open(log_path, "w")

def log(msg=""):
    """Writes to both terminal and log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print(f">> Loading model: {args.model}")
graph       = xir.Graph.deserialize(args.model)
root        = graph.get_root_subgraph()
dpu_sg      = None
for s in root.get_children():
    if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
        dpu_sg = s
        break

if dpu_sg is None:
    raise RuntimeError("DPU subgraph not found!")

runner = vart.Runner.create_runner(dpu_sg, "run")
in_t   = runner.get_input_tensors()
out_t  = runner.get_output_tensors()
in_dim = tuple(in_t[0].dims)    # [1, H, W, C]
out_dim= tuple(out_t[0].dims)   # [1, 2]
H, W   = in_dim[1], in_dim[2]
print(f">> Input: {in_dim}  Output: {out_dim}")

in_buf  = np.zeros(in_dim,  dtype=np.float32)
out_buf = np.zeros(out_dim, dtype=np.float32)

# --------------------------------------------------
# OPEN CAMERA
# --------------------------------------------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera (index={args.camera})")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f">> Camera opened (index={args.camera})")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
CLASSES = ["Awake", "Sleepy"]

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def preprocess(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (W, H)).astype(np.float32) / 255.0
    tensor  = np.stack([resized, resized, resized], axis=-1)
    return np.expand_dims(tensor, axis=0)

def infer(tensor):
    np.copyto(in_buf, tensor)
    job = runner.execute_async([np.ascontiguousarray(in_buf)], [out_buf])
    runner.wait(job)
    return softmax(out_buf[0])

# --------------------------------------------------
# STATE MACHINE
# --------------------------------------------------
# LABEL_MODE: collect awake_frames first, then sleepy_frames
# FPS_MODE  : run until 'q' is pressed

y_true      = []
y_pred      = []
frame_count = 0
warmup_done = False

# FPS measurement
fps_deque    = collections.deque(maxlen=30)   # last 30 frames
dpu_times    = []
total_times  = []

if LABEL_MODE:
    phase_plan = []
    if args.awake_frames  > 0: phase_plan.append((0, args.awake_frames,  "AWAKE"))
    if args.sleepy_frames > 0: phase_plan.append((1, args.sleepy_frames, "SLEEPY"))
    phase_idx       = 0
    phase_collected = 0
    current_label, phase_quota, phase_name = phase_plan[phase_idx]
    waiting_for_space = True   # SPACE basilana kadar toplamaya basma
    print(f"\n>> LABEL MODE: Collecting {args.awake_frames} AWAKE frames, then {args.sleepy_frames} SLEEPY frames.")
    print(f">> First {args.warmup} frames will be skipped for warmup.")
    print(f">> Once the camera opens, get into AWAKE position and press SPACE when ready.")
else:
    print(f"\n>> FPS MODE: Press 'q' to quit.")
    print(f">> First {args.warmup} frames will be skipped for warmup.")

print(">> Starting...\n")

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
try:
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Could not read frame, skipping...")
            continue

        # --- Preprocess ---
        tensor = preprocess(frame)

        # --- DPU Inference ---
        t_dpu_start = time.time()
        probs = infer(tensor)
        t_dpu_end   = time.time()

        pred       = int(np.argmax(probs))
        confidence = float(probs[pred])
        dpu_ms     = (t_dpu_end - t_dpu_start) * 1000

        frame_count += 1
        if frame_count > args.warmup:
            warmup_done = True
            dpu_times.append(dpu_ms)

        t1 = time.time()
        total_ms = (t1 - t0) * 1000
        fps_deque.append(1000.0 / total_ms if total_ms > 0 else 0)
        if warmup_done:
            total_times.append(total_ms)

        # --- Label mode: collect data ---
        if LABEL_MODE and warmup_done and not waiting_for_space:
            y_true.append(current_label)
            y_pred.append(pred)
            phase_collected += 1

            # Move to next phase
            if phase_collected >= phase_quota:
                phase_idx += 1
                if phase_idx >= len(phase_plan):
                    print(f"\n>> All {len(y_true)} frames collected. Computing results...")
                    log(f">> All {len(y_true)} frames collected.")
                    break
                current_label, phase_quota, phase_name = phase_plan[phase_idx]
                phase_collected = 0
                frame_count = 0
                waiting_for_space = True
                print(f"\n>> {phase_name} position, press SPACE when ready.")

        # --- Display ---
        live_fps = np.mean(fps_deque) if fps_deque else 0
        color    = (0, 255, 0) if pred == 0 else (0, 0, 255)
        label_str = f"{CLASSES[pred]} {confidence*100:.1f}%"

        # background box
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, label_str,
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"System FPS: {live_fps:.1f}  |  DPU: {dpu_ms:.1f}ms",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if LABEL_MODE and warmup_done:
            if waiting_for_space:
                msg = f">> {phase_name} position ready, press SPACE"
                cv2.rectangle(frame, (0, frame.shape[0]-40), (640, frame.shape[0]), (50,50,0), -1)
                cv2.putText(frame, msg, (10, frame.shape[0]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            else:
                prog = f"Collected: {len(y_true)} / {args.awake_frames + args.sleepy_frames}  ({phase_name}: {phase_collected}/{phase_quota})"
                cv2.rectangle(frame, (0, frame.shape[0]-40), (640, frame.shape[0]), (0,50,0), -1)
                cv2.putText(frame, prog, (10, frame.shape[0]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

        cv2.imshow("Drowsiness Evaluator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and LABEL_MODE and waiting_for_space:
            waiting_for_space = False
            frame_count = 0
            print(f">> {phase_name} collection started ({phase_quota} frames)...")

except KeyboardInterrupt:
    print("\n>> Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    del runner

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
log_file.write(f"\nDate/Time  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"Model      : {args.model}\n")
log_file.write(f"Awake      : {args.awake_frames} frames  |  Sleepy: {args.sleepy_frames} frames\n")

log("\n" + "="*55)
log("  VIDEO EVALUATE ACCURACY RESULTS")
log("="*55)

if total_times:
    avg_sys_fps = 1000.0 / np.mean(total_times)
    avg_dpu_ms  = np.mean(dpu_times)
    avg_dpu_fps = 1000.0 / avg_dpu_ms

    log(f"  Total frames (excl. warmup) : {len(total_times)}")
    log(f"\n  [SYSTEM - Real End-to-End]")
    log(f"  Avg frame duration          : {np.mean(total_times):.2f} ms")
    log(f"  Avg system FPS              : {avg_sys_fps:.2f}")
    log(f"  Min / Max system FPS        : {1000/max(total_times):.2f} / {1000/min(total_times):.2f}")
    log(f"\n  [DPU - Inference Only]")
    log(f"  Avg DPU duration            : {avg_dpu_ms:.3f} ms")
    log(f"  Avg DPU FPS                 : {avg_dpu_fps:.2f}")
    log(f"  Min / Max DPU ms            : {min(dpu_times):.3f} / {max(dpu_times):.3f}")

if LABEL_MODE and len(y_true) > 0:
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, roc_auc_score
    )
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

    log(f"\n  [ACCURACY]")
    log(f"  Total samples               : {len(y_true)}")
    log(f"  Accuracy                    : {acc:.4f}  ({acc*100:.2f}%)")
    log(f"\n  Confusion Matrix (Awake=0, Sleepy=1):")
    log(f"    {cm}")
    log(f"\n  Classification Report:")
    log(rep)

log("="*55)
log(f"\n>> Results saved to: {log_path}")
log_file.close()
