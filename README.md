# KV260 Driver Drowsiness Detection

Real-time driver drowsiness detection system deployed on the AMD Xilinx KV260 Vision AI Starter Kit.

The project uses a ResNet50 model trained to classify eye states as **Awake** or **Sleepy** and performs real-time inference from a camera stream on embedded hardware.

---

## Overview

Driver drowsiness is one of the leading causes of traffic accidents worldwide. This project aims to detect signs of driver fatigue by monitoring eye states in real time.

The system captures frames from a camera, preprocesses the eye region, performs inference using a trained neural network, and displays the predicted eye state on the video stream.

---

## Features

- Real-time camera inference
- Awake / Sleepy detection
- Embedded deployment on KV260
- PyTorch-based model
- OpenCV video processing pipeline
- ResNet50 architecture

---

## Hardware

- AMD Xilinx KV260 Vision AI Starter Kit
- USB Camera
- Linux / PetaLinux Environment

---

## Software Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Vitis AI Environment

---

## Project Structure

## Project Structure

```text
.
├── train.py                    # Model training script
├── train.ipynb                 # Jupyter notebook for training experiments
├── best_model.pth              # Trained PyTorch model

├── quantize.py                 # Model quantization script
├── arch.json                   # Quantization architecture configuration

├── deploy_img.py               # Single image inference
├── deploy_video.py             # Real-time camera inference

├── resnet50_compiled.xmodel    # Compiled model for KV260 DPU
├── compiled_output/            # Vitis AI compilation outputs
├── quantized_output/           # Quantized model outputs
├── reports/                    # VAI Trace and power/performance reports

├── data/                       # Dataset directory

├── awake.png                   # Sample awake eye image
├── sleepy.png                  # Sample sleepy eye image

├── how_to_run_analyzers.txt    # Evaluation & power tools instructions
├── image_evaluate_accuracy.py  # Image-based evaluation results 
├── video_evaluate_accuracy.py  # Real-time evaluation results
├── power_monitor.sh            # To analyze power metrics

├── how_to_compile.txt          # Vitis AI compilation instructions
├── notes.txt                   # Development notes
├── requirements.txt            # Virtual environment libraries for training

└── README.md
```

## Model

The model is trained to classify eye images into two categories:

| Class | Description |
|---------|-------------|
| Awake | Driver is alert |
| Sleepy | Driver shows signs of drowsiness |

The trained model is stored as:

```text
best_model.pth
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/bartuyzc/KV260-Driver-Drowsiness-Detection.git
cd KV260-Driver-Drowsiness-Detection
```

---

## Running the Application

Start real-time inference:

```bash
python3 deploy_video.py
```
Start single-image inference:

```bash
# change the image path inside the code 
python3 deploy_image.py
```

The application will:

1. Open the camera stream
2. Capture video frames
3. Run model inference
4. Display the predicted eye state

---

## Future Improvements

- Face detection integration
- Eye localization pipeline
- Drowsiness score calculation
- Temporal smoothing
- FPGA/DPU acceleration using Vitis AI
- Driver monitoring dashboard
- Gaze estimation
- Yawn detection

---

## Applications

- Driver Monitoring Systems (DMS)
- Automotive Safety
- Intelligent Transportation Systems
- Edge AI Vision Applications
- Embedded Computer Vision

---

## Author

Hüseyin Bartu Yazıcı

M.Sc. Electronics Engineering

Gebze Technical University

GitHub: https://github.com/bartuyzc

---

## License

This project is intended for educational and research purposes.
