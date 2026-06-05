#!/usr/bin/env python3
import os
import cv2
import numpy as np
import vart
import xir
import sys

# ---------------------------------------------------------
# 1. Yapılandırma ve Sınıf Etiketleri
# ---------------------------------------------------------
MODEL_PATH = "resnet50_compiled.xmodel"
CLASSES = ["AWAKE", "SLEEPY"] 

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Hata: {MODEL_PATH} dosyası bulunamadı!")

# ---------------------------------------------------------
# 2. Modeli ve DPU Sürücüsünü Yükleme
# ---------------------------------------------------------
print(">> Model yükleniyor...")
graph = xir.Graph.deserialize(MODEL_PATH)
root = graph.get_root_subgraph()

def get_dpu_subgraph(root_graph):
    child_subgraphs = root_graph.get_children()
    for s in child_subgraphs:
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
            return s
    return None

dpu_subgraph = get_dpu_subgraph(root)
if dpu_subgraph is None:
    raise Exception("Hata: Model içerisinde DPU için derlenmiş bir subgraph bulunamadı!")

print(">> DPU Runner oluşturuluyor...")
runner = vart.Runner.create_runner(dpu_subgraph, "run")

# Tensor Boyutlarını Alma
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

input_dims = tuple(input_tensors[0].dims)   # Örn: (1, 224, 224, 1)
output_dims = tuple(output_tensors[0].dims) # Örn: (1, 2)

target_height = input_dims[1]
target_width  = input_dims[2]

# ---------------------------------------------------------
# 3. USB Webcam Bağlantısı (V4L2)
# ---------------------------------------------------------
print(">> Kamera bağlantısı başlatılıyor...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Hata: USB Webcam açılamadı!")
    del runner
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(">> Canlı analiz başladı. Çıkmak için 'q' tuşuna basın.\n")

# ---------------------------------------------------------
# 4. Sonsuz Video Döngüsü
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Kameradan görüntü alınamadı.")
        break

    display_frame = frame.copy()

    # ---- GÜVENLİ GRAYSCALE PIPELINE ----
    # 1. Görüntüyü Siyah-Beyaz yap
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Modelin beklediği boyuta ölçekle
    img = cv2.resize(gray, (target_width, target_height))
    
    # 3. Float32 tipine dönüştür ve normalize et
    img = img.astype(np.float32) / 255.0
    
    # 4. SEGMENTATION FAULT ÖNLEYİCİ: 
    # Boş bir DPU giriş matrisi oluşturup veriyi içine güvenli bir şekilde kopyalıyoruz.
    # Bu yöntem bellek adreslemesinin donanımla %100 uyumlu olmasını sağlar.
    input_data = np.zeros(input_dims, dtype=np.float32)
    input_data[0, :, :, 0] = img  # Grayscale veriyi tek kanala yerleştir

    # ---- BELLEK VE ÇIKARIM ----
    # Sürücüye (Runner) veriyi beslerken ardışık bellek garantisi veriyoruz
    input_buffer = [np.ascontiguousarray(input_data)]
    output_buffer = [np.zeros(output_dims, dtype=np.float32)]

    try:
        # DPU Çıkarımı
        job_id = runner.execute_async(input_buffer, output_buffer)
        runner.wait(job_id)
    except Exception as e:
        print(f"❌ Çıkarım esnasında hata oluştu: {e}")
        break

    # ---- SONUÇLARI ANLAMLANDIRMA ----
    raw_outputs = output_buffer[0]
    probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs), axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities)
    
    label = CLASSES[predicted_class]
    score = probabilities[0][predicted_class] * 100

    # ---- EKRANA YAZDIRMA ----
    color = (0, 255, 0) if label == "AWAKE" else (0, 0, 255)
    text = f"{label} ({score:.1f}%)"
    cv2.putText(display_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    
    cv2.imshow("Drowsiness Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------------
# 5. Kaynakları Temizleme
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
del runner
print(">> Sistem kapatıldı.")
