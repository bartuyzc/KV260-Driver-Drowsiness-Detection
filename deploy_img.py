import os
import cv2
import numpy as np
import vart
import xir

# 1. Modeli ve DPU Alt Grafiğini (Subgraph) Yükleme
MODEL_PATH = "resnet50_compiled.xmodel"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Hata: {MODEL_PATH} dosyası bulunamadı! Lütfen yolu kontrol edin.")

print(">> Model yükleniyor...")
graph = xir.Graph.deserialize(MODEL_PATH)
root = graph.get_root_subgraph()

# Model içindeki gerçek DPU (donanım) parçalarını otomatik bulan fonksiyon
def get_dpu_subgraph(root_graph):
    child_subgraphs = root_graph.get_children()
    dpu_subgraphs = []
    for s in child_subgraphs:
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU":
            dpu_subgraphs.append(s)
    return dpu_subgraphs

dpu_subgraphs = get_dpu_subgraph(root)

if not dpu_subgraphs:
    raise Exception("Hata: Model içerisinde DPU için derlenmiş bir subgraph bulunamadı!")

# DPU Runner'ını (Sürücüsünü) başlatıyoruz
print(">> DPU Runner oluşturuluyor...")
runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

# 2. Giriş ve Çıkış Tensor Bilgilerini Alma
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

input_dims = tuple(input_tensors[0].dims)   # Örn: [1, 224, 224, 3]
output_dims = tuple(output_tensors[0].dims) # Örn: [1, 2]

# 3. Görüntü Önişleme (Senin Eğitim Kodundaki Transform Mantığı)
IMAGE_NAME = "test_image.jpg" # Test etmek istediğin resmin adı

if not os.path.exists(IMAGE_NAME):
    print(f"Hata: Test için '{IMAGE_NAME}' bulunamadı. Lütfen klasöre bir test resmi koyun.")
    del runner
    exit()

# Resmi oku ve eğitimdeki gibi 224x224 boyutuna getir
img = cv2.imread(IMAGE_NAME)
img = cv2.resize(img, (224, 224))

# Eğitimdeki transforms.ToTensor() mantığı: 
# Resmi Float32 yap ve [0, 255] aralığından [0.0, 1.0] arasına ölçekle
img = img.astype(np.float32) / 255.0

# Vitis AI Giriş Formatı Ayarı: Modeli [Batch_Size, Height, Width, Channel] olarak paketle
input_data = np.expand_dims(img, axis=0)

# 4. Bellek Alanlarını (Buffer) Hazırlama
input_buffer = [np.ascontiguousarray(input_data)]
output_buffer = [np.zeros(output_dims, dtype=np.float32)]

# 5. DPU Üzerinde Çıkarım (Inference) Yapma
print(">> DPU üzerinde tahmin işlemi başlatıldı...")
job_id = runner.execute_async(input_buffer, output_buffer)
runner.wait(job_id)

# 6. Sonuçları Ekrana Yazdırma
raw_outputs = output_buffer[0]
probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs), axis=1, keepdims=True) # Softmax hilesi
predicted_class = np.argmax(probabilities)

print("\n--- TAHMİN SONUÇLARI ---")
print(f"Tahmin Edilen Sınıf İndeksi: {predicted_class}")
print(f"Sınıf Olasılıkları: {probabilities[0]}")
print("-------------------------\n")

# Belleği temizle ve kapat
del runner
print(">> İşlem tamamlandı, kaynaklar serbest bırakıldı.")
