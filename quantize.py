import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_nndct.apis import torch_quantizer

# 1. Kendi Model Mimarini Tanımlama Fonksiyonu
def build_model(model_path):
    # Eğittiğin ResNet50 mimarisini kuruyoruz
    model = resnet50(weights=None) # Ağırlıkları birazdan pth dosyasından yükleyeceğiz
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    
    # Eğitilmiş ağırlıkları yükle
    print(f">> Model ağırlıkları yükleniyor: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# 2. Senin Kodundaki Transform Yapısı
def get_transforms():
    # Klasik torchvision.transforms kullanımı
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Resimleri otomatik float32 yapar ve [0, 1] arasına ölçekler
        transforms.Grayscale(num_output_channels=3), # 3 kanal hilesi
    ])

def quantize(model_dir, output_dir, target, data_dir):
    model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Modeli Hazırla
    model = build_model(model_path)

    # 3. Gerçek Veri Yükleyiciyi (DataLoader) Bağlama
    # Vitis AI kalibrasyonu için 100-200 adet gerçek resim yeterlidir.
    # Bu yüzden val veya train klasörünü kullanabiliriz. batch_size=32 idealdir.
    print(f">> Kalibrasyon için veri seti yükleniyor: {data_dir}")
    calib_dataset = ImageFolder(data_dir, transform=get_transforms())
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True) 

    # Vitis AI'ın graph yapısını çözmesi için örnek bir girdi boyutu (Batch_size, Kanal, Yükseklik, Genişlik)
    inputs = torch.randn([1, 3, 224, 224])

    # 4. Vitis AI Quantizer Başlatma (CALIB MODU)
    print(">> Vitis AI Quantizer başlatılıyor (Calibration Modu)...")
    quantizer = torch_quantizer(
        quant_mode='calib',
        module=model,
        input_args=(inputs,),
        output_dir=output_dir,
        target=target
    )
    
    quant_model = quantizer.quant_model
    print(">> Gerçek görüntülerle Kalibrasyon (Forward Pass) işlemi yapılıyor...")
    
    # Toplamda ~100-200 arası resmi kalibrasyondan geçirmek yeterli (Örn: 5 batch x 32 = 160 resim)
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            _ = quant_model(images)
            if i >= 5: # 5 batch sonra durduruyoruz (yeterli örneklem için)
                break
            
    quantizer.export_quant_config()
    print(">> Kalibrasyon konfigürasyonu dışa aktarıldı.")

    # 5. Vitis AI Quantizer Başlatma (TEST MODU)
    print(">> Vitis AI Quantizer başlatılıyor (Test Modu)...")
    quantizer = torch_quantizer(
        quant_mode='test',
        module=model,
        input_args=(inputs,),
        output_dir=output_dir,
        target=target
    )
    
    quant_model = quantizer.quant_model
    print(">> Test modu doğrulaması yapılıyor...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            _ = quant_model(images)
            if i >= 5:
                break

    # 6. .xmodel Çıktısı Üretme
    print(">> .xmodel ve deploy dosyaları dump ediliyor...")
    quantizer.export_xmodel(output_dir=output_dir, deploy_check=False)
    print(f">> İşlem başarıyla tamamlandı! Çıktılar '{output_dir}' klasöründe.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='.', help='best_model.pth dosyasının bulunduğu klasör')
    parser.add_argument('--data_dir', type=str, default='data/val', help='Kalibrasyon için kullanılacak resimlerin klasörü')
    parser.add_argument('--output_dir', type=str, default='./quant_output', help='Çıktı (.xmodel) klasörü')
    parser.add_argument('--target', type=str, default='DPUCZDX8G_ISA1_B3136', help='Hedef FPGA/DPU mimarisi')
    args = parser.parse_args()
    
    quantize(args.model_dir, args.output_dir, args.target, args.data_dir)
