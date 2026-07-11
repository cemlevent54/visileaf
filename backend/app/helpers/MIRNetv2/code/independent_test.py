"""
Bağımsız Low-Light Enhancement Test Scripti
MIRNet-v2 modelini kullanarak görüntü iyileştirme yapar.
Enhancement/test.py ile uyumlu parametreler kullanır.

Kullanım:
    python independent_test.py --input_image path/to/input.jpg --model_path path/to/enhancement_lol.pth --output_image path/to/output.png
    python independent_test.py --input_image input.jpg --model_path Enhancement/pretrained_models/enhancement_lol.pth --output_image output.png --yaml_file Enhancement/Options/Enhancement_MIRNet_v2_Lol.yml
"""

import torch
import torch.nn.functional as F
import os
import argparse
import yaml
from skimage import img_as_ubyte
import cv2
import numpy as np

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from basicsr.models.archs.mirnet_v2_arch import MIRNet_v2

def load_img(filepath):
    """Görüntüyü yükler ve RGB formatına çevirir"""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {filepath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    """Görüntüyü kaydeder"""
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def enhance_image(input_image_path, model_path, output_image_path, yaml_file=None, tile_size=None, tile_overlap=32):
    """
    Low-light enhancement işlemi yapar
    
    Args:
        input_image_path: Giriş görüntüsünün yolu
        model_path: Model dosyasının yolu (.pth)
        output_image_path: Çıkış görüntüsünün kaydedileceği yol
        yaml_file: YAML konfigürasyon dosyası (opsiyonel, Enhancement/test.py ile uyumluluk için)
        tile_size: Büyük görüntüler için tile boyutu (None = orijinal çözünürlükte işle)
        tile_overlap: Tile'lar arasındaki örtüşme miktarı
    """
    
    # Model dosyasının varlığını kontrol et
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    # Giriş görüntüsünün varlığını kontrol et
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Giriş görüntüsü bulunamadı: {input_image_path}")
    
    print(f"Giriş görüntüsü yükleniyor: {input_image_path}")
    img = load_img(input_image_path)
    
    # Model parametrelerini yükle (YAML varsa ondan, yoksa varsayılan değerler)
    if yaml_file and os.path.exists(yaml_file):
        print(f"YAML dosyasından parametreler yükleniyor: {yaml_file}")
        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')  # Enhancement/test.py ile aynı işlem
        parameters = x['network_g']
    else:
        # Varsayılan parametreler (Enhancement_MIRNet_v2_Lol.yml ile aynı)
        print("Varsayılan parametreler kullanılıyor...")
        parameters = {
            'inp_channels': 3,
            'out_channels': 3,
            'n_feat': 80,
            'chan_factor': 1.5,
            'n_RRG': 4,
            'n_MRB': 2,
            'height': 3,
            'width': 2,
            'scale': 1
        }
    
    # Model mimarisini yükle (Enhancement/test.py ile aynı şekilde)
    print("Model mimarisi yükleniyor...")
    model = MIRNet_v2(**parameters)
    
    # Cihazı belirle (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hesaplama cihazı: {device}")
    model.to(device)
    
    # Model ağırlıklarını yükle
    print(f"Model ağırlıkları yükleniyor: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    
    # Görüntüyü tensor formatına çevir
    input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Padding işlemi (4'ün katı olması için) - Enhancement/test.py ile aynı
    factor = 4  # Enhancement/test.py ile uyumlu
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    H = ((height + factor) // factor) * factor
    W = ((width + factor) // factor) * factor
    padh = H - height if height % factor != 0 else 0
    padw = W - width if width % factor != 0 else 0
    input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')
    
    print("Görüntü iyileştirme işlemi başlatılıyor...")
    
    # Inference
    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if tile_size is None:
            # Orijinal çözünürlükte işle
            restored = model(input_tensor)
        else:
            # Tile modunda işle (büyük görüntüler için)
            print(f"Tile modu: {tile_size}x{tile_size}, örtüşme: {tile_overlap}")
            b, c, h, w = input_tensor.shape
            tile = min(tile_size, h, w)
            assert tile % 4 == 0, "Tile boyutu 4'ün katı olmalıdır"
            
            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_tensor)
            W = torch.zeros_like(E)
            
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    
                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            
            restored = E.div_(W)
        
        # Değerleri [0, 1] aralığına sınırla
        restored = torch.clamp(restored, 0, 1)
        
        # Padding'i kaldır
        restored = restored[:, :, :height, :width]
        
        # NumPy formatına çevir
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
    
    # Çıkış dizinini oluştur
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Görüntüyü kaydet
    print(f"İyileştirilmiş görüntü kaydediliyor: {output_image_path}")
    save_img(output_image_path, restored)
    
    print("✓ İşlem tamamlandı!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIRNet-v2 Low-Light Enhancement Test Scripti')
    parser.add_argument('--input_image', required=True, type=str, 
                       help='Giriş görüntüsünün yolu (örn: input.jpg)')
    parser.add_argument('--model_path', required=True, type=str,
                       help='Model dosyasının yolu (örn: Enhancement/pretrained_models/enhancement_lol.pth)')
    parser.add_argument('--output_image', required=True, type=str,
                       help='Çıkış görüntüsünün kaydedileceği yol (örn: output.png)')
    parser.add_argument('--yaml_file', type=str, default=None,
                       help='YAML konfigürasyon dosyası (opsiyonel, Enhancement/Options/Enhancement_MIRNet_v2_Lol.yml)')
    parser.add_argument('--tile', type=int, default=None,
                       help='Büyük görüntüler için tile boyutu (örn: 720). Belirtilmezse orijinal çözünürlükte işlenir')
    parser.add_argument('--tile_overlap', type=int, default=32,
                       help='Tile\'lar arasındaki örtüşme miktarı (varsayılan: 32)')
    
    args = parser.parse_args()
    
    try:
        enhance_image(
            input_image_path=args.input_image,
            model_path=args.model_path,
            output_image_path=args.output_image,
            yaml_file=args.yaml_file,
            tile_size=args.tile,
            tile_overlap=args.tile_overlap
        )
    except Exception as e:
        print(f"❌ Hata: {e}")
        exit(1)

