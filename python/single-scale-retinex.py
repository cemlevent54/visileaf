import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

def single_scale_retinex(img, sigma):
    """
    Single-Scale Retinex (SSR) uygulamasını gerçekleştirir.
    
    Args:
        img (np.array): RGB veya BGR formatında girdi görüntüsü.
        sigma (int): Gauss filtresinin standart sapması (Örn: 80).
        
    Returns:
        np.array: İşlenmiş R (Yansıtma) bileşeni görüntüsü.
    """
    
    # 1. Logaritmik Dönüşüm
    # Görüntü değerlerini 1.0 ile 256.0 arasına normalize edip logaritmasını al
    log_img = np.log10(img.astype(float) + 1.0) 
    
    # 2. Aydınlatma Tahmini (Gauss Konvolüsyonu)
    # Gauss filtresinin boyutu sigma'ya göre otomatik belirlenir (Örn: 3*sigma + 1)
    # cv2.GaussianBlur, img'yi yumuşatır ve L (Aydınlatma) tahmini verir.
    illumination = cv2.GaussianBlur(img, (0, 0), sigma)
    log_illumination = np.log10(illumination.astype(float) + 1.0)
    
    # 3. Yansıtma Bileşenini Hesaplama (log R = log S - log L)
    # Log R bileşenini bulmak için çıkarma işlemi yapılır
    log_retinex = log_img - log_illumination
    
    # 4. Dinamik Aralık Sıkıştırma/Normalizasyon (Son İşlem)
    # İyileştirilmiş kontrast için log R bileşenini normalize et
    min_val, max_val, _, _ = cv2.minMaxLoc(log_retinex)
    normalized_retinex = (log_retinex - min_val) * (255.0 / (max_val - min_val))
    
    # 5. Doğrusal Alana Geri Dönüş
    # Sonucu uint8 formatına çevir
    final_retinex = normalized_retinex.astype(np.uint8)
    
    return final_retinex

def apply_ssr(image_path, sigma=80, save_results=True):
    """
    Single-Scale Retinex (SSR) işlemini uygular ve sonuçları kaydeder.
    
    Args:
        image_path (str veya Path): İşlenecek görüntünün dosya yolu.
        sigma (int): Gauss filtresinin standart sapması. Küçük sigma yüksek kontrast, 
                     büyük sigma daha iyi renk tutarlılığı verir. Örn: 30-150.
        save_results (bool): Sonuçları results klasörüne kaydet.
    
    Returns:
        np.array: İşlenmiş SSR görüntüsü.
    """
    # Script'in bulunduğu klasörü baz al (python klasörü)
    script_dir = Path(__file__).parent.resolve()
    
    # Path objesi ise string'e çevir
    if isinstance(image_path, Path):
        image_path_obj = image_path
        image_path_str = str(image_path)
    else:
        image_path_obj = Path(image_path)
        image_path_str = image_path
    
    try:
        # Dosyanın varlığını kontrol et
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path_str}")
        
        img = cv2.imread(image_path_str)
        
        if img is None:
            raise FileNotFoundError(f"Görüntü dosyası yüklenemedi (muhtemelen geçersiz format): {image_path_str}")
    
    except FileNotFoundError as e:
        print(e)
        return None
    
    # Renkli görüntü için her BGR kanalına SSR uygula
    b, g, r = cv2.split(img)
    
    b_retinex = single_scale_retinex(b, sigma)
    g_retinex = single_scale_retinex(g, sigma)
    r_retinex = single_scale_retinex(r, sigma)
    
    ssr_output = cv2.merge([b_retinex, g_retinex, r_retinex])
    
    # Results klasörünü oluştur (kaydetme için)
    if save_results:
        results_dir = script_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Zaman damgası ile dosya adları oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = str(results_dir / f'ssr_comparison_{timestamp}.png')
        output_image_filename = str(results_dir / f'ssr_output_{timestamp}.jpg')
        info_filename = str(results_dir / f'ssr_info_{timestamp}.txt')
    
    # Görüntüleme
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal Görüntü", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(ssr_output, cv2.COLOR_BGR2RGB))
    plt.title(f"SSR (Sigma: {sigma})", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Karşılaştırma grafiğini kaydet
    if save_results:
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma grafiği kaydedildi: {comparison_filename}")
        
        # İyileştirilmiş görüntüyü kaydet
        cv2.imwrite(output_image_filename, ssr_output)
        print(f"İyileştirilmiş görüntü kaydedildi: {output_image_filename}")
        
        # Bilgi dosyasını kaydet
        with open(info_filename, 'w', encoding='utf-8') as f:
            f.write(f"Single-Scale Retinex (SSR) İşleme Bilgileri\n")
            f.write(f"İşlem Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Orijinal Görüntü: {image_path_str}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Sigma Değeri: {sigma}\n")
            f.write(f"Görüntü Boyutu: {img.shape}\n")
            f.write(f"\nSSR Açıklaması:\n")
            f.write(f"- Sigma, Gauss filtresinin standart sapmasıdır.\n")
            f.write(f"- Küçük sigma (örn: 30): Yüksek kontrast, daha fazla detay\n")
            f.write(f"- Büyük sigma (örn: 150): Daha iyi renk tutarlılığı, daha yumuşak sonuç\n")
            f.write(f"- Önerilen değer aralığı: 50-100\n")
            f.write(f"\nÇıktı Dosyaları:\n")
            f.write(f"- Karşılaştırma Grafiği: {comparison_filename}\n")
            f.write(f"- İyileştirilmiş Görüntü: {output_image_filename}\n")
        
        print(f"İşlem bilgileri kaydedildi: {info_filename}")
    
    plt.show()
    print("SSR işlemi tamamlandı. Sonuçlar görüntülendi.")
    
    return ssr_output

# --- KULLANIM ---
if __name__ == "__main__":
    # Script'in bulunduğu klasörü baz al (python klasörü)
    script_dir = Path(__file__).parent.resolve()
    # Proje kök dizini (bir üst dizin)
    project_root = script_dir.parent
    
    # Dataset klasöründen resim yolu - Buradaki dosya yolunu değiştirebilirsiniz
    relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_16-59-03.jpg'
    # Proje kök dizinine göre mutlak yol oluştur
    image_path = project_root / relative_image_path
    
    # Sigma değerini deneyerek en iyi sonucu bulun.
    # Küçük sigma (örn: 30) yüksek kontrast, büyük sigma (örn: 150) daha iyi renk tutarlılığı verir.
    SSR_SIGMA = 300
    
    apply_ssr(image_path, sigma=SSR_SIGMA, save_results=True)