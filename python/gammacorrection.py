import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

def apply_gamma_correction(image_path, gamma=0.5, save_results=True):
    """
    Belirtilen görüntüye Gamma Düzeltme uygular.

    Args:
        image_path (str veya Path): İşlenecek görüntünün dosya yolu.
        gamma (float): Gamma değeri. 
                       Karanlık görüntüler için < 1 (örn: 0.5) kullanılır.
                       Çok parlak görüntüler için > 1 (örn: 1.5) kullanılır.
        save_results (bool): Sonuçları results klasörüne kaydet.
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
        
        # 1. Görüntüyü yükle
        img = cv2.imread(image_path_str)
        
        if img is None:
            raise FileNotFoundError(f"Görüntü dosyası yüklenemedi (muhtemelen geçersiz format): {image_path_str}")

    except FileNotFoundError as e:
        print(e)
        return None

    # --- Gamma Düzeltme Matrisi Oluşturma ---
    # 2. Piksel yoğunluklarını normalize et (0.0 ile 1.0 arasına getir)
    #    Bu, formülün düzgün çalışması için gereklidir.
    
    # 3. Dönüşüm tablosunu (lookup table - LUT) hesapla
    #    0'dan 255'e kadar her piksel değeri için yeni değeri hesaplarız.
    #    Formül: V_out = 255 * (V_in / 255) ^ gamma
    
    # 0'dan 255'e kadar bir dizi oluştur
    table = np.array([((i / 255.0) ** gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # 4. Dönüşümü Görüntüye Uygulama
    # cv2.LUT (Look Up Table) fonksiyonu, piksel değerlerini 
    # oluşturduğumuz tabloya göre hızlıca değiştirir.
    gamma_corrected_img = cv2.LUT(img, table)

    # 5. Results klasörünü oluştur (kaydetme için)
    if save_results:
        results_dir = script_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Zaman damgası ile dosya adları oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = str(results_dir / f'gamma_comparison_{timestamp}.png')
        output_image_filename = str(results_dir / f'gamma_output_{timestamp}.jpg')
        info_filename = str(results_dir / f'gamma_info_{timestamp}.txt')

    # 6. Sonuçları Görüntüleme
    plt.figure(figsize=(14, 7))
    
    # Orijinal Görüntü
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal Görüntü", fontsize=14, fontweight='bold')
    plt.axis('off')

    # Gamma Düzeltme Uygulanmış Görüntü
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(gamma_corrected_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Gamma Düzeltme Uygulanmış\n(Gamma: {gamma})", 
              fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    
    # Karşılaştırma grafiğini kaydet
    if save_results:
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma grafiği kaydedildi: {comparison_filename}")
        
        # İyileştirilmiş görüntüyü kaydet
        cv2.imwrite(output_image_filename, gamma_corrected_img)
        print(f"İyileştirilmiş görüntü kaydedildi: {output_image_filename}")
        
        # Bilgi dosyasını kaydet
        with open(info_filename, 'w', encoding='utf-8') as f:
            f.write(f"Gamma Düzeltme İşleme Bilgileri\n")
            f.write(f"İşlem Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Orijinal Görüntü: {image_path_str}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Gamma Değeri: {gamma}\n")
            f.write(f"Görüntü Boyutu: {img.shape}\n")
            f.write(f"\nGamma Düzeltme Açıklaması:\n")
            if gamma < 1.0:
                f.write(f"- Gamma < 1.0: Görüntüyü aydınlatır (karanlık görüntüler için)\n")
            elif gamma > 1.0:
                f.write(f"- Gamma > 1.0: Görüntüyü karartır (parlak görüntüler için)\n")
            else:
                f.write(f"- Gamma = 1.0: Değişiklik yapmaz\n")
            f.write(f"\nÇıktı Dosyaları:\n")
            f.write(f"- Karşılaştırma Grafiği: {comparison_filename}\n")
            f.write(f"- İyileştirilmiş Görüntü: {output_image_filename}\n")
        
        print(f"İşlem bilgileri kaydedildi: {info_filename}")
    
    plt.show()
    print("Gamma Düzeltme işlemi tamamlandı. Sonuçlar görüntülendi.")
    
    return gamma_corrected_img

# --- KULLANIM ---
if __name__ == "__main__":
    # Script'in bulunduğu klasörü baz al (python klasörü)
    script_dir = Path(__file__).parent.resolve()
    # Proje kök dizini (bir üst dizin)
    project_root = script_dir.parent
    
    # Dataset klasöründen resim yolu - Buradaki dosya yolunu değiştirebilirsiniz
    relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_17-39-33.jpg'
    # Proje kök dizinine göre mutlak yol oluştur
    image_path = project_root / relative_image_path
    
    # Karanlık bir görüntüyü aydınlatmak için genellikle 0.4 ile 0.7 arasında bir değer kullanılır.
    # Çok parlak görüntüler için 1.5 ile 2.5 arası değerler kullanılabilir.
    apply_gamma_correction(image_path, gamma=0.4, save_results=True)