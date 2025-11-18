import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8), save_results=True):
    """
    Belirtilen görüntüye CLAHE uygular.

    Args:
        image_path (str veya Path): İşlenecek görüntünün dosya yolu.
        clip_limit (float): Kontrast sınırlama eşiği. Genellikle 2.0 ile 4.0 arasında.
        tile_grid_size (tuple): Görüntünün bölüneceği ızgara boyutu (NxN).
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

    # 2. Görüntüyü Renk Uzayına Dönüştürme (LAB)
    # CLAHE'yi sadece parlaklık (L) kanalı üzerinde uygulamak en iyisidir.
    # Bu, renklerin değişmeden kalmasını sağlar.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # LAB renk uzayını kanallara ayır
    # L (Parlaklık), A (Yeşilden Kırmızıya), B (Maviden Sarıya)
    l, a, b = cv2.split(lab)

    # 3. CLAHE Nesnesini Oluşturma
    # Kontrastı sınırlama (Clip Limit) ve ızgara boyutu (Tile Grid Size) ayarlanır.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # 4. CLAHE'yi Parlaklık (L) Kanalına Uygulama
    cl = clahe.apply(l)

    # 5. Birleştirme ve Renk Uzayını Geri Dönüştürme
    # İşlenmiş L kanalını, A ve B kanalları ile birleştir
    limg = cv2.merge((cl, a, b))
    
    # LAB'dan BGR (standart renk) uzayına geri dönüştür
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 6. Results klasörünü oluştur (kaydetme için)
    if save_results:
        results_dir = script_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Zaman damgası ile dosya adları oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = str(results_dir / f'clahe_comparison_{timestamp}.png')
        output_image_filename = str(results_dir / f'clahe_output_{timestamp}.jpg')
        info_filename = str(results_dir / f'clahe_info_{timestamp}.txt')

    # 7. Sonuçları Görüntüleme
    plt.figure(figsize=(14, 7))
    
    # Orijinal Görüntü
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal Görüntü", fontsize=14, fontweight='bold')
    plt.axis('off')

    # CLAHE Uygulanmış Görüntü
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title(f"CLAHE Uygulanmış\n(Clip: {clip_limit}, Tile: {tile_grid_size})", 
              fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    
    # Karşılaştırma grafiğini kaydet
    if save_results:
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma grafiği kaydedildi: {comparison_filename}")
        
        # İyileştirilmiş görüntüyü kaydet
        cv2.imwrite(output_image_filename, final_img)
        print(f"İyileştirilmiş görüntü kaydedildi: {output_image_filename}")
        
        # Bilgi dosyasını kaydet
        with open(info_filename, 'w', encoding='utf-8') as f:
            f.write(f"CLAHE İşleme Bilgileri\n")
            f.write(f"İşlem Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Orijinal Görüntü: {image_path_str}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Clip Limit: {clip_limit}\n")
            f.write(f"Tile Grid Size: {tile_grid_size}\n")
            f.write(f"Görüntü Boyutu: {img.shape}\n")
            f.write(f"\nÇıktı Dosyaları:\n")
            f.write(f"- Karşılaştırma Grafiği: {comparison_filename}\n")
            f.write(f"- İyileştirilmiş Görüntü: {output_image_filename}\n")
        
        print(f"İşlem bilgileri kaydedildi: {info_filename}")
    
    plt.show()
    print("CLAHE işlemi tamamlandı. Sonuçlar görüntülendi.")
    
    return final_img

# --- KULLANIM ---
if __name__ == "__main__":
    # Script'in bulunduğu klasörü baz al (python klasörü)
    script_dir = Path(__file__).parent.resolve()
    # Proje kök dizini (bir üst dizin)
    project_root = script_dir.parent
    
    # Dataset klasöründen resim yolu - Buradaki dosya yolunu değiştirebilirsiniz
    relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_16-48-55.jpg'
    # Proje kök dizinine göre mutlak yol oluştur
    image_path = project_root / relative_image_path
    
    # Önerilen başlangıç ayarları
    # Parametreleri deneme yanılma ile ayarlayarak en iyi sonucu bulabilirsiniz.
    apply_clahe(image_path, clip_limit=3.0, tile_grid_size=(8, 8), save_results=True)