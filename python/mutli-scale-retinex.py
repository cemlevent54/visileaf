import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

def multi_scale_retinex(img, sigma_list):
    """
    Multi-Scale Retinex (MSR) uygulamasını gerçekleştirir.
    
    Args:
        img (np.array): RGB veya BGR formatında girdi görüntüsü.
        sigma_list (list): Kullanılacak standart sapma (sigma) değerleri listesi.
        
    Returns:
        np.array: İşlenmiş MSR görüntüsü.
    """
    
    # 1. Logaritmik Dönüşüm
    img_float = img.astype(float) + 1.0
    log_img = np.log10(img_float)
    
    MSR_result = np.zeros_like(log_img)
    num_scales = len(sigma_list)
    
    # Boş sigma listesi kontrolü
    if num_scales == 0:
        return img.astype(np.uint8)
    
    # 2. Her Ölçek İçin SSR Uygulama ve Toplama
    for sigma in sigma_list:
        # Aydınlatma Tahmini (Gauss Konvolüsyonu)
        illumination = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log_illumination = np.log10(illumination)
        
        # Log R = log S - log L
        log_retinex = log_img - log_illumination
        
        # Sonuçları eşit ağırlıklarla topla (W_i = 1/N)
        MSR_result += (1.0 / num_scales) * log_retinex

    # 3. Dinamik Aralık Sıkıştırma/Normalizasyon
    # Renk tutarlılığı için bu adım MSR'de kritiktir.
    min_val, max_val, _, _ = cv2.minMaxLoc(MSR_result)
    
    # Sıfıra bölme hatasını önle: max_val == min_val durumunda
    if max_val == min_val or abs(max_val - min_val) < 1e-10:
        # Tüm değerler aynıysa, orijinal görüntüyü döndür
        final_msr = img.astype(np.uint8)
        return final_msr
    
    # Normalizasyon: [0, 255] arasına sığdır
    normalized_msr = (MSR_result - min_val) * (255.0 / (max_val - min_val))
    
    # 4. Doğrusal Alana Geri Dönüş
    final_msr = normalized_msr.astype(np.uint8)
    
    return final_msr

def apply_msr(image_path, sigma_list=[15, 80, 250], save_results=True):
    """
    Multi-Scale Retinex (MSR) işlemini uygular ve sonuçları kaydeder.
    
    Args:
        image_path (str veya Path): İşlenecek görüntünün dosya yolu.
        sigma_list (list): Kullanılacak standart sapma (sigma) değerleri listesi.
                           Geleneksel olarak 3 ölçek kullanılır: [15, 80, 250]
        save_results (bool): Sonuçları results klasörüne kaydet.
    
    Returns:
        np.array: İşlenmiş MSR görüntüsü.
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
    
    # Renkli görüntü için her BGR kanalına MSR uygula
    b, g, r = cv2.split(img)
    
    b_msr = multi_scale_retinex(b, sigma_list)
    g_msr = multi_scale_retinex(g, sigma_list)
    r_msr = multi_scale_retinex(r, sigma_list)
    
    msr_output = cv2.merge([b_msr, g_msr, r_msr])
    
    # Results klasörünü oluştur (kaydetme için)
    if save_results:
        results_dir = script_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Zaman damgası ile dosya adları oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = str(results_dir / f'msr_comparison_{timestamp}.png')
        output_image_filename = str(results_dir / f'msr_output_{timestamp}.jpg')
        info_filename = str(results_dir / f'msr_info_{timestamp}.txt')
    
    # Görüntüleme
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal Görüntü", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(msr_output, cv2.COLOR_BGR2RGB))
    plt.title(f"MSR (Sigmas: {sigma_list})", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Karşılaştırma grafiğini kaydet
    if save_results:
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma grafiği kaydedildi: {comparison_filename}")
        
        # İyileştirilmiş görüntüyü kaydet
        cv2.imwrite(output_image_filename, msr_output)
        print(f"İyileştirilmiş görüntü kaydedildi: {output_image_filename}")
        
        # Bilgi dosyasını kaydet
        with open(info_filename, 'w', encoding='utf-8') as f:
            f.write(f"Multi-Scale Retinex (MSR) İşleme Bilgileri\n")
            f.write(f"İşlem Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Orijinal Görüntü: {image_path_str}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Sigma Değerleri: {sigma_list}\n")
            f.write(f"Ölçek Sayısı: {len(sigma_list)}\n")
            f.write(f"Görüntü Boyutu: {img.shape}\n")
            f.write(f"\nMSR Açıklaması:\n")
            f.write(f"- MSR, birden fazla ölçekte Retinex işlemini birleştirir.\n")
            f.write(f"- Her sigma değeri farklı ölçekteki detayları yakalar:\n")
            for i, sigma in enumerate(sigma_list, 1):
                if sigma < 50:
                    scale_desc = "Yüksek frekans detayları (ince detaylar)"
                elif sigma < 150:
                    scale_desc = "Orta frekans detayları (genel yapı)"
                else:
                    scale_desc = "Düşük frekans detayları (genel ışık)"
                f.write(f"  {i}. Sigma={sigma}: {scale_desc}\n")
            f.write(f"- Geleneksel 3 ölçek: [15, 80, 250]\n")
            f.write(f"\nÇıktı Dosyaları:\n")
            f.write(f"- Karşılaştırma Grafiği: {comparison_filename}\n")
            f.write(f"- İyileştirilmiş Görüntü: {output_image_filename}\n")
        
        print(f"İşlem bilgileri kaydedildi: {info_filename}")
    
    plt.show()
    print("MSR işlemi tamamlandı. Sonuçlar görüntülendi.")
    
    return msr_output

# --- KULLANIM ---
if __name__ == "__main__":
    # Script'in bulunduğu klasörü baz al (python klasörü)
    script_dir = Path(__file__).parent.resolve()
    # Proje kök dizini (bir üst dizin)
    project_root = script_dir.parent
    
    # Dataset klasöründen resim yolu - Buradaki dosya yolunu değiştirebilirsiniz
    relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_17-29-25.jpg'
    # Proje kök dizinine göre mutlak yol oluştur
    image_path = project_root / relative_image_path
    
    # Geleneksel 3 ölçekli sigma değerleri
    sigma_list = [5, 80, 350]
    MSR_SIGMAS = sigma_list
    
    apply_msr(image_path, sigma_list=MSR_SIGMAS, save_results=True)