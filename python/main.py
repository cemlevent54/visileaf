import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

# Script'in bulunduğu klasörü baz al (python klasörü)
script_dir = Path(__file__).parent.resolve()
# Proje kök dizini (bir üst dizin)
project_root = script_dir.parent

# Dataset klasöründen resim yolu - Buradaki dosya yolunu değiştirebilirsiniz
relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_16-48-55.jpg'
# Proje kök dizinine göre mutlak yol oluştur
image_path = project_root / relative_image_path
# Path objesini string'e çevir (Windows için uyumlu)
image_path_str = str(image_path)

# 1. Görüntüyü Yükleme
try:
    # Dosyanın varlığını kontrol et
    if not image_path.exists():
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path_str}")
    
    img = cv2.imread(image_path_str, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Görüntü dosyası yüklenemedi (muhtemelen geçersiz format): {image_path_str}")

except FileNotFoundError as e:
    print(e)
    # Basit bir örnek matris oluşturarak devam edelim
    print("Örnek 50x50 düşük parlaklıkta görüntü matrisi oluşturuluyor.")
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:40, 10:40] = 50 # 0-255 arasında düşük bir parlaklık değeri

# Eğer görüntü renkliyse, onu gri tonlamaya çevirmek analizi kolaylaştırır
if len(img.shape) == 3:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = img

## ---
## 2. Histogramı Hesaplama
## ---
# cv2.calcHist(görüntüler, kanallar, maske, hist_boyutu, aralıklar)
# [gray_img]: Histogramı hesaplanacak görüntü (listede olmalı).
# [0]: Kanal indeksi (Gri görüntüde tek kanal olduğu için 0).
# None: Maske yok.
# [256]: Histogramdaki bölme (bin) sayısı. 8-bit görüntüler için 256.
# [0, 256]: Piksel değeri aralığı.

hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

# Results klasörünü script klasörü altında oluştur
results_dir = script_dir / 'results'
results_dir.mkdir(exist_ok=True)

# Zaman damgası ile dosya adları oluştur
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
histogram_filename = str(results_dir / f'histogram_{timestamp}.png')
results_filename = str(results_dir / f'analiz_sonuclari_{timestamp}.txt')

## ---
## 3. Histogramı Görüntüleme ve Kaydetme (Analiz için Kritik)
## ---
# Grafik türü: 'bar', 'plot', 'fill', 'step'
graph_type = 'bar'  # Buradan grafik türünü değiştirebilirsiniz

plt.figure(figsize=(12, 6))
plt.title("Görüntü Histogramı", fontsize=14, fontweight='bold')
plt.xlabel("Piksel Yoğunluğu (0: Siyah, 255: Beyaz)", fontsize=12)
plt.ylabel("Piksel Sayısı", fontsize=12)

# Histogram değerlerini düzleştir (1D array'e çevir)
hist_flat = hist.flatten()
x_values = np.arange(256)

if graph_type == 'bar':
    # Bar chart - klasik histogram görünümü
    plt.bar(x_values, hist_flat, color='steelblue', alpha=0.7, width=1.0)
    plt.xlim([-0.5, 255.5])
elif graph_type == 'fill':
    # Filled area plot - daha modern görünüm
    plt.fill_between(x_values, hist_flat, alpha=0.6, color='steelblue')
    plt.plot(x_values, hist_flat, color='darkblue', linewidth=1)
    plt.xlim([0, 256])
elif graph_type == 'step':
    # Step plot - histogram benzeri adım görünümü
    plt.step(x_values, hist_flat, where='mid', color='steelblue', linewidth=1.5)
    plt.fill_between(x_values, hist_flat, alpha=0.3, color='steelblue', step='mid')
    plt.xlim([0, 256])
else:
    # Default: Line plot - çizgi grafiği
    plt.plot(x_values, hist_flat, color='black', linewidth=1.5)
    plt.xlim([0, 256])

plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
print(f"Histogram grafiği kaydedildi: {histogram_filename} (Grafik türü: {graph_type})")
plt.show()

## ---
## 4. Temel Analiz ve Yorumlama
## ---

# Histogramın nerede yoğunlaştığını anlamak için ortalama yoğunluğu hesaplayın
# Yoğunluk * Piksel Sayısı toplamını, Toplam Piksel Sayısına bölerek ortalamayı buluruz.
ortalama_parlaklik = gray_img.mean()

# Karanlık görüntülerde histogramın büyük çoğunluğu 0'a (siyah) yakın değerlerde toplanır.
# Örneğin, ilk 50 bölmedeki (0-49 yoğunluk) piksel sayısının oranını hesaplayalım.
karanlik_esik = 50
karanlik_piksel_sayisi = np.sum(hist[:karanlik_esik])
toplam_piksel_sayisi = gray_img.size

karanlik_oran = (karanlik_piksel_sayisi / toplam_piksel_sayisi) * 100

# Analiz sonuçlarını hazırla
analiz_sonuclari = []
analiz_sonuclari.append("\n--- Analiz Sonuçları ---")
analiz_sonuclari.append(f"Görüntü Boyutu: {gray_img.shape}")
analiz_sonuclari.append(f"Ortalama Piksel Parlaklığı: {ortalama_parlaklik:.2f} (Düşük olması karanlığı işaret eder.)")
analiz_sonuclari.append(f"0'dan {karanlik_esik}'e kadar olan yoğunluktaki piksellerin oranı: {karanlik_oran:.2f}%")

if ortalama_parlaklik < 80 and karanlik_oran > 60:
    yorum = "Yorum: Bu histogram, **düşük ışık (karanlık)** koşullarında çekilmiş bir görüntüyü kuvvetle işaret ediyor."
    oneri = "Önerilen Çözümler: Histogram Eşitleme (HE), CLAHE veya Gamma Düzeltme."
    analiz_sonuclari.append(yorum)
    analiz_sonuclari.append(oneri)
else:
    yorum = "Yorum: Görüntünün piksel dağılımı nispeten daha geniş bir aralığa yayılmış görünüyor."
    analiz_sonuclari.append(yorum)

# Sonuçları konsola yazdır
for satir in analiz_sonuclari:
    print(satir)

# Sonuçları dosyaya kaydet
with open(results_filename, 'w', encoding='utf-8') as f:
    f.write(f"Görüntü Histogram Analizi Sonuçları\n")
    f.write(f"Analiz Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Analiz Edilen Görüntü: {relative_image_path}\n")
    f.write(f"Tam Yol: {image_path_str}\n")
    f.write("=" * 50 + "\n")
    for satir in analiz_sonuclari:
        f.write(satir + "\n")
    f.write(f"\nHistogram Grafiği: {histogram_filename}\n")

print(f"\nAnaliz sonuçları kaydedildi: {results_filename}")