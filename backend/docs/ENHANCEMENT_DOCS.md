# Görüntü İyileştirme Modülü Dokümantasyonu

## Genel Bakış

`enchancement.py` modülü, birden fazla görüntü iyileştirme tekniğini birleştirerek hibrit görüntü iyileştirme yapmanıza olanak sağlar. Tek bir görüntü üzerinde farklı yöntemleri sıralı olarak uygulayabilir ve sonuçları görselleştirip kaydedebilirsiniz.

## Desteklenen İyileştirme Yöntemleri

### 1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Ne yapar?** Kontrastı artırır, özellikle karanlık bölgeleri aydınlatır
- **Parametreler:**
  - `clip_limit`: Kontrast sınırlama eşiği (varsayılan: 3.0)
  - `tile_size`: Izgara boyutu (varsayılan: 8x8)

### 2. **Gamma Düzeltme**
- **Ne yapar?** Görüntünün parlaklığını ayarlar
- **Parametreler:**
  - `gamma`: Gamma değeri
    - `< 1.0`: Görüntüyü aydınlatır (karanlık görüntüler için)
    - `> 1.0`: Görüntüyü karartır (parlak görüntüler için)
    - Varsayılan: 0.5

### 3. **SSR** (Single-Scale Retinex)
- **Ne yapar?** Tek ölçekli Retinex ile renk dengesini iyileştirir
- **Parametreler:**
  - `sigma`: Gauss filtresi standart sapması (varsayılan: 80)

### 4. **MSR** (Multi-Scale Retinex)
- **Ne yapar?** Çok ölçekli Retinex ile daha gelişmiş renk dengesi sağlar
- **Parametreler:**
  - `sigmas`: Sigma değerleri listesi (varsayılan: [15, 80, 250])

### 5. **Sharpening** (Kenar Netleştirme)
- **Ne yapar?** Kenarları netleştirir, detayları belirginleştirir
- **Yöntemler:**
  - `unsharp`: Unsharp masking (varsayılan)
  - `laplacian`: Laplacian filtre
- **Parametreler:**
  - `strength`: Netleştirme gücü (varsayılan: 1.0, 2.0 = güçlü)
  - `kernel_size`: Kernel boyutu - unsharp için (varsayılan: 5)

## Kullanım Şekilleri

### 1. Komut Satırından Kullanım

#### Temel Kullanım
```bash
# Sadece CLAHE uygula
python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-clahe

# CLAHE + Gamma kombinasyonu
python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-clahe --clahe-clip 3.0 --use-gamma --gamma 0.5

# MSR + CLAHE kombinasyonu
python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-msr --msr-sigmas 15 80 250 --use-clahe --clahe-clip 2.5
```

#### Sıralama Belirterek Kullanım
```bash
# Yöntemlerin uygulanma sırasını belirle (Gamma → MSR → CLAHE)
python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-gamma --gamma 0.6 --use-msr --msr-sigmas 15 80 250 --use-clahe --order gamma msr clahe

# Karartma + Netleştirme
python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-gamma --gamma 2.0 --use-sharpen --sharpen-strength 2.0 --order gamma sharpen
```

#### Hazır Örnekler
```bash
# Örnek 1: CLAHE + Gamma
python enchancement.py --example 1

# Örnek 2: MSR + CLAHE
python enchancement.py --example 2

# Örnek 3: Tam Hibrit (Gamma → MSR → CLAHE)
python enchancement.py --example 3

# Örnek 4: Sadece SSR
python enchancement.py --example 4
```

#### Sonuçları Kaydetmeden Kullanım
```bash
# Sadece ekranda göster, dosyaya kaydetme
python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-clahe --no-save
```

### 2. Python Kodundan Kullanım

```python
from enchancement import hybrid_enhancement
from pathlib import Path

# Görüntü yolunu belirle
image_path = Path("dataset/F/set1/2024-05-18/image.jpg")

# CLAHE + Gamma kombinasyonu
processed_img = hybrid_enhancement(
    image_path=image_path,
    use_clahe=True,
    clahe_clip_limit=3.0,
    clahe_tile_size=(8, 8),
    use_gamma=True,
    gamma_value=0.5,
    combination_order=['clahe', 'gamma'],
    save_results=True
)

# MSR + CLAHE kombinasyonu
processed_img = hybrid_enhancement(
    image_path=image_path,
    use_msr=True,
    msr_sigmas=[15, 80, 250],
    use_clahe=True,
    clahe_clip_limit=2.5,
    combination_order=['msr', 'clahe'],
    save_results=True
)
```

## Çıktılar

Modül çalıştırıldığında şunları üretir:

1. **Ekranda Görselleştirme**: Orijinal ve işlenmiş görüntüyü yan yana gösterir
2. **Karşılaştırma Grafiği**: `python/results/hybrid_comparison_[timestamp].png`
3. **İyileştirilmiş Görüntü**: `python/results/hybrid_output_[timestamp].jpg`
4. **İşlem Bilgileri**: `python/results/hybrid_info_[timestamp].txt`

## Önemli Notlar

- Yöntemlerin uygulanma sırası önemlidir! Farklı sıralar farklı sonuçlar üretir
- `combination_order` parametresi ile yöntemlerin sırasını kontrol edebilirsiniz
- Birden fazla yöntemi birleştirerek daha iyi sonuçlar elde edebilirsiniz
- Her yöntem için parametreleri ayarlayarak görüntüye özel optimizasyon yapabilirsiniz

## Komut Satırı Parametreleri Özeti

| Parametre | Açıklama |
|-----------|----------|
| `--image`, `-i` | İşlenecek görüntü yolu |
| `--use-clahe` | CLAHE kullan |
| `--clahe-clip` | CLAHE clip limit (varsayılan: 3.0) |
| `--clahe-tile` | CLAHE tile size (varsayılan: 8 8) |
| `--use-gamma` | Gamma düzeltme kullan |
| `--gamma` | Gamma değeri (varsayılan: 0.5) |
| `--use-ssr` | SSR kullan |
| `--ssr-sigma` | SSR sigma (varsayılan: 80) |
| `--use-msr` | MSR kullan |
| `--msr-sigmas` | MSR sigma listesi (varsayılan: 15 80 250) |
| `--use-sharpen` | Kenar netleştirme kullan |
| `--sharpen-method` | Netleştirme yöntemi (unsharp/laplacian) |
| `--sharpen-strength` | Netleştirme gücü (varsayılan: 1.0) |
| `--sharpen-kernel` | Kernel boyutu (varsayılan: 5) |
| `--order` | Yöntemlerin uygulanma sırası |
| `--no-save` | Sonuçları kaydetme |
| `--example` | Hazır örnek kullan (1-4) |

