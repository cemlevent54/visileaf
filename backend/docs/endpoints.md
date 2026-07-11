# Image Enhancement API Endpoints Dokümantasyonu

Bu dokümantasyon, Visileaf görüntü iyileştirme API'sinin 3 ana endpoint'ini detaylı olarak açıklar.

---

## 📋 İçindekiler

1. [POST /api/enhancement/enhance](#1-post-apienhancementenhance)
2. [POST /api/enhancement/enhance-with-dcp](#2-post-apienhancementenhance-with-dcp)
3. [POST /api/enhancement/dcp-guided-filter](#3-post-apienhancementdcp-guided-filter)

---

## 1. POST /api/enhancement/enhance

### Genel Bakış

Ana görüntü iyileştirme endpoint'i. Birden fazla iyileştirme yöntemini birleştirerek hibrit görüntü iyileştirme yapmanıza olanak sağlar. Tüm aktif yöntemler, belirtilen `order` sırasına göre sırayla uygulanır.

### İstek Formatı

- **Method**: `POST`
- **URL**: `/api/enhancement/enhance`
- **Content-Type**: `multipart/form-data`
- **Headers**:
  - `Authorization: Bearer <access_token>` (Zorunlu)

### İstek Parametreleri

| Parametre | Tip | Zorunlu | Açıklama |
|-----------|-----|---------|----------|
| `image` | File | ✅ | İyileştirilecek görüntü dosyası (JPEG, PNG, vb.) |
| `params_json` | String (JSON) | ✅ | İyileştirme parametreleri (JSON string formatında) |

### EnhancementParams Şeması

`params_json` içinde gönderilecek parametreler:

#### Temel İyileştirme Yöntemleri

| Parametre | Tip | Varsayılan | Alt Sınır | Üst Sınır | Açıklama |
|-----------|-----|------------|-----------|-----------|----------|
| `use_gamma` | boolean | `false` | - | - | Gamma düzeltme kullan |
| `gamma` | float | `0.5` | `> 0` | - | Gamma değeri (<1.0 aydınlatır, >1.0 karartır) |
| `use_clahe` | boolean | `false` | - | - | CLAHE kullan |
| `clahe_clip` | float | `3.0` | `> 0` | - | CLAHE kontrast sınırlama eşiği |
| `clahe_tile_size` | [int, int] | `[8, 8]` | `[>0, >0]` | - | CLAHE tile grid boyutu [genişlik, yükseklik] |
| `use_ssr` | boolean | `false` | - | - | Single-Scale Retinex kullan |
| `ssr_sigma` | int | `80` | `> 0` | - | SSR Gauss filtresi standart sapması |
| `use_msr` | boolean | `false` | - | - | Multi-Scale Retinex kullan |
| `msr_sigmas` | [int, ...] | `[15, 80, 250]` | Her eleman `> 0` | - | MSR sigma değerleri listesi (en az 1 eleman) |
| `use_sharpen` | boolean | `false` | - | - | Keskinleştirme kullan |
| `sharpen_method` | string | `"unsharp"` | - | - | Keskinleştirme yöntemi: `"unsharp"` veya `"laplacian"` |
| `sharpen_strength` | float | `1.0` | `> 0` | - | Keskinleştirme gücü (1.0 = normal, 2.0 = güçlü) |
| `sharpen_kernel_size` | int | `5` | `> 0` (tek sayı) | - | Unsharp method için kernel boyutu (tek sayı olmalı: 1, 3, 5, 7, 9, ...) |

#### Eğitimlik Filtreler (Educational Filters)

| Parametre | Tip | Varsayılan | Alt Sınır | Üst Sınır | Açıklama |
|-----------|-----|------------|-----------|-----------|----------|
| `use_negative` | boolean | `false` | - | - | Klasik negatif görüntü filtresi uygula |
| `use_threshold` | boolean | `false` | - | - | Binary eşikleme uygula (grayscale) |
| `threshold_value` | int | `128` | `0` | `255` | Binary eşikleme için eşik değeri |
| `use_gray_slice` | boolean | `false` | - | - | Gri seviye dilimleme uygula |
| `gray_slice_low` | int | `100` | `0` | `255` | Gri dilimleme alt sınırı (low <= high olmalı) |
| `gray_slice_high` | int | `180` | `0` | `255` | Gri dilimleme üst sınırı (low <= high olmalı) |
| `use_bitplane` | boolean | `false` | - | - | Bit-plane dilimleme uygula (grayscale) |
| `bitplane_bit` | int | `7` | `0` | `7` | Bit-plane bit indeksi (0-7 arası) |

#### Gürültü Giderme (Denoising)

| Parametre | Tip | Varsayılan | Alt Sınır | Üst Sınır | Açıklama |
|-----------|-----|------------|-----------|-----------|----------|
| `use_denoise` | boolean | `false` | - | - | Renk gürültülerini temizle (mavi/kırmızı lekeler) |
| `denoise_strength` | float | `3.0` | `> 0` | `20` | Gürültü giderme gücü (3.0 = hafif, 10.0 = güçlü) |

#### DCP Tabanlı Low-light İyileştirme

| Parametre | Tip | Varsayılan | Alt Sınır | Üst Sınır | Açıklama |
|-----------|-----|------------|-----------|-----------|----------|
| `use_dcp` | boolean | `false` | - | - | Dark Channel Prior (DCP) tabanlı low-light enhancement |
| `use_dcp_guided` | boolean | `false` | - | - | DCP + Guided Filter tabanlı gelişmiş low-light enhancement |

**Not**: `use_dcp` ve `use_dcp_guided` aynı anda `true` olamaz. Pipeline içinde sadece biri aktif olabilir.

#### Low-light İyileştirme (LIME/DUAL benzeri)

| Parametre | Tip | Varsayılan | Alt Sınır | Üst Sınır | Açıklama |
|-----------|-----|------------|-----------|-----------|----------|
| `use_lowlight_lime` | boolean | `false` | - | - | Low-light enhancement (LIME benzeri, illumination-map tabanlı) |
| `use_lowlight_dual` | boolean | `false` | - | - | Low-light enhancement (DUAL benzeri, under/over-exposed bölgeler için) |
| `lowlight_gamma` | float | `0.6` | `> 0` | - | Low-light gamma düzeltme parametresi |
| `lowlight_lambda` | float | `0.15` | `> 0` | - | İllumination refinement ağırlığı |
| `lowlight_sigma` | float | `3.0` | `> 0` | - | Gaussian ağırlıklar için spatial standart sapma |
| `lowlight_bc` | float | `1.0` | `>= 0` | - | Mertens kontrast ölçüsü ağırlığı |
| `lowlight_bs` | float | `1.0` | `>= 0` | - | Mertens doygunluk ölçüsü ağırlığı |
| `lowlight_be` | float | `1.0` | `>= 0` | - | Mertens well-exposedness ölçüsü ağırlığı |

#### İşlem Sırası (Order)

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `order` | [string, ...] | `null` | Aktif yöntemlerin uygulanma sırası. Örnek: `["gamma", "msr", "clahe", "sharpen"]` |

**Desteklenen method isimleri**:
- `"gamma"` - Gamma düzeltme
- `"clahe"` - CLAHE
- `"ssr"` - Single-Scale Retinex
- `"msr"` - Multi-Scale Retinex
- `"sharpen"` - Keskinleştirme
- `"negative"` - Negatif görüntü
- `"threshold"` - Binary eşikleme
- `"gray_slice"` - Gri seviye dilimleme
- `"bitplane"` - Bit-plane dilimleme
- `"denoise"` - Gürültü giderme
- `"dcp"` - Dark Channel Prior
- `"dcp_guided"` - DCP + Guided Filter
- `"lowlight_lime"` - Low-light (LIME benzeri)
- `"lowlight_dual"` - Low-light (DUAL benzeri)

### Order Mantığı

1. **Aktif Methodların Belirlenmesi**: `use_*` bayrakları `true` olan tüm yöntemler aktif hale gelir.

2. **Sıralama**:
   - Eğer `order` parametresi belirtilmişse:
     - `order` dizisindeki sıraya göre yöntemler sıralanır.
     - `order`'da belirtilmeyen ama aktif olan yöntemler, `order`'daki yöntemlerden sonra eklenir.
   - Eğer `order` belirtilmemişse (`null` veya boş):
     - Yöntemler varsayılan sırayla uygulanır (kod içindeki tanımlanma sırası).

3. **Uygulama**: Her yöntem sırayla uygulanır. Bir önceki yöntemin çıktısı, bir sonraki yöntemin girdisi olur.

**Örnek**:
```json
{
  "use_gamma": true,
  "gamma": 0.5,
  "use_clahe": true,
  "use_sharpen": true,
  "order": ["clahe", "gamma", "sharpen"]
}
```
Bu durumda işlem sırası: **CLAHE → Gamma → Sharpen**

### Örnek İstek

```bash
curl -X POST "http://127.0.0.1:8000/api/enhancement/enhance" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "image=@/path/to/image.jpg" \
  -F 'params_json={
    "use_gamma": true,
    "gamma": 0.5,
    "use_clahe": true,
    "clahe_clip": 2.5,
    "clahe_tile_size": [8, 8],
    "use_msr": true,
    "msr_sigmas": [15, 80, 250],
    "use_sharpen": true,
    "sharpen_method": "unsharp",
    "sharpen_strength": 1.5,
    "sharpen_kernel_size": 5,
    "order": ["gamma", "msr", "clahe", "sharpen"]
  }'
```

### Yanıt

- **Status Code**: `200 OK`
- **Content-Type**: `image/jpeg`
- **Body**: İyileştirilmiş görüntü (JPEG bytes)
- **Headers**:
  - `Content-Disposition: attachment; filename=enhanced_image.jpg`

### Hata Durumları

| Status Code | Açıklama |
|-------------|----------|
| `400` | Geçersiz parametreler (örn: gamma <= 0, boş msr_sigmas listesi) |
| `401` | Yetkilendirme hatası (token eksik veya geçersiz) |
| `500` | Sunucu hatası (görüntü işleme hatası, modül yükleme hatası) |

### Notlar

- Tüm görüntü işlemleri BGR formatında yapılır (OpenCV standardı).
- Çıktı her zaman JPEG formatındadır.
- İşlem sonuçları veritabanına kaydedilir (input/output görüntüler ve parametreler).
- `order` dizisinde belirtilmeyen ama aktif olan yöntemler, belirtilenlerden sonra eklenir.

---

## 2. POST /api/enhancement/enhance-with-dcp

### Genel Bakış

Dark Channel Prior (DCP) algoritması tabanlı low-light görüntü iyileştirme endpoint'i. İki modda çalışabilir:

1. **Standalone Mode**: Sadece DCP algoritmasını uygular (`params_json` gönderilmezse).
2. **Pipeline Mode**: DCP'yi diğer yöntemlerle birleştirerek pipeline içinde kullanır (`params_json` gönderilirse).

### İstek Formatı

- **Method**: `POST`
- **URL**: `/api/enhancement/enhance-with-dcp`
- **Content-Type**: `multipart/form-data`
- **Headers**:
  - `Authorization: Bearer <access_token>` (Zorunlu)

### İstek Parametreleri

| Parametre | Tip | Zorunlu | Açıklama |
|-----------|-----|---------|----------|
| `image` | File | ✅ | İyileştirilecek görüntü dosyası |
| `params_json` | String (JSON) | ❌ | İsteğe bağlı enhancement parametreleri |

### Çalışma Mantığı

#### Senaryo 1: `params_json` Gönderilmezse (Standalone Mode)

- Sadece DCP algoritması uygulanır.
- Diğer yöntemler kullanılmaz.
- `enhancement_type` = `"dcp"` olarak kaydedilir.

#### Senaryo 2: `params_json` Gönderilirse (Pipeline Mode)

- `params_json` içindeki `EnhancementParams` şeması kullanılır.
- **Önemli**: `use_dcp` otomatik olarak `true` yapılır, `use_dcp_guided` `false` yapılır.
- `order` parametresi korunur (frontend'den gelen sıra aynen kullanılır).
- Tüm aktif yöntemler (DCP dahil) belirtilen `order` sırasına göre uygulanır.
- `enhancement_type` = `"dcp_pipeline"` olarak kaydedilir.

### Örnek İstekler

#### Standalone Mode

```bash
curl -X POST "http://127.0.0.1:8000/api/enhancement/enhance-with-dcp" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "image=@/path/to/image.jpg"
```

#### Pipeline Mode

```bash
curl -X POST "http://127.0.0.1:8000/api/enhancement/enhance-with-dcp" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "image=@/path/to/image.jpg" \
  -F 'params_json={
    "use_gamma": true,
    "gamma": 0.5,
    "use_clahe": true,
    "clahe_clip": 2.5,
    "use_dcp": true,
    "order": ["dcp", "clahe", "gamma"]
  }'
```

**Not**: Pipeline mode'da `use_dcp` zaten `true` yapılır, ancak `params_json` içinde belirtilmesi önerilir.

### Yanıt

- **Status Code**: `200 OK`
- **Content-Type**: `image/jpeg`
- **Body**: İyileştirilmiş görüntü (JPEG bytes)
- **Headers**:
  - `Content-Disposition: attachment; filename=enhanced_image_dcp.jpg`

### Hata Durumları

| Status Code | Açıklama |
|-------------|----------|
| `400` | Geçersiz JSON formatı veya parametreler |
| `401` | Yetkilendirme hatası |
| `500` | DCP modülü yüklenemedi veya görüntü işleme hatası |

### Notlar

- DCP algoritması özellikle karanlık veya sisli görüntüler için tasarlanmıştır.
- Pipeline mode'da `order` parametresi tamamen korunur; backend tarafında manipüle edilmez.
- Standalone mode'da sadece DCP uygulanır, diğer yöntemler kullanılmaz.

---

## 3. POST /api/enhancement/dcp-guided-filter

### Genel Bakış

Dark Channel Prior (DCP) + Guided Filter algoritması tabanlı gelişmiş low-light görüntü iyileştirme endpoint'i. DCP'nin gelişmiş bir versiyonudur ve transmission map'i Guided Filter ile refine eder. İki modda çalışabilir:

1. **Standalone Mode**: Sadece DCP + Guided Filter algoritmasını uygular (`params_json` gönderilmezse).
2. **Pipeline Mode**: DCP + Guided Filter'ı diğer yöntemlerle birleştirerek pipeline içinde kullanır (`params_json` gönderilirse).

### İstek Formatı

- **Method**: `POST`
- **URL**: `/api/enhancement/dcp-guided-filter`
- **Content-Type**: `multipart/form-data`
- **Headers**:
  - `Authorization: Bearer <access_token>` (Zorunlu)

### İstek Parametreleri

| Parametre | Tip | Zorunlu | Açıklama |
|-----------|-----|---------|----------|
| `image` | File | ✅ | İyileştirilecek görüntü dosyası |
| `params_json` | String (JSON) | ❌ | İsteğe bağlı enhancement parametreleri |

### Çalışma Mantığı

#### Senaryo 1: `params_json` Gönderilmezse (Standalone Mode)

- Sadece DCP + Guided Filter algoritması uygulanır.
- Diğer yöntemler kullanılmaz.
- `enhancement_type` = `"dcp_guided"` olarak kaydedilir.

#### Senaryo 2: `params_json` Gönderilirse (Pipeline Mode)

- `params_json` içindeki `EnhancementParams` şeması kullanılır.
- **Önemli**: `use_dcp_guided` otomatik olarak `true` yapılır, `use_dcp` `false` yapılır.
- `order` parametresi korunur (frontend'den gelen sıra aynen kullanılır).
- Tüm aktif yöntemler (DCP + Guided Filter dahil) belirtilen `order` sırasına göre uygulanır.
- `enhancement_type` = `"dcp_guided_pipeline"` olarak kaydedilir.

### Örnek İstekler

#### Standalone Mode

```bash
curl -X POST "http://127.0.0.1:8000/api/enhancement/dcp-guided-filter" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "image=@/path/to/image.jpg"
```

#### Pipeline Mode

```bash
curl -X POST "http://127.0.0.1:8000/api/enhancement/dcp-guided-filter" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "image=@/path/to/image.jpg" \
  -F 'params_json={
    "use_gamma": true,
    "gamma": 0.5,
    "use_clahe": true,
    "clahe_clip": 2.5,
    "use_dcp_guided": true,
    "order": ["dcp_guided", "clahe", "gamma"]
  }'
```

**Not**: Pipeline mode'da `use_dcp_guided` zaten `true` yapılır, ancak `params_json` içinde belirtilmesi önerilir.

### Yanıt

- **Status Code**: `200 OK`
- **Content-Type**: `image/jpeg`
- **Body**: İyileştirilmiş görüntü (JPEG bytes)
- **Headers**:
  - `Content-Disposition: attachment; filename=enhanced_image_dcp_guided.jpg`

### Hata Durumları

| Status Code | Açıklama |
|-------------|----------|
| `400` | Geçersiz JSON formatı veya parametreler |
| `401` | Yetkilendirme hatası |
| `500` | DCP modülü yüklenemedi veya görüntü işleme hatası |

### Notlar

- DCP + Guided Filter, DCP'nin gelişmiş bir versiyonudur ve transmission map'i Guided Filter ile refine eder.
- Genellikle standalone DCP'den daha iyi sonuçlar verir, ancak daha yavaştır.
- Pipeline mode'da `order` parametresi tamamen korunur; backend tarafında manipüle edilmez.
- Standalone mode'da sadece DCP + Guided Filter uygulanır, diğer yöntemler kullanılmaz.

---

## 🔄 Order Parametresi Detaylı Açıklama

### Genel Mantık

`order` parametresi, aktif yöntemlerin uygulanma sırasını kontrol eder. Bu parametre tüm 3 endpoint'te de aynı şekilde çalışır.

### Algoritma

1. **Aktif Yöntemlerin Toplanması**: `use_*` bayrakları `true` olan tüm yöntemler bir listeye eklenir.

2. **Sıralama**:
   ```python
   if order and methods_to_apply:
       ordered_methods = []
       # 1. Order'daki sıraya göre yöntemleri ekle
       for method_name in order:
           for method in methods_to_apply:
               if method[0] == method_name:
                   ordered_methods.append(method)
                   break
       # 2. Order'da olmayan ama aktif olan yöntemleri sona ekle
       for method in methods_to_apply:
           if method not in ordered_methods:
               ordered_methods.append(method)
       methods_to_apply = ordered_methods
   ```

3. **Uygulama**: Her yöntem sırayla uygulanır. Bir önceki yöntemin çıktısı, bir sonraki yöntemin girdisi olur.

### Örnek Senaryolar

#### Senaryo 1: Order Belirtilmiş

```json
{
  "use_gamma": true,
  "use_clahe": true,
  "use_sharpen": true,
  "order": ["clahe", "gamma", "sharpen"]
}
```

**Sonuç**: CLAHE → Gamma → Sharpen

#### Senaryo 2: Order Kısmen Belirtilmiş

```json
{
  "use_gamma": true,
  "use_clahe": true,
  "use_sharpen": true,
  "use_denoise": true,
  "order": ["denoise", "clahe"]
}
```

**Sonuç**: Denoise → CLAHE → Gamma → Sharpen

(Order'da belirtilmeyen `gamma` ve `sharpen` sona eklenir)

#### Senaryo 3: Order Boş veya Null

```json
{
  "use_gamma": true,
  "use_clahe": true,
  "use_sharpen": true,
  "order": null
}
```

**Sonuç**: Kod içindeki tanımlanma sırasına göre (varsayılan sıra)

#### Senaryo 4: DCP Pipeline ile Order

```json
{
  "use_dcp": true,
  "use_clahe": true,
  "use_gamma": true,
  "order": ["dcp", "clahe", "gamma"]
}
```

**Sonuç**: DCP → CLAHE → Gamma

### Önemli Notlar

- `order` dizisinde belirtilen ama aktif olmayan (use_* = false) yöntemler göz ardı edilir.
- `order` dizisinde belirtilmeyen ama aktif olan yöntemler, belirtilenlerden sonra eklenir.
- Aynı yöntem `order` içinde birden fazla kez belirtilirse, sadece ilk geçtiği yerde uygulanır.
- DCP endpoint'lerinde (`/enhance-with-dcp`, `/dcp-guided-filter`) `order` parametresi tamamen korunur; backend tarafında manipüle edilmez.

---

## 📊 Parametre Sınırları Özet Tablosu

| Parametre | Tip | Alt Sınır | Üst Sınır | Varsayılan |
|-----------|-----|-----------|-----------|------------|
| `gamma` | float | `> 0` | - | `0.5` |
| `clahe_clip` | float | `> 0` | - | `3.0` |
| `clahe_tile_size` | [int, int] | `[>0, >0]` | - | `[8, 8]` |
| `ssr_sigma` | int | `> 0` | - | `80` |
| `msr_sigmas` | [int, ...] | Her eleman `> 0` | - | `[15, 80, 250]` |
| `sharpen_strength` | float | `> 0` | - | `1.0` |
| `sharpen_kernel_size` | int | `> 0` (tek sayı) | - | `5` |
| `threshold_value` | int | `0` | `255` | `128` |
| `gray_slice_low` | int | `0` | `255` | `100` |
| `gray_slice_high` | int | `0` | `255` | `180` |
| `bitplane_bit` | int | `0` | `7` | `7` |
| `denoise_strength` | float | `> 0` | `20` | `3.0` |
| `lowlight_gamma` | float | `> 0` | - | `0.6` |
| `lowlight_lambda` | float | `> 0` | - | `0.15` |
| `lowlight_sigma` | float | `> 0` | - | `3.0` |
| `lowlight_bc` | float | `>= 0` | - | `1.0` |
| `lowlight_bs` | float | `>= 0` | - | `1.0` |
| `lowlight_be` | float | `>= 0` | - | `1.0` |

---

## 🔍 Desteklenen Method İsimleri (Order İçin)

| Method İsmi | Açıklama |
|------------|----------|
| `"gamma"` | Gamma düzeltme |
| `"clahe"` | CLAHE (Contrast Limited Adaptive Histogram Equalization) |
| `"ssr"` | Single-Scale Retinex |
| `"msr"` | Multi-Scale Retinex |
| `"sharpen"` | Keskinleştirme |
| `"negative"` | Negatif görüntü filtresi |
| `"threshold"` | Binary eşikleme |
| `"gray_slice"` | Gri seviye dilimleme |
| `"bitplane"` | Bit-plane dilimleme |
| `"denoise"` | Gürültü giderme |
| `"dcp"` | Dark Channel Prior |
| `"dcp_guided"` | DCP + Guided Filter |
| `"lowlight_lime"` | Low-light enhancement (LIME benzeri) |
| `"lowlight_dual"` | Low-light enhancement (DUAL benzeri) |

---

## 📝 Genel Notlar

1. **Kimlik Doğrulama**: Tüm endpoint'ler JWT token gerektirir (`Authorization: Bearer <token>`).

2. **Veritabanı Kaydı**: Tüm işlemler veritabanına kaydedilir:
   - Input görüntü: `uploads/YYYY_MM_DD_HH_MM_SS/input.{ext}`
   - Output görüntü: `uploads/YYYY_MM_DD_HH_MM_SS/output.{ext}`
   - Parametreler: JSON formatında `params` alanında saklanır.

3. **Görüntü Formatları**: 
   - Giriş: JPEG, PNG, vb. (OpenCV desteklediği tüm formatlar)
   - Çıkış: Her zaman JPEG

4. **Hata Yönetimi**: 
   - Parametre validasyonu yapılır (alt/üst sınırlar kontrol edilir).
   - Geçersiz parametreler için `400 Bad Request` döner.
   - Görüntü işleme hataları için `500 Internal Server Error` döner.

5. **Performans**: 
   - DCP ve DCP + Guided Filter algoritmaları CPU yoğun işlemlerdir.
   - Büyük görüntüler için işlem süresi artabilir.
   - Pipeline mode'da birden fazla yöntem uygulanacağı için süre daha da artar.

---

## 📚 İlgili Dokümantasyon

- [ENHANCEMENT_DOCS.md](./ENHANCEMENT_DOCS.md) - Genel enhancement modülü dokümantasyonu
- [commands.md](./commands.md) - Backend komutları ve kurulum

