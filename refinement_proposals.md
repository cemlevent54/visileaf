# Visileaf Projesi: İyileştirme ve Optimizasyon Önerileri (Refinement Proposals)

Bu doküman, **Visileaf** uygulamasının arayüz, görüntü işleme motoru ve arka plan altyapısında gerçekleştirilebilecek iyileştirmeleri ve optimizasyon önerilerini teknik detayları, kod taslakları ve kütüphane tavsiyeleri ile birlikte sunmaktadır.

---

## 1. Arayüz (UI/UX) İyileştirmeleri

### 📷 1.1. Split-Screen (Öncesi / Sonrası) Karşılaştırma Slider'ı
Mevcut arayüzde girdi ve çıktı görselleri yan yana iki ayrı kutuda statik olarak gösterilmektedir. Kullanıcının iki görüntünün farkını piksel seviyesinde görebilmesi için interaktif bir slider yapısı kurulmalıdır.

#### Teknoloji Tavsiyesi:
* Kütüphane kullanmak isterseniz: `react-compare-image`
* Kütüphanesiz (Sıfırdan CSS/JS ile):
  ```tsx
  // SplitSlider.tsx - Örnek Taslak
  import React, { useState } from 'react';
  import './SplitSlider.css';

  interface SplitSliderProps {
    leftImage: string;
    rightImage: string;
  }

  export const SplitSlider: React.FC<SplitSliderProps> = ({ leftImage, rightImage }) => {
    const [sliderPosition, setSliderPosition] = useState(50);

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setSliderPosition(Number(e.target.value));
    };

    return (
      <div className="split-slider-container">
        <div className="image-wrapper left-image" style={{ width: `${sliderPosition}%` }}>
          <img src={leftImage} alt="Orijinal" />
        </div>
        <div className="image-wrapper right-image">
          <img src={rightImage} alt="İyileştirilmiş" />
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPosition}
          onChange={handleSliderChange}
          className="slider-bar"
        />
      </div>
    );
  };
  ```
  ```css
  /* SplitSlider.css */
  .split-slider-container {
    position: relative;
    width: 100%;
    height: 400px;
    overflow: hidden;
  }
  .image-wrapper {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    overflow: hidden;
  }
  .image-wrapper img {
    height: 100%;
    object-fit: cover;
  }
  .right-image {
    width: 100%;
    z-index: 1;
  }
  .left-image {
    z-index: 2;
  }
  .slider-bar {
    position: absolute;
    z-index: 3;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    background: transparent;
    appearance: none;
    outline: none;
    cursor: ew-resize;
  }
  ```

### 📊 1.2. Canlı RGB / Grayscale Histogram Grafiği
Resimlerin aydınlatma ve histogram eşitleme adımlarından sonraki piksel dağılımlarını bilimsel olarak göstermek için histogram grafikleri çizilmelidir.

#### Tasarım Adımları:
1. Backend, görsel işleme sonrasında görüntünün 256 kanallı renk dağılımını (RGB histogram) hesaplar:
   ```python
   # histogram_helper.py
   def calculate_histogram(img):
       # BGR kanallarını ayır
       b_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().tolist()
       g_hist = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten().tolist()
       r_hist = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten().tolist()
       return {"red": r_hist, "green": g_hist, "blue": b_hist}
   ```
2. Bu veri JSON olarak HTTP yanıtına eklenir.
3. Frontend tarafında `recharts` veya `chart.js` kütüphanesi kullanılarak alan grafiği (Area Chart) şeklinde çizilir.

---

## 2. Görüntü İşleme (Algoritmik) İyileştirmeleri

### ⚡ 2.1. LIME Optimizasyonu: Downsampling Piramidi
LIME algoritmasında aydınlatma haritasının optimizasyonu çözülürken doğrusal sistem çözücü (`spsolve`), yüksek çözünürlüklü (örn. 4K) görüntülerde sunucu işlemcisini bloke eder ve RAM yetersizliğinden dolayı çökmelere yol açar.

#### Çözüm Yaklaşımı:
Resmin aydınlatma bileşeni düşük frekanslı (pürüzsüz) bir yapıya sahip olduğundan, optimizasyon küçük çözünürlükte çözülüp kenar koruyucu bir filtreyle büyütülebilir.

```python
# LIME optimizasyon hızlandırma taslağı
def _refine_illumination_map_optimized(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray) -> np.ndarray:
    h, w = L.shape
    # 1. Çözünürlüğü küçült (Hızlandırma ve bellek tasarrufu)
    scale = 0.25 # Çözünürlüğü %25'e düşür
    h_small, w_small = int(h * scale), int(w * scale)
    L_small = cv2.resize(L, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    # 2. Optimizasyonu küçük resimde çöz
    # (Mevcut _refine_illumination_map_linear fonksiyonu L_small ile çağrılır)
    L_refined_small = _refine_illumination_map_linear_core(L_small, gamma, lambda_, kernel)
    
    # 3. Sonucu kenar koruyarak orijinal boyutuna büyüt (Guided Filter ile)
    L_refined_large = cv2.resize(L_refined_small, (w, h), interpolation=cv2.INTER_CUBIC)
    L_refined_guided = cv2.ximgproc.guidedFilter(
        guide=L.astype(np.float32),
        src=L_refined_large.astype(np.float32),
        radius=15,
        eps=1e-3
    )
    
    return np.clip(L_refined_guided, 1e-3, 1.0)
```

### 🌈 2.2. Retinex Renk Restorasyonu (MSRCR)
SSR ve MSR algoritmaları, kontrastı artırırken renk kanalları arasındaki dengesizlikten dolayı renklerin grileşmesine veya solmasına yol açabilir. Bu durumu düzeltmek için renk restorasyon faktörü (MSRCR) eklenmelidir.

#### Formülasyon:
$$R_{MSRCR_i}(x,y) = C_i(x,y) R_{MSR_i}(x,y)$$
$$C_i(x,y) = \beta \log \left( \alpha \frac{I_i(x,y)}{\sum_{k=1}^3 I_k(x,y)} \right)$$

* Burada $C_i(x,y)$ renk restorasyon fonksiyonudur.
* $I_i(x,y)$, $i$. kanalın yoğunluğudur.
* $\alpha$ ve $\beta$ renk doygunluğunu kontrol eden sabitlerdir (genellikle $\alpha=125$, $\beta=46$ tercih edilir).

---

## 3. Altyapı ve Dağıtım (Backend & DevOps) İyileştirmeleri

### 📦 3.1. PyTorch Modellerinin ONNX Formatına Dönüştürülmesi
Sunucu tarafında PyTorch bağımlılığını kaldırmak, Docker imaj boyutunu küçültmek ve CPU çıkarım hızını artırmak için derin öğrenme modelleri ONNX formatına aktarılmalıdır.

#### ONNX Export İşlemi (Local'de bir kez çalıştırılacak):
```python
import torch
import model as zero_dce_model

# 1. Modeli yükle
DCE_net = zero_dce_model.enhance_net_nopool().cpu()
DCE_net.load_state_dict(torch.load("Zero-DCE_weight.pth", map_location="cpu"))
DCE_net.eval()

# 2. Sahte girdi oluştur (Batch=1, Kanal=3, Yükseklik=512, Genişlik=512)
dummy_input = torch.randn(1, 3, 512, 512)

# 3. ONNX formatına aktar (Dinamik eksenler ekleyerek değişken çözünürlükleri destekleyin)
torch.onnx.export(
    DCE_net,
    dummy_input,
    "Zero-DCE.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
```

#### FastAPI Backend Çıkarım (Inference) Değişikliği:
Artık `torch` kütüphanesini import etmenize gerek kalmaz, sadece `onnxruntime` yeterlidir:
```python
import onnxruntime as ort
import numpy as np

# Modeli yükle
ort_session = ort.InferenceSession("Zero-DCE.onnx")

def run_zero_dce_onnx(img_numpy: np.ndarray) -> np.ndarray:
    # img_numpy: [1, 3, H, W] boyutunda ve float32 [0, 1] aralığında olmalıdır
    ort_inputs = {ort_session.get_inputs()[0].name: img_numpy}
    ort_outs = ort_session.run(None, ort_inputs)
    enhanced = ort_outs[1] # Model çıktısı
    return enhanced
```

### 🚦 3.2. Asenkron İş Kuyruğu ve İlerleme Göstergesi
Büyük resimlerin işlenmesi ve ağır derin öğrenme çıkarımları (LLFlow/MIRNetv2) HTTP bağlantısının zaman aşımına (timeout) uğramasına neden olabilir.

#### Önerilen Yapı:
1. **İstek Aşaması:** Kullanıcı resmi yükler ve `/api/enhance/async` adresine istek gönderir. Backend, görevi asenkron olarak başlatır ve veritabanına `status="processing"` olarak kaydeder, hemen ardından `{"task_id": "uuid"}` döner.
2. **FastAPI BackgroundTasks kullanımı:**
   ```python
   from fastapi import BackgroundTasks

   @app.post("/enhance/async")
   def enhance_async(file: UploadFile, background_tasks: BackgroundTasks):
       task_id = uuid.uuid4()
       # Görevi arka planda başlat
       background_tasks.add_task(process_heavy_image, task_id, file.file.read())
       return {"task_id": task_id, "status": "queued"}
   ```
3. **Frontend Polling:** Frontend, 1-2 saniye aralıklarla `/api/tasks/{task_id}` adresine istek atarak durum kontrolü yapar. İşlem tamamlandığında (`status="completed"`) resmi indirir ve ekrana basar. Bu sayede tarayıcı/sunucu bağlantısı asla kopmaz.
