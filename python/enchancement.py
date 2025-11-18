import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import argparse
from datetime import datetime
from pathlib import Path

# Tüm görüntü iyileştirme yöntemlerini import et
# Not: Dosya adlarındaki tire nedeniyle import kullanımı için alternatif yöntem

# CLAHE ve Gamma için direkt import
from claheequalization import apply_clahe
from gammacorrection import apply_gamma_correction

# Retinex fonksiyonları için dosyaları dinamik olarak yükle
script_dir = Path(__file__).parent.resolve()

# Single-scale retinex
ssr_spec = importlib.util.spec_from_file_location("single_scale_retinex", 
                                                   script_dir / "single-scale-retinex.py")
ssr_module = importlib.util.module_from_spec(ssr_spec)
ssr_spec.loader.exec_module(ssr_module)
single_scale_retinex = ssr_module.single_scale_retinex

# Multi-scale retinex
msr_spec = importlib.util.spec_from_file_location("mutli_scale_retinex", 
                                                   script_dir / "mutli-scale-retinex.py")
msr_module = importlib.util.module_from_spec(msr_spec)
msr_spec.loader.exec_module(msr_module)
multi_scale_retinex = msr_module.multi_scale_retinex


def apply_clahe_to_image(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE'yi numpy array üzerinde uygular (dosya yüklemeden).
    
    Args:
        img: BGR formatında görüntü (numpy array)
        clip_limit: Kontrast sınırlama eşiği
        tile_grid_size: Izgara boyutu
    
    Returns:
        İşlenmiş görüntü
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img


def apply_gamma_to_image(img, gamma=0.5):
    """
    Gamma düzeltmeyi numpy array üzerinde uygular.
    
    Args:
        img: BGR formatında görüntü (numpy array)
        gamma: Gamma değeri
    
    Returns:
        İşlenmiş görüntü
    """
    table = np.array([((i / 255.0) ** gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_img = cv2.LUT(img, table)
    return gamma_corrected_img


def apply_ssr_to_image(img, sigma=80):
    """
    SSR'yi numpy array üzerinde uygular.
    
    Args:
        img: BGR formatında görüntü (numpy array)
        sigma: Gauss filtresi standart sapması
    
    Returns:
        İşlenmiş görüntü
    """
    b, g, r = cv2.split(img)
    b_retinex = single_scale_retinex(b, sigma)
    g_retinex = single_scale_retinex(g, sigma)
    r_retinex = single_scale_retinex(r, sigma)
    ssr_output = cv2.merge([b_retinex, g_retinex, r_retinex])
    return ssr_output


def apply_msr_to_image(img, sigma_list=[15, 80, 250]):
    """
    MSR'yi numpy array üzerinde uygular.
    
    Args:
        img: BGR formatında görüntü (numpy array)
        sigma_list: Sigma değerleri listesi
    
    Returns:
        İşlenmiş görüntü
    """
    b, g, r = cv2.split(img)
    b_msr = multi_scale_retinex(b, sigma_list)
    g_msr = multi_scale_retinex(g, sigma_list)
    r_msr = multi_scale_retinex(r, sigma_list)
    msr_output = cv2.merge([b_msr, g_msr, r_msr])
    return msr_output


def apply_sharpen_to_image(img, method='unsharp', strength=1.0, kernel_size=5):
    """
    Kenar netleştirme (sharpening) uygular.
    
    Args:
        img: BGR formatında görüntü (numpy array)
        method: Netleştirme yöntemi ('unsharp' veya 'laplacian')
        strength: Netleştirme gücü (1.0 = normal, 2.0 = güçlü)
        kernel_size: Kernel boyutu (unsharp için)
    
    Returns:
        İşlenmiş görüntü
    """
    if method == 'unsharp':
        # Unsharp Masking yöntemi
        # 1. Görüntüyü blur'le (yumuşat)
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # 2. Orijinal görüntüden blur'ü çıkar (kenarları bul)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        
        # Değerleri [0, 255] aralığında tut
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
    elif method == 'laplacian':
        # Laplacian filtre ile netleştirme
        # Laplacian kernel (kenar tespiti)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * strength
        
        sharpened = cv2.filter2D(img, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
    else:
        sharpened = img.copy()
    
    return sharpened


def hybrid_enhancement(image_path, 
                      # CLAHE parametreleri
                      use_clahe=False, clahe_clip_limit=3.0, clahe_tile_size=(8, 8),
                      # Gamma parametreleri
                      use_gamma=False, gamma_value=0.5,
                      # SSR parametreleri
                      use_ssr=False, ssr_sigma=80,
                      # MSR parametreleri
                      use_msr=False, msr_sigmas=[15, 80, 250],
                      # Sharpening parametreleri
                      use_sharpen=False, sharpen_method='unsharp', sharpen_strength=1.0, sharpen_kernel_size=5,
                      # Kombinasyon parametreleri
                      combination_order=['clahe', 'gamma', 'ssr', 'msr', 'sharpen'],
                      save_results=True):
    """
    Birden fazla görüntü iyileştirme yöntemini hibrit şekilde uygular.
    
    Args:
        image_path: Görüntü dosyası yolu
        use_clahe: CLAHE kullanılsın mı?
        clahe_clip_limit: CLAHE clip limit değeri
        clahe_tile_size: CLAHE tile grid size
        use_gamma: Gamma düzeltme kullanılsın mı?
        gamma_value: Gamma değeri (>1.0 karartır, <1.0 aydınlatır)
        use_ssr: SSR kullanılsın mı?
        ssr_sigma: SSR sigma değeri
        use_msr: MSR kullanılsın mı?
        msr_sigmas: MSR sigma değerleri listesi
        use_sharpen: Kenar netleştirme kullanılsın mı?
        sharpen_method: Netleştirme yöntemi ('unsharp' veya 'laplacian')
        sharpen_strength: Netleştirme gücü (1.0 = normal, 2.0 = güçlü)
        sharpen_kernel_size: Kernel boyutu (unsharp için)
        combination_order: Yöntemlerin uygulanma sırası
        save_results: Sonuçları kaydet
    
    Returns:
        İşlenmiş görüntü
    """
    # Script'in bulunduğu klasörü baz al
    script_dir = Path(__file__).parent.resolve()
    
    # Path objesi ise string'e çevir
    if isinstance(image_path, Path):
        image_path_obj = image_path
        image_path_str = str(image_path)
    else:
        image_path_obj = Path(image_path)
        image_path_str = image_path
    
    # Görüntüyü yükle
    try:
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path_str}")
        
        original_img = cv2.imread(image_path_str)
        if original_img is None:
            raise FileNotFoundError(f"Görüntü dosyası yüklenemedi: {image_path_str}")
    except FileNotFoundError as e:
        print(e)
        return None
    
    # İşleme sırasını belirle
    methods_to_apply = []
    if use_clahe:
        methods_to_apply.append(('clahe', {'clip_limit': clahe_clip_limit, 'tile_size': clahe_tile_size}))
    if use_gamma:
        methods_to_apply.append(('gamma', {'gamma': gamma_value}))
    if use_ssr:
        methods_to_apply.append(('ssr', {'sigma': ssr_sigma}))
    if use_msr:
        methods_to_apply.append(('msr', {'sigmas': msr_sigmas}))
    if use_sharpen:
        methods_to_apply.append(('sharpen', {'method': sharpen_method, 'strength': sharpen_strength, 'kernel_size': sharpen_kernel_size}))
    
    # Sıralamayı uygula (varsayılan sıraya göre)
    if combination_order and methods_to_apply:
        ordered_methods = []
        for method_name in combination_order:
            for method in methods_to_apply:
                if method[0] == method_name:
                    ordered_methods.append(method)
                    break
        # Sıralamada olmayanları sona ekle
        for method in methods_to_apply:
            if method not in ordered_methods:
                ordered_methods.append(method)
        methods_to_apply = ordered_methods
    
    # Görüntüyü işle
    processed_img = original_img.copy()
    applied_methods = []
    
    print("\n=== Hibrit Görüntü İyileştirme ===")
    print(f"Orijinal görüntü: {image_path_str}")
    print(f"Uygulanacak yöntemler: {[m[0].upper() for m in methods_to_apply]}")
    
    for method_name, params in methods_to_apply:
        print(f"\nUygulanıyor: {method_name.upper()}")
        
        if method_name == 'clahe':
            processed_img = apply_clahe_to_image(
                processed_img, 
                clip_limit=params['clip_limit'], 
                tile_grid_size=params['tile_size']
            )
            applied_methods.append(f"CLAHE(clip={params['clip_limit']}, tile={params['tile_size']})")
            
        elif method_name == 'gamma':
            processed_img = apply_gamma_to_image(processed_img, gamma=params['gamma'])
            applied_methods.append(f"Gamma(γ={params['gamma']})")
            
        elif method_name == 'ssr':
            processed_img = apply_ssr_to_image(processed_img, sigma=params['sigma'])
            applied_methods.append(f"SSR(σ={params['sigma']})")
            
        elif method_name == 'msr':
            processed_img = apply_msr_to_image(processed_img, sigma_list=params['sigmas'])
            applied_methods.append(f"MSR(σ={params['sigmas']})")
            
        elif method_name == 'sharpen':
            processed_img = apply_sharpen_to_image(
                processed_img,
                method=params['method'],
                strength=params['strength'],
                kernel_size=params['kernel_size']
            )
            applied_methods.append(f"Sharpen({params['method']}, strength={params['strength']})")
    
    # Sonuçları görselleştir
    fig = plt.figure(figsize=(16, 8))
    
    # Orijinal görüntü
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal Görüntü", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # İşlenmiş görüntü
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    methods_str = " → ".join(applied_methods) if applied_methods else "Hiçbiri"
    plt.title(f"Hibrit İyileştirme\n{methods_str}", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Sonuçları kaydet
    if save_results:
        results_dir = script_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = str(results_dir / f'hybrid_comparison_{timestamp}.png')
        output_image_filename = str(results_dir / f'hybrid_output_{timestamp}.jpg')
        info_filename = str(results_dir / f'hybrid_info_{timestamp}.txt')
        
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"\nKarşılaştırma grafiği kaydedildi: {comparison_filename}")
        
        cv2.imwrite(output_image_filename, processed_img)
        print(f"İyileştirilmiş görüntü kaydedildi: {output_image_filename}")
        
        # Bilgi dosyasını kaydet
        with open(info_filename, 'w', encoding='utf-8') as f:
            f.write(f"Hibrit Görüntü İyileştirme Bilgileri\n")
            f.write(f"İşlem Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Orijinal Görüntü: {image_path_str}\n")
            f.write("=" * 60 + "\n")
            f.write(f"\nUygulanan Yöntemler ve Parametreler:\n")
            f.write(f"{'=' * 60}\n")
            
            if use_clahe:
                f.write(f"\n✓ CLAHE (Contrast Limited Adaptive Histogram Equalization)\n")
                f.write(f"  - Clip Limit: {clahe_clip_limit}\n")
                f.write(f"  - Tile Grid Size: {clahe_tile_size}\n")
            
            if use_gamma:
                f.write(f"\n✓ Gamma Düzeltme\n")
                f.write(f"  - Gamma Değeri: {gamma_value}\n")
                if gamma_value < 1.0:
                    f.write(f"  - Etki: Görüntüyü aydınlatır (karanlık görüntüler için)\n")
                elif gamma_value > 1.0:
                    f.write(f"  - Etki: Görüntüyü karartır (parlak görüntüler için)\n")
            
            if use_ssr:
                f.write(f"\n✓ SSR (Single-Scale Retinex)\n")
                f.write(f"  - Sigma: {ssr_sigma}\n")
                f.write(f"  - Açıklama: Gauss filtresi standart sapması\n")
            
            if use_msr:
                f.write(f"\n✓ MSR (Multi-Scale Retinex)\n")
                f.write(f"  - Sigma Değerleri: {msr_sigmas}\n")
                f.write(f"  - Ölçek Sayısı: {len(msr_sigmas)}\n")
            
            if use_sharpen:
                f.write(f"\n✓ Kenar Netleştirme (Sharpening)\n")
                f.write(f"  - Yöntem: {sharpen_method}\n")
                f.write(f"  - Güç: {sharpen_strength}\n")
                if sharpen_method == 'unsharp':
                    f.write(f"  - Kernel Boyutu: {sharpen_kernel_size}\n")
                f.write(f"  - Açıklama: Kenarları netleştirir, detayları belirginleştirir\n")
            
            f.write(f"\n{'=' * 60}\n")
            f.write(f"\nUygulama Sırası:\n")
            for i, method in enumerate(applied_methods, 1):
                f.write(f"  {i}. {method}\n")
            
            f.write(f"\nGörüntü Bilgileri:\n")
            f.write(f"  - Orijinal Boyut: {original_img.shape}\n")
            f.write(f"  - İşlenmiş Boyut: {processed_img.shape}\n")
            
            f.write(f"\nÇıktı Dosyaları:\n")
            f.write(f"  - Karşılaştırma Grafiği: {comparison_filename}\n")
            f.write(f"  - İyileştirilmiş Görüntü: {output_image_filename}\n")
        
        print(f"İşlem bilgileri kaydedildi: {info_filename}")
    
    plt.show()
    print("\n=== İşlem Tamamlandı ===")
    
    return processed_img


def parse_arguments():
    """
    Komut satırı argümanlarını parse eder.
    """
    parser = argparse.ArgumentParser(
        description='Hibrit Görüntü İyileştirme - CLAHE, Gamma, SSR, MSR kombinasyonları',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanımlar:
  
  # Sadece CLAHE + Gamma:
  python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-clahe --clahe-clip 3.0 --use-gamma --gamma 0.5

  # MSR + CLAHE kombinasyonu:
  python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-msr --msr-sigmas 15 80 250 --use-clahe --clahe-clip 2.5

  # Tam hibrit (Gamma → MSR → CLAHE):
  python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-gamma --gamma 0.6 --use-msr --msr-sigmas 15 80 250 --use-clahe --order gamma msr clahe

  # Sadece SSR:
  python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-ssr --ssr-sigma 300

  # Karartma + Kenar Netleştirme:
  python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-gamma --gamma 1.5 --use-sharpen --sharpen-strength 1.5

  # Karartma + Netleştirme (sıralama belirtilerek):
  python enchancement.py --image dataset/F/set1/2024-05-18/image.jpg --use-gamma --gamma 2.0 --use-sharpen --sharpen-method unsharp --sharpen-strength 2.0 --order gamma sharpen

  # Parametreleri kullanmadan (varsayılan görüntü ile):
  python enchancement.py
        """
    )
    
    # Görüntü yolu
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='İşlenecek görüntünün yolu (dataset/... ile başlayan göreli yol veya tam yol)'
    )
    
    # CLAHE parametreleri
    parser.add_argument(
        '--use-clahe',
        action='store_true',
        help='CLAHE yöntemini kullan'
    )
    parser.add_argument(
        '--clahe-clip',
        type=float,
        default=3.0,
        help='CLAHE clip limit değeri (varsayılan: 3.0)'
    )
    parser.add_argument(
        '--clahe-tile',
        type=int,
        nargs=2,
        default=[8, 8],
        metavar=('W', 'H'),
        help='CLAHE tile grid size (varsayılan: 8 8)'
    )
    
    # Gamma parametreleri
    parser.add_argument(
        '--use-gamma',
        action='store_true',
        help='Gamma düzeltme yöntemini kullan'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Gamma değeri (varsayılan: 0.5, <1.0 aydınlatır, >1.0 karartır)'
    )
    
    # SSR parametreleri
    parser.add_argument(
        '--use-ssr',
        action='store_true',
        help='SSR (Single-Scale Retinex) yöntemini kullan'
    )
    parser.add_argument(
        '--ssr-sigma',
        type=int,
        default=80,
        help='SSR sigma değeri (varsayılan: 80)'
    )
    
    # MSR parametreleri
    parser.add_argument(
        '--use-msr',
        action='store_true',
        help='MSR (Multi-Scale Retinex) yöntemini kullan'
    )
    parser.add_argument(
        '--msr-sigmas',
        type=int,
        nargs='+',
        default=[15, 80, 250],
        metavar='SIGMA',
        help='MSR sigma değerleri listesi (varsayılan: 15 80 250)'
    )
    
    # Sharpening parametreleri
    parser.add_argument(
        '--use-sharpen',
        action='store_true',
        help='Kenar netleştirme (sharpening) yöntemini kullan'
    )
    parser.add_argument(
        '--sharpen-method',
        type=str,
        choices=['unsharp', 'laplacian'],
        default='unsharp',
        help='Netleştirme yöntemi (varsayılan: unsharp)'
    )
    parser.add_argument(
        '--sharpen-strength',
        type=float,
        default=1.0,
        help='Netleştirme gücü (varsayılan: 1.0, 2.0 = güçlü)'
    )
    parser.add_argument(
        '--sharpen-kernel',
        type=int,
        default=5,
        help='Kernel boyutu - unsharp için (varsayılan: 5)'
    )
    
    # Kombinasyon sırası
    parser.add_argument(
        '--order',
        type=str,
        nargs='+',
        choices=['clahe', 'gamma', 'ssr', 'msr', 'sharpen'],
        default=None,
        help='Yöntemlerin uygulanma sırası (örn: --order gamma sharpen clahe)'
    )
    
    # Sonuçları kaydetme
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Sonuçları kaydetme (sadece ekranda göster)'
    )
    
    # Varsayılan örnekler
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4],
        help='Hazır örnek kombinasyonları kullan (1-4)'
    )
    
    return parser.parse_args()


# --- KULLANIM ÖRNEKLERİ ---
if __name__ == "__main__":
    args = parse_arguments()
    
    # Script'in bulunduğu klasörü baz al (python klasörü)
    script_dir = Path(__file__).parent.resolve()
    # Proje kök dizini (bir üst dizin)
    project_root = script_dir.parent
    
    # Görüntü yolunu belirle
    if args.example:
        # Hazır örnekler için varsayılan görüntü
        relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_17-39-33.jpg'
    elif args.image:
        relative_image_path = args.image
    else:
        # Varsayılan görüntü
        relative_image_path = 'dataset/F/set1/2024-05-18/image_USB_VID_0C45_PID_62C0_MI_00_8_2B673C3B_0_0000_2024-05-18_17-39-33.jpg'
    
    # Proje kök dizinine göre mutlak yol oluştur
    if Path(relative_image_path).is_absolute():
        image_path = Path(relative_image_path)
    else:
        image_path = project_root / relative_image_path
    
    # Örnek kombinasyonları kullan
    if args.example:
        if args.example == 1:
            # ÖRNEK 1: CLAHE + Gamma
            hybrid_enhancement(
                image_path=image_path,
                use_clahe=True, 
                clahe_clip_limit=3.0, 
                clahe_tile_size=(8, 8),
                use_gamma=True, 
                gamma_value=0.5,
                use_ssr=False,
                use_msr=False,
                combination_order=['clahe', 'gamma'],
                save_results=not args.no_save
            )
        elif args.example == 2:
            # ÖRNEK 2: MSR + CLAHE
            hybrid_enhancement(
                image_path=image_path,
                use_clahe=True, 
                clahe_clip_limit=2.5, 
                clahe_tile_size=(8, 8),
                use_gamma=False,
                use_ssr=False,
                use_msr=True, 
                msr_sigmas=[15, 80, 250],
                combination_order=['msr', 'clahe'],
                save_results=not args.no_save
            )
        elif args.example == 3:
            # ÖRNEK 3: Tam Hibrit
            hybrid_enhancement(
                image_path=image_path,
                use_clahe=True, 
                clahe_clip_limit=2.5, 
                clahe_tile_size=(8, 8),
                use_gamma=True, 
                gamma_value=0.6,
                use_ssr=False,
                use_msr=True, 
                msr_sigmas=[15, 80, 250],
                combination_order=['gamma', 'msr', 'clahe'],
                save_results=not args.no_save
            )
        elif args.example == 4:
            # ÖRNEK 4: SSR
            hybrid_enhancement(
                image_path=image_path,
                use_clahe=False,
                use_gamma=False,
                use_ssr=True, 
                ssr_sigma=300,
                use_msr=False,
                save_results=not args.no_save
            )
    else:
        # Kullanıcı parametreleriyle çalıştır
        combination_order = args.order if args.order else None
        
        hybrid_enhancement(
            image_path=image_path,
            use_clahe=args.use_clahe,
            clahe_clip_limit=args.clahe_clip,
            clahe_tile_size=tuple(args.clahe_tile),
            use_gamma=args.use_gamma,
            gamma_value=args.gamma,
            use_ssr=args.use_ssr,
            ssr_sigma=args.ssr_sigma,
            use_msr=args.use_msr,
            msr_sigmas=args.msr_sigmas,
            use_sharpen=args.use_sharpen,
            sharpen_method=args.sharpen_method,
            sharpen_strength=args.sharpen_strength,
            sharpen_kernel_size=args.sharpen_kernel,
            combination_order=combination_order,
            save_results=not args.no_save
        )

