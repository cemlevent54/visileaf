"""
Enhancement service - Image enhancement business logic
"""
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import logging
from scipy.spatial import distance
from scipy.ndimage import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)

# Python klasörünü path'e ekle (lazy initialization)
_project_root = None
_python_dir = None



def _get_python_dir():
    """Python klasörünü döndürür (lazy initialization)"""
    global _project_root, _python_dir
    if _python_dir is None:
        _project_root = Path(__file__).parent.parent.parent.parent
        _python_dir = _project_root / "python"
        if str(_python_dir) not in sys.path:
            sys.path.insert(0, str(_python_dir))
    return _python_dir


def _single_scale_retinex_core(img: np.ndarray, variance: float) -> np.ndarray:
    """
    Single-Scale Retinex core function (GitHub repo implementation).
    
    Args:
        img: Input image (numpy array)
        variance: Gaussian filter standard deviation (sigma)
    
    Returns:
        Retinex result in log domain
    """
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def _multi_scale_retinex_core(img: np.ndarray, variance_list: list) -> np.ndarray:
    """
    Multi-Scale Retinex core function (GitHub repo implementation).
    
    Args:
        img: Input image (numpy array)
        variance_list: List of Gaussian filter standard deviations
    
    Returns:
        Retinex result in log domain
    """
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += _single_scale_retinex_core(img, variance)
    retinex = retinex / len(variance_list)
    return retinex


# ============================================================================
# LIME/DUAL Low-Light Enhancement Implementation
# ============================================================================

def _get_sparse_neighbor(p: int, n: int, m: int):
    """
    Komşuluk bilgilerini döndürür.
    
    Orijinal LIME/DUAL implementasyonundaki `get_sparse_neighbor` fonksiyonuna
    benzer şekilde, her piksel için 4-komşulu (yukarı, aşağı, sol, sağ) komşuları
    ve bunların yön bilgisini (yatay/dikey) üretir.
    
    Args:
        p: Düzleştirilmiş indeks (0..n*m-1)
        n: Yükseklik
        m: Genişlik
    
    Returns:
        dict[int, tuple[int, int, bool]]:
            q -> (k, l, x) şeklinde:
              - q: komşu pikselin düzleştirilmiş indeksi
              - k, l: komşu pikselin satır ve sütunu
              - x: True ise yatay (left/right), False ise dikey (up/down)
    """
    i = p // m
    j = p % m
    neighbors = {}
    
    # Sol komşu (yatay)
    if j - 1 >= 0:
        q = p - 1
        neighbors[q] = (i, j - 1, True)
    
    # Sağ komşu (yatay)
    if j + 1 < m:
        q = p + 1
        neighbors[q] = (i, j + 1, True)
    
    # Üst komşu (dikey)
    if i - 1 >= 0:
        q = p - m
        neighbors[q] = (i - 1, j, False)
    
    # Alt komşu (dikey)
    if i + 1 < n:
        q = p + m
        neighbors[q] = (i + 1, j, False)
    
    return neighbors


def _create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15) -> np.ndarray:
    """
    Uzamsal yakınlık temelli Gaussian ağırlık kerneli oluşturur.
    """
    kernel = np.zeros((size, size))
    center = (size // 2, size // 2)
    
    for i in range(size):
        for j in range(size):
            d = distance.euclidean((i, j), center)
            kernel[i, j] = np.exp(-0.5 * (d ** 2) / (spatial_sigma ** 2))
    
    return kernel


def _compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Aydınlatma haritası için düzgünlük ağırlıklarını hesaplar.
    
    Args:
        L: Başlangıç aydınlatma haritası
        x: Yön (1: yatay, 0: dikey)
        kernel: Uzamsal affinity matrisi
        eps: Sayısal kararlılık için küçük sabit
    """
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    
    T = convolve(np.ones_like(L), kernel, mode="constant")
    T = T / (np.abs(convolve(Lp, kernel, mode="constant")) + eps)
    
    return T / (np.abs(Lp) + eps)


def _fuse_multi_exposure_images(
    im: np.ndarray,
    under_ex: np.ndarray,
    over_ex: np.ndarray,
    bc: float = 1.0,
    bs: float = 1.0,
    be: float = 1.0,
) -> np.ndarray:
    """
    DUAL makalesindeki exposure fusion yöntemini uygular.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    
    return fused_images


def _refine_illumination_map_linear(
    L: np.ndarray,
    gamma: float,
    lambda_: float,
    kernel: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    LIME/DUAL'de tanımlanan optimizasyon problemini çözerek aydınlatma
    haritasını rafine eder (hızlandırılmış lineer çözücü).
    """
    # Düzgünlük ağırlıkları
    wx = _compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = _compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)
    
    n, m = L.shape
    L_1d = L.copy().flatten()
    
    # Beş-noktalı, uzamsal olarak inhomogeneous Laplace matrisi
    row, column, data = [], [], []
    
    for p in range(n * m):
        diag = 0.0
        for q, (k, l, is_horizontal) in _get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if is_horizontal else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        
        row.append(p)
        column.append(p)
        data.append(diag)
    
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))
    
    # Lineer sistemi çöz
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    
    L_refined = spsolve(csr_matrix(A), L_1d).reshape((n, m))
    
    # Gamma düzeltmesi
    L_refined = np.clip(L_refined, eps, 1.0) ** gamma
    
    return L_refined


def _correct_underexposure(
    im: np.ndarray,
    gamma: float,
    lambda_: float,
    kernel: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    LIME/DUAL'deki retinex-tabanlı algoritma ile düşük pozlanmış bölgeleri düzeltir.
    """
    # İlk aydınlatma haritası tahmini
    L = np.max(im, axis=-1)
    
    # Aydınlatma haritasını rafine et
    L_refined = _refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)
    
    # Görüntüyü düzelt
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d
    
    return im_corrected


def _enhance_image_exposure(
    im: np.ndarray,
    gamma: float,
    lambda_: float,
    dual: bool = True,
    sigma: int = 3,
    bc: float = 1.0,
    bs: float = 1.0,
    be: float = 1.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Girdi görüntüsünün pozlamasını LIME veya DUAL yöntemleriyle iyileştirir.
    """
    # Uzamsal affinity kerneli
    kernel = _create_spacial_affinity_kernel(sigma)
    
    # Normalize et
    im_normalized = im.astype(float) / 255.0
    
    # Düşük pozlanmış bölgeleri düzelt
    under_corrected = _correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)
    
    if dual:
        # DUAL için: aşırı pozlanmış bölgeleri de düzelt ve birleştir
        inv_im_normalized = 1.0 - im_normalized
        over_corrected = 1.0 - _correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
        
        im_corrected = _fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
    else:
        # LIME için yalnızca under_corrected kullanılır
        im_corrected = under_corrected
    
    # 8-bit aralığa geri dön
    return np.clip(im_corrected * 255.0, 0, 255).astype("uint8")


# ============================================================================
# Dark Channel Prior (DCP) Low-Light Enhancement Implementation
# ============================================================================

def _get_dark_channel(img: np.ndarray, size: int) -> np.ndarray:
    """
    Görüntüdeki en karanlık pikselleri bulur (Dark Channel Prior).
    img: float64 veya float32, [0,1] aralığında veya 0-255, BGR
    """
    # Kanal bazında minimum
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def _estimate_atmospheric_light(img: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    """
    Ortamdaki en parlak ışığı (atmosferik ışık) tahmin eder.
    img: float64/float32, [0,1] aralığında, BGR
    """
    h, w = img.shape[:2]
    num_pixels = h * w
    # En parlak yaklaşık %0.1'lik kısmı al
    num_top_pixels = int(max(num_pixels / 1000, 1))
    
    dark_vec = dark_channel.reshape(num_pixels)
    img_vec = img.reshape(num_pixels, 3)
    
    indices = dark_vec.argsort()[-num_top_pixels:]
    
    # En parlak piksellerin ortalamasını alarak atmosferik ışığı bul
    atm_light = np.mean(img_vec[indices], axis=0)
    return atm_light


def _get_transmission(img: np.ndarray, atm_light: np.ndarray, size: int, omega: float = 0.95) -> np.ndarray:
    """
    Işığın ne kadarının geçtiğini (Transmission Map) hesaplar.
    img: float64/float32, [0,1], BGR
    """
    # Bölme sırasında sıfıra gitmemek için küçük epsilon
    eps = 1e-6
    norm_img = img / (atm_light.reshape(1, 1, 3) + eps)
    dark_channel = _get_dark_channel(norm_img, size)
    transmission = 1.0 - omega * dark_channel
    return transmission


def _simple_guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    ximgproc.guidedFilter olmadığı durumda kullanılan basit guided filter implementasyonu.
    I: guide image (grayscale, float64)
    p: input image (transmission map, float64)
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * I + mean_b
    return q


def _guided_filter_refinement(
    img_gray: np.ndarray,
    transmission: np.ndarray,
    radius: int = 60,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Guided filter ile kaba transmission haritasını iyileştirir.
    Mümkünse cv2.ximgproc.guidedFilter kullanır, değilse simple_guided_filter'e düşer.
    """
    # Önce ximgproc içindeki guidedFilter'ı dene
    try:
        guided = cv2.ximgproc.guidedFilter(  # type: ignore[attr-defined]
            guide=img_gray.astype(np.float32),
            src=transmission.astype(np.float32),
            radius=radius,
            eps=eps,
        )
        return guided.astype(np.float64)
    except (AttributeError, cv2.error):
        # opencv-contrib yoksa veya ximgproc bulunamazsa basit guided filter kullan
        return _simple_guided_filter(
            img_gray.astype(np.float64),
            transmission.astype(np.float64),
            radius,
            eps,
        )


def _dehaze(img: np.ndarray, patch_size: int = 15, tmin: float = 0.1) -> np.ndarray:
    """
    Sis giderme (Dehazing) işlemini uygular.
    img: BGR uint8 görüntü
    """
    img_float = img.astype("float64") / 255.0
    
    # 1. Dark Channel Hesapla
    dark = _get_dark_channel(img_float, patch_size)
    
    # 2. Atmosferik Işığı Tahmin Et
    A = _estimate_atmospheric_light(img_float, dark)
    
    # 3. İletim Haritasını (Transmission) Hesapla
    te = _get_transmission(img_float, A, patch_size)
    
    # Sıfıra bölünme hatasını önlemek için t değerini sınırla
    t = np.maximum(te, tmin)
    
    # 4. Görüntüyü Kurtar (Recover Scene Radiance)
    res = np.empty_like(img_float)
    for i in range(3):
        res[:, :, i] = (img_float[:, :, i] - A[i]) / t + A[i]
    
    # Değerleri 0-255 arasına sıkıştır
    res = np.clip(res, 0.0, 1.0)
    return (res * 255.0).astype(np.uint8)


def _dehaze_advanced(img: np.ndarray, patch_size: int = 15, tmin: float = 0.1) -> np.ndarray:
    """
    Guided filter ile iyileştirilmiş gelişmiş sis giderme (dehazing).
    img: BGR uint8 görüntü
    """
    img_float = img.astype("float64") / 255.0
    
    # 1. Dark Channel ve Atmosferik Işık
    dark = _get_dark_channel(img_float, patch_size)
    A = _estimate_atmospheric_light(img_float, dark)
    
    # 2. Kaba Transmission Haritası
    te = _get_transmission(img_float, A, patch_size)
    
    # 3. Guided Filter ile transmission'ı iyileştir
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float64") / 255.0
    t_refined = _guided_filter_refinement(img_gray, te, radius=60, eps=1e-4)
    
    t = np.maximum(t_refined, tmin)
    
    # 4. Görüntüyü Kurtar (Recover Scene Radiance)
    res = np.empty_like(img_float)
    for i in range(3):
        res[:, :, i] = (img_float[:, :, i] - A[i]) / t + A[i]
    
    res = np.clip(res, 0.0, 1.0)
    return (res * 255.0).astype(np.uint8)


class EnhancementService:
    """Image enhancement service for business logic"""

    # ------------------------------------------------------------------
    # Core enhancement operations (moved from module-level functions)
    # ------------------------------------------------------------------

    def apply_clahe_to_image(
        self,
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ) -> np.ndarray:
        """
        Apply CLAHE to numpy array image.

        Args:
            img: BGR format image (numpy array)
            clip_limit: Contrast limiting threshold
            tile_grid_size: Grid size

        Returns:
            Processed image
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_img

    def apply_gamma_to_image(self, img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        """
        Apply gamma correction to numpy array image.

        Args:
            img: BGR format image (numpy array)
            gamma: Gamma value (must be > 0)

        Returns:
            Processed image
        """
        # Validate gamma value
        if gamma <= 0:
            raise ValueError(f"Gamma value must be positive, got {gamma}")

        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(
            "uint8"
        )
        gamma_corrected_img = cv2.LUT(img, table)
        return gamma_corrected_img

    def apply_ssr_to_image(self, img: np.ndarray, sigma: int = 80) -> np.ndarray:
        """
        Apply SSR to numpy array image using GitHub repo implementation with
        histogram-based normalization.

        Args:
            img: BGR format image (numpy array)
            sigma: Gaussian filter standard deviation (variance)

        Returns:
            Processed image
        """
        # Validate sigma
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")

        try:
            # Convert to float64 and add 1.0 to avoid log(0)
            img_float = img.astype(np.float64) + 1.0

            # Apply single-scale retinex
            img_retinex = _single_scale_retinex_core(img_float, float(sigma))

            # Apply histogram-based normalization for each channel (GitHub repo approach)
            for i in range(img_retinex.shape[2]):
                unique, count = np.unique(
                    np.int32(img_retinex[:, :, i] * 100), return_counts=True
                )

                # Find zero count
                zero_count = 0
                for u, c in zip(unique, count):
                    if u == 0:
                        zero_count = c
                        break

                # Initialize bounds
                low_val = unique[0] / 100.0
                high_val = unique[-1] / 100.0

                # Outlier clipping: values with count < 10% of zero_count (if zero_count exists)
                if zero_count > 0:
                    for u, c in zip(unique, count):
                        if u < 0 and c < zero_count * 0.1:
                            low_val = u / 100.0
                        if u > 0 and c < zero_count * 0.1:
                            high_val = u / 100.0
                            break

                # Apply clipping
                img_retinex[:, :, i] = np.maximum(
                    np.minimum(img_retinex[:, :, i], high_val), low_val
                )

                # Min-max normalization to [0, 255]
                min_val = np.min(img_retinex[:, :, i])
                max_val = np.max(img_retinex[:, :, i])

                if max_val == min_val or abs(max_val - min_val) < 1e-10:
                    # Uniform channel, return original
                    img_retinex[:, :, i] = img[:, :, i].astype(np.float64)
                else:
                    img_retinex[:, :, i] = (
                        (img_retinex[:, :, i] - min_val) / (max_val - min_val) * 255
                    )

            # Convert back to uint8
            img_retinex = np.uint8(img_retinex)
            return img_retinex

        except Exception as e:
            logger.error(
                f"SSR processing error: {e}. Sigma: {sigma}, Image shape: {img.shape}"
            )
            raise ValueError(f"SSR processing failed: {str(e)}")

    def apply_msr_to_image(
        self, img: np.ndarray, sigma_list: list = [15, 80, 250]
    ) -> np.ndarray:
        """
        Apply MSR to numpy array image using GitHub repo implementation with
        histogram-based normalization.

        Args:
            img: BGR format image (numpy array)
            sigma_list: List of Gaussian filter standard deviations (variance values)

        Returns:
            Processed image
        """
        # Validate sigma_list
        if not sigma_list or len(sigma_list) == 0:
            raise ValueError("MSR sigma_list cannot be empty")

        for sigma in sigma_list:
            if sigma <= 0:
                raise ValueError(
                    f"All MSR sigma values must be positive, got {sigma_list}"
                )

        try:
            # Convert to float64 and add 1.0 to avoid log(0)
            img_float = img.astype(np.float64) + 1.0

            # Apply multi-scale retinex
            img_retinex = _multi_scale_retinex_core(
                img_float, [float(s) for s in sigma_list]
            )

            # Apply histogram-based normalization for each channel (GitHub repo approach)
            for i in range(img_retinex.shape[2]):
                unique, count = np.unique(
                    np.int32(img_retinex[:, :, i] * 100), return_counts=True
                )

                # Find zero count
                zero_count = 0
                for u, c in zip(unique, count):
                    if u == 0:
                        zero_count = c
                        break

                # Initialize bounds
                low_val = unique[0] / 100.0
                high_val = unique[-1] / 100.0

                # Outlier clipping: values with count < 10% of zero_count (if zero_count exists)
                if zero_count > 0:
                    for u, c in zip(unique, count):
                        if u < 0 and c < zero_count * 0.1:
                            low_val = u / 100.0
                        if u > 0 and c < zero_count * 0.1:
                            high_val = u / 100.0
                            break

                # Apply clipping
                img_retinex[:, :, i] = np.maximum(
                    np.minimum(img_retinex[:, :, i], high_val), low_val
                )

                # Min-max normalization to [0, 255]
                min_val = np.min(img_retinex[:, :, i])
                max_val = np.max(img_retinex[:, :, i])

                if max_val == min_val or abs(max_val - min_val) < 1e-10:
                    # Uniform channel, return original
                    img_retinex[:, :, i] = img[:, :, i].astype(np.float64)
                else:
                    img_retinex[:, :, i] = (
                        (img_retinex[:, :, i] - min_val) / (max_val - min_val) * 255
                    )

            # Convert back to uint8
            img_retinex = np.uint8(img_retinex)
            return img_retinex

        except Exception as e:
            logger.error(
                f"MSR processing error: {e}. Sigma list: {sigma_list}, Image shape: {img.shape}"
            )
            raise ValueError(f"MSR processing failed: {str(e)}")

    def apply_lowlight_lime(
        self,
        img: np.ndarray,
        gamma: float = 0.6,
        lambda_: float = 0.15,
        sigma: float = 3.0,
        bc: float = 1.0,
        bs: float = 1.0,
        be: float = 1.0,
    ) -> np.ndarray:
        """
        LIME (Low-light Image Enhancement) benzeri low-light iyileştirme.
        Doğrudan bu sınıf içinde implement edilmiştir, harici modül gerektirmez.
        """
        logger.debug(
            "Applying Low-light LIME with params: "
            f"gamma={gamma}, lambda_={lambda_}, sigma={sigma}, bc={bc}, bs={bs}, be={be}"
        )

        # BGR uint8 bekler, kendi içinde normalize eder
        enhanced = _enhance_image_exposure(
            img,
            gamma=gamma,
            lambda_=lambda_,
            dual=False,
            sigma=int(sigma),
            bc=bc,
            bs=bs,
            be=be,
        )

        return enhanced

    def apply_lowlight_dual(
        self,
        img: np.ndarray,
        gamma: float = 0.6,
        lambda_: float = 0.15,
        sigma: float = 3.0,
        bc: float = 1.0,
        bs: float = 1.0,
        be: float = 1.0,
    ) -> np.ndarray:
        """
        DUAL (Dual Illumination Estimation) benzeri low-light iyileştirme.
        Doğrudan bu sınıf içinde implement edilmiştir, harici modül gerektirmez.
        """
        logger.debug(
            "Applying Low-light DUAL with params: "
            f"gamma={gamma}, lambda_={lambda_}, sigma={sigma}, bc={bc}, bs={bs}, be={be}"
        )

        enhanced = _enhance_image_exposure(
            img,
            gamma=gamma,
            lambda_=lambda_,
            dual=True,
            sigma=int(sigma),
            bc=bc,
            bs=bs,
            be=be,
        )

        return enhanced

    def apply_dcp_lowlight(self, img: np.ndarray) -> np.ndarray:
        """
        Dark Channel Prior tabanlı low-light enhancement uygular.
        Doğrudan bu sınıf içinde implement edilmiştir, harici modül gerektirmez.

        Yöntem:
          1. Görüntüyü ters çevir (invert)
          2. İnvert edilmiş görüntüye dehaze uygula
          3. Sonucu tekrar ters çevir
        """
        logger.debug("Applying Dark Channel Prior based low-light enhancement")

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # 1. Adım: Invert
        inverted_img = 255 - img

        # 2. Adım: Dehaze uygula
        dehazed_inverted = _dehaze(inverted_img)

        # 3. Adım: Sonucu tekrar ters çevir
        enhanced_img = 255 - dehazed_inverted

        return enhanced_img

    def apply_dcp_lowlight_guided(self, img: np.ndarray) -> np.ndarray:
        """
        Dark Channel Prior + Guided Filter tabanlı gelişmiş low-light enhancement
        uygular. Doğrudan bu sınıf içinde implement edilmiştir, harici modül
        gerektirmez.

        Yöntem:
          1. Görüntüyü ters çevir
          2. Gelişmiş dehaze (guided filter ile) uygula
          3. Tekrar ters çevir
        """
        logger.debug(
            "Applying Dark Channel Prior + Guided Filter based low-light enhancement"
        )

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        inverted_img = 255 - img
        dehazed_inverted = _dehaze_advanced(inverted_img)
        enhanced_img = 255 - dehazed_inverted

        return enhanced_img

    def apply_sharpen_to_image(
        self,
        img: np.ndarray,
        method: str = "unsharp",
        strength: float = 1.0,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Apply sharpening to numpy array image.

        Args:
            img: BGR format image (numpy array)
            method: Sharpening method ('unsharp' or 'laplacian')
            strength: Sharpening strength (1.0 = normal, 2.0 = strong)
            kernel_size: Kernel size (for unsharp method)

        Returns:
            Processed image
        """
        if method == "unsharp":
            # Unsharp Masking method
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        elif method == "laplacian":
            # Laplacian filter sharpening
            kernel = (
                np.array(
                    [
                        [-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1],
                    ]
                )
                * strength
            )
            sharpened = cv2.filter2D(img, -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = img.copy()

        return sharpened

    def apply_denoise(self, img: np.ndarray, strength: float = 3.0) -> np.ndarray:
        """
        Renk gürültülerini (mavi/kırmızı lekeler) temizler.

        strength: Temizleme gücü. 3.0 hafif, 10.0 oldukça güçlüdür.
        """
        # fastNlMeansDenoisingColored özellikle renkli lekeler için tasarlanmıştır.
        # h: Luma (parlaklık) temizleme gücü
        # hColor: Chroma (renk) temizleme gücü
        h = float(strength)
        h_color = float(strength + 7.0)

        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=h,
            hColor=h_color,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return denoised

    def enhance_image(
        self,
        image_bytes: bytes,
        use_gamma: bool = False,
        gamma: float = 0.5,
        use_clahe: bool = False,
        clahe_clip: float = 3.0,
        clahe_tile_size: list = [8, 8],
        use_ssr: bool = False,
        ssr_sigma: int = 80,
        use_msr: bool = False,
        msr_sigmas: list = [15, 80, 250],
        use_sharpen: bool = False,
        sharpen_method: str = 'unsharp',
        sharpen_strength: float = 1.0,
        sharpen_kernel_size: int = 5,
        # Eğitimlik temel dönüşümler
        use_negative: bool = False,
        use_threshold: bool = False,
        threshold_value: int = 128,
        use_gray_slice: bool = False,
        gray_slice_low: float = 100.0,
        gray_slice_high: float = 180.0,
        use_bitplane: bool = False,
        bitplane_bit: int = 7,
        # Denoise
        use_denoise: bool = False,
        denoise_strength: float = 3.0,
        # DCP tabanlı low-light modları (pipeline içinden çağrılabilir)
        use_dcp: bool = False,
        use_dcp_guided: bool = False,
        # Low-light seçenekleri (LIME/DUAL)
        use_lowlight_lime: bool = False,
        use_lowlight_dual: bool = False,
        lowlight_gamma: float = 0.6,
        lowlight_lambda: float = 0.15,
        lowlight_sigma: float = 3.0,
        lowlight_bc: float = 1.0,
        lowlight_bs: float = 1.0,
        lowlight_be: float = 1.0,
        order: Optional[list] = None
    ) -> bytes:
        """
        Enhance image with specified methods.
        
        Args:
            image_bytes: Image file bytes
            use_gamma: Use gamma correction
            gamma: Gamma value
            use_clahe: Use CLAHE
            clahe_clip: CLAHE clip limit
            use_ssr: Use SSR
            ssr_sigma: SSR sigma value
            use_msr: Use MSR
            msr_sigmas: MSR sigma values list
            order: Order of enhancement methods
        
        Returns:
            Enhanced image as bytes (JPEG format)
        
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image format")
            
            # Validate parameters before processing
            if use_gamma and gamma <= 0:
                raise ValueError(f"Gamma value must be positive, got {gamma}")
            if use_clahe:
                if clahe_clip <= 0:
                    raise ValueError(f"CLAHE clip limit must be positive, got {clahe_clip}")
                if clahe_tile_size:
                    if len(clahe_tile_size) != 2:
                        raise ValueError(f"CLAHE tile size must have 2 elements, got {clahe_tile_size}")
                    if clahe_tile_size[0] <= 0 or clahe_tile_size[1] <= 0:
                        raise ValueError(f"CLAHE tile size values must be positive, got {clahe_tile_size}")
            if use_ssr and ssr_sigma <= 0:
                raise ValueError(f"SSR sigma must be positive, got {ssr_sigma}")
            if use_msr:
                if not msr_sigmas or len(msr_sigmas) == 0:
                    raise ValueError("MSR sigma list cannot be empty")
                for sigma in msr_sigmas:
                    if sigma <= 0:
                        raise ValueError(f"All MSR sigma values must be positive, got {msr_sigmas}")
            if use_sharpen and sharpen_kernel_size <= 0:
                raise ValueError(f"Sharpen kernel size must be positive, got {sharpen_kernel_size}")

            # Low-light parametreleri için temel doğrulama
            if use_lowlight_lime or use_lowlight_dual:
                if lowlight_gamma <= 0:
                    raise ValueError(f"Low-light gamma must be positive, got {lowlight_gamma}")
                if lowlight_lambda <= 0:
                    raise ValueError(f"Low-light lambda must be positive, got {lowlight_lambda}")
                if lowlight_sigma <= 0:
                    raise ValueError(f"Low-light sigma must be positive, got {lowlight_sigma}")
                for name, val in (("bc", lowlight_bc), ("bs", lowlight_bs), ("be", lowlight_be)):
                    if val < 0:
                        raise ValueError(f"Low-light {name} must be non-negative, got {val}")

            # Eğitimlik filtreler için temel doğrulama
            if use_threshold:
                if not (0 <= threshold_value <= 255):
                    raise ValueError(f"Threshold value must be between 0 and 255, got {threshold_value}")
            if use_gray_slice:
                if not (0 <= gray_slice_low <= 255) or not (0 <= gray_slice_high <= 255):
                    raise ValueError(
                        f"Gray slice bounds must be between 0 and 255, got "
                        f"low={gray_slice_low}, high={gray_slice_high}"
                    )
                if gray_slice_low > gray_slice_high:
                    raise ValueError(
                        f"Gray slice low must be <= high, got low={gray_slice_low}, high={gray_slice_high}"
                    )
            if use_bitplane:
                if not (0 <= bitplane_bit <= 7):
                    raise ValueError(f"Bitplane bit must be between 0 and 7, got {bitplane_bit}")
            if use_denoise:
                if denoise_strength <= 0:
                    raise ValueError(f"Denoise strength must be positive, got {denoise_strength}")
                if denoise_strength > 20:
                    raise ValueError(f"Denoise strength is too high (max 20), got {denoise_strength}")
            # DCP modları için şu an ekstra numerik doğrulama yok; sadece bool bayraklar
            
            # Determine methods to apply
            methods_to_apply = []
            if use_clahe:
                tile_size_tuple = tuple(clahe_tile_size) if isinstance(clahe_tile_size, list) else clahe_tile_size
                methods_to_apply.append(('clahe', {'clip_limit': clahe_clip, 'tile_size': tile_size_tuple}))
            if use_gamma:
                methods_to_apply.append(('gamma', {'gamma': gamma}))
            if use_ssr:
                methods_to_apply.append(('ssr', {'sigma': ssr_sigma}))
            if use_msr:
                methods_to_apply.append(('msr', {'sigmas': msr_sigmas}))
            if use_sharpen:
                methods_to_apply.append(('sharpen', {
                    'method': sharpen_method,
                    'strength': sharpen_strength,
                    'kernel_size': sharpen_kernel_size
                }))
            # Eğitimlik temel dönüşümler
            if use_negative:
                methods_to_apply.append(('negative', {}))
            if use_threshold:
                methods_to_apply.append(('threshold', {'thresh': threshold_value}))
            if use_gray_slice:
                methods_to_apply.append((
                    'gray_slice',
                    {'low': gray_slice_low, 'high': gray_slice_high},
                ))
            if use_bitplane:
                methods_to_apply.append(('bitplane', {'bit': bitplane_bit}))
            if use_denoise:
                methods_to_apply.append(('denoise', {'strength': denoise_strength}))
            # DCP tabanlı modlar
            if use_dcp:
                methods_to_apply.append(('dcp', {}))
            if use_dcp_guided:
                methods_to_apply.append(('dcp_guided', {}))
            # Low-light modlarını da birer yöntem olarak ekle
            if use_lowlight_lime:
                methods_to_apply.append((
                    'lowlight_lime',
                    {
                        'gamma': lowlight_gamma,
                        'lambda_': lowlight_lambda,
                        'sigma': lowlight_sigma,
                        'bc': lowlight_bc,
                        'bs': lowlight_bs,
                        'be': lowlight_be,
                    },
                ))
            if use_lowlight_dual:
                methods_to_apply.append((
                    'lowlight_dual',
                    {
                        'gamma': lowlight_gamma,
                        'lambda_': lowlight_lambda,
                        'sigma': lowlight_sigma,
                        'bc': lowlight_bc,
                        'bs': lowlight_bs,
                        'be': lowlight_be,
                    },
                ))
            
            # Apply order if specified
            if order and methods_to_apply:
                ordered_methods = []
                for method_name in order:
                    for method in methods_to_apply:
                        if method[0] == method_name:
                            ordered_methods.append(method)
                            break
                # Add remaining methods not in order
                for method in methods_to_apply:
                    if method not in ordered_methods:
                        ordered_methods.append(method)
                methods_to_apply = ordered_methods
            
            # Process image
            processed_img = img.copy()
            
            for method_name, params in methods_to_apply:
                if method_name == 'clahe':
                    processed_img = self.apply_clahe_to_image(
                        processed_img,
                        clip_limit=params['clip_limit'],
                        tile_grid_size=params['tile_size']
                    )
                elif method_name == 'gamma':
                    processed_img = self.apply_gamma_to_image(
                        processed_img,
                        gamma=params['gamma']
                    )
                elif method_name == 'ssr':
                    processed_img = self.apply_ssr_to_image(
                        processed_img,
                        sigma=params['sigma']
                    )
                elif method_name == 'msr':
                    processed_img = self.apply_msr_to_image(
                        processed_img,
                        sigma_list=params['sigmas']
                    )
                elif method_name == 'sharpen':
                    processed_img = self.apply_sharpen_to_image(
                        processed_img,
                        method=params['method'],
                        strength=params['strength'],
                        kernel_size=params['kernel_size']
                    )
                elif method_name == 'negative':
                    # Klasik negatif görüntü filtresi
                    processed_img = 255 - processed_img
                elif method_name == 'threshold':
                    # Basit binary threshold (grayscale)
                    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(
                        gray, params['thresh'], 255, cv2.THRESH_BINARY
                    )
                    processed_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                elif method_name == 'gray_slice':
                    # Gri seviye dilimleme: belirli aralığı beyaz, diğerlerini siyah yap
                    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                    low, high = float(params['low']), float(params['high'])
                    # Float değerleri int'e çevir (OpenCV uint8 için)
                    low_int = int(round(low))
                    high_int = int(round(high))
                    mask = (gray >= low_int) & (gray <= high_int)
                    result = np.zeros_like(gray, dtype=np.uint8)
                    result[mask] = 255
                    processed_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                elif method_name == 'bitplane':
                    # Bit-plane slicing
                    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                    bit = params['bit']
                    plane = ((gray >> bit) & 1) * 255
                    processed_img = cv2.cvtColor(plane.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif method_name == 'denoise':
                    processed_img = self.apply_denoise(
                        processed_img,
                        strength=params.get('strength', 3.0),
                    )
                elif method_name == 'lowlight_lime':
                    processed_img = self.apply_lowlight_lime(
                        processed_img,
                        gamma=params['gamma'],
                        lambda_=params['lambda_'],
                        sigma=params['sigma'],
                        bc=params['bc'],
                        bs=params['bs'],
                        be=params['be'],
                    )
                elif method_name == 'lowlight_dual':
                    processed_img = self.apply_lowlight_dual(
                        processed_img,
                        gamma=params['gamma'],
                        lambda_=params['lambda_'],
                        sigma=params['sigma'],
                        bc=params['bc'],
                        bs=params['bs'],
                        be=params['be'],
                    )
                elif method_name == 'dcp':
                    processed_img = self.apply_dcp_lowlight(processed_img)
                elif method_name == 'dcp_guided':
                    processed_img = self.apply_dcp_lowlight_guided(processed_img)
            
            # Encode to JPEG bytes
            _, encoded_img = cv2.imencode('.jpg', processed_img)
            return encoded_img.tobytes()
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            raise ValueError(f"Image enhancement failed: {str(e)}")


    def enhance_image_with_dcp(self, image_bytes: bytes) -> bytes:
        """
        Dark Channel Prior (DCP) tabanlı low-light enhancement uygular.

        Args:
            image_bytes: Görüntü dosyası byte'ları

        Returns:
            İyileştirilmiş görüntü (JPEG bytes)
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Invalid image format")

            # DCP tabanlı low-light enhancement uygula
            processed_img = self.apply_dcp_lowlight(img)

            # Encode to JPEG bytes
            _, encoded_img = cv2.imencode('.jpg', processed_img)
            return encoded_img.tobytes()

        except Exception as e:
            logger.error(f"DCP image enhancement failed: {e}")
            raise ValueError(f"DCP image enhancement failed: {str(e)}")


    def enhance_image_with_dcp_guided(self, image_bytes: bytes) -> bytes:
        """
        Dark Channel Prior (DCP) + Guided Filter tabanlı gelişmiş low-light enhancement uygular.

        Args:
            image_bytes: Görüntü dosyası byte'ları

        Returns:
            İyileştirilmiş görüntü (JPEG bytes)
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Invalid image format")

            processed_img = self.apply_dcp_lowlight_guided(img)

            _, encoded_img = cv2.imencode('.jpg', processed_img)
            return encoded_img.tobytes()

        except Exception as e:
            logger.error(f"DCP Guided image enhancement failed: {e}")
            raise ValueError(f"DCP Guided image enhancement failed: {str(e)}")

