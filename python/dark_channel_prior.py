import cv2
import numpy as np


def get_dark_channel(img: np.ndarray, size: int) -> np.ndarray:
    """
    Görüntüdeki en karanlık pikselleri bulur (Dark Channel Prior).
    img: float64 veya float32, [0,1] aralığında veya 0-255, BGR
    """
    # Kanal bazında minimum
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(img: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
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


def get_transmission(img: np.ndarray, atm_light: np.ndarray, size: int, omega: float = 0.95) -> np.ndarray:
    """
    Işığın ne kadarının geçtiğini (Transmission Map) hesaplar.
    img: float64/float32, [0,1], BGR
    """
    # Bölme sırasında sıfıra gitmemek için küçük epsilon
    eps = 1e-6
    norm_img = img / (atm_light.reshape(1, 1, 3) + eps)
    dark_channel = get_dark_channel(norm_img, size)
    transmission = 1.0 - omega * dark_channel
    return transmission


def guided_filter_refinement(
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
        return simple_guided_filter(
            img_gray.astype(np.float64),
            transmission.astype(np.float64),
            radius,
            eps,
        )


def simple_guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
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


def dehaze(img: np.ndarray, patch_size: int = 15, tmin: float = 0.1) -> np.ndarray:
    """
    Sis giderme (Dehazing) işlemini uygular.
    img: BGR uint8 görüntü
    """
    img_float = img.astype("float64") / 255.0

    # 1. Dark Channel Hesapla
    dark = get_dark_channel(img_float, patch_size)

    # 2. Atmosferik Işığı Tahmin Et
    A = estimate_atmospheric_light(img_float, dark)

    # 3. İletim Haritasını (Transmission) Hesapla
    te = get_transmission(img_float, A, patch_size)

    # Sıfıra bölünme hatasını önlemek için t değerini sınırla
    t = np.maximum(te, tmin)

    # 4. Görüntüyü Kurtar (Recover Scene Radiance)
    res = np.empty_like(img_float)
    for i in range(3):
        res[:, :, i] = (img_float[:, :, i] - A[i]) / t + A[i]

    # Değerleri 0-255 arasına sıkıştır
    res = np.clip(res, 0.0, 1.0)
    return (res * 255.0).astype(np.uint8)


def dehaze_advanced(img: np.ndarray, patch_size: int = 15, tmin: float = 0.1) -> np.ndarray:
    """
    Guided filter ile iyileştirilmiş gelişmiş sis giderme (dehazing).
    img: BGR uint8 görüntü
    """
    img_float = img.astype("float64") / 255.0

    # 1. Dark Channel ve Atmosferik Işık
    dark = get_dark_channel(img_float, patch_size)
    A = estimate_atmospheric_light(img_float, dark)

    # 2. Kaba Transmission Haritası
    te = get_transmission(img_float, A, patch_size)

    # 3. Guided Filter ile transmission'ı iyileştir
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float64") / 255.0
    t_refined = guided_filter_refinement(img_gray, te, radius=60, eps=1e-4)

    t = np.maximum(t_refined, tmin)

    # 4. Görüntüyü Kurtar (Recover Scene Radiance)
    res = np.empty_like(img_float)
    for i in range(3):
        res[:, :, i] = (img_float[:, :, i] - A[i]) / t + A[i]

    res = np.clip(res, 0.0, 1.0)
    return (res * 255.0).astype(np.uint8)


def enhance_low_light_with_dcp(img: np.ndarray) -> np.ndarray:
    """
    Dark Channel Prior tabanlı low-light iyileştirme.

    Videodaki yöntem:
      1. Görüntüyü ters çevir (invert)
      2. İnvert edilmiş görüntüye dehaze uygula
      3. Sonucu tekrar ters çevir

    Args:
        img: BGR uint8 görüntü

    Returns:
        BGR uint8 iyileştirilmiş görüntü
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 1. Adım: Invert
    inverted_img = 255 - img

    # 2. Adım: Dehaze uygula
    dehazed_inverted = dehaze(inverted_img)

    # 3. Adım: Sonucu tekrar ters çevir
    enhanced_img = 255 - dehazed_inverted

    return enhanced_img


def enhance_low_light_with_dcp_guided(img: np.ndarray) -> np.ndarray:
    """
    Dark Channel Prior + Guided Filter tabanlı gelişmiş low-light iyileştirme.

    1. Görüntüyü ters çevir
    2. Gelişmiş dehaze (guided filter ile) uygula
    3. Tekrar ters çevir
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    inverted_img = 255 - img
    dehazed_inverted = dehaze_advanced(inverted_img)
    enhanced_img = 255 - dehazed_inverted

    return enhanced_img


