"""
Enhancement service - Image enhancement business logic
"""
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Python klasörünü path'e ekle (lazy initialization)
_project_root = None
_python_dir = None
_single_scale_retinex = None
_multi_scale_retinex = None
_retinex_import_error = None

# Low-light LIME / DUAL modülü (LIME/DUAL orijinaline yakın implementasyon)
_lowlight_module = None
_lowlight_import_error = None


def _get_python_dir():
    """Python klasörünü döndürür (lazy initialization)"""
    global _project_root, _python_dir
    if _python_dir is None:
        _project_root = Path(__file__).parent.parent.parent.parent
        _python_dir = _project_root / "python"
        if str(_python_dir) not in sys.path:
            sys.path.insert(0, str(_python_dir))
    return _python_dir


def _load_retinex_modules():
    """Retinex modüllerini lazy olarak yükler"""
    global _single_scale_retinex, _multi_scale_retinex, _retinex_import_error
    
    if _single_scale_retinex is not None or _retinex_import_error is not None:
        return  # Already loaded or failed
    
    try:
        python_dir = _get_python_dir()
        import importlib.util
        
        # Single-scale retinex
        ssr_spec = importlib.util.spec_from_file_location(
            "single_scale_retinex", 
            python_dir / "single-scale-retinex.py"
        )
        ssr_module = importlib.util.module_from_spec(ssr_spec)
        ssr_spec.loader.exec_module(ssr_module)
        _single_scale_retinex = ssr_module.single_scale_retinex
        
        # Multi-scale retinex
        msr_spec = importlib.util.spec_from_file_location(
            "multi_scale_retinex", 
            python_dir / "mutli-scale-retinex.py"
        )
        msr_module = importlib.util.module_from_spec(msr_spec)
        msr_spec.loader.exec_module(msr_module)
        _multi_scale_retinex = msr_module.multi_scale_retinex
        
        logger.info("Python Retinex enhancement modules loaded successfully")
    except Exception as e:
        _retinex_import_error = str(e)
        logger.warning(f"Python Retinex enhancement modules could not be imported: {e}")
        _single_scale_retinex = None
        _multi_scale_retinex = None


def _load_lowlight_module():
    """
    Low-light LIME/DUAL modülünü lazy olarak yükler.
    python/lowlight_enhancement.py içindeki `enhance_image_exposure` fonksiyonunu kullanır.
    """
    global _lowlight_module, _lowlight_import_error

    if _lowlight_module is not None or _lowlight_import_error is not None:
        return

    try:
        python_dir = _get_python_dir()
        import importlib.util

        ll_spec = importlib.util.spec_from_file_location(
            "lowlight_enhancement",
            python_dir / "lowlight_enhancement.py",
        )
        ll_module = importlib.util.module_from_spec(ll_spec)
        ll_spec.loader.exec_module(ll_module)
        _lowlight_module = ll_module

        logger.info("Low-light LIME/DUAL module loaded successfully")
    except Exception as e:
        _lowlight_import_error = str(e)
        logger.warning(f"Low-light LIME/DUAL module could not be imported: {e}")
        _lowlight_module = None


def apply_clahe_to_image(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
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


def apply_gamma_to_image(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
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
    
    table = np.array([((i / 255.0) ** gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_img = cv2.LUT(img, table)
    return gamma_corrected_img


def apply_ssr_to_image(img: np.ndarray, sigma: int = 80) -> np.ndarray:
    """
    Apply SSR to numpy array image with safe normalization.
    
    Args:
        img: BGR format image (numpy array)
        sigma: Gaussian filter standard deviation
    
    Returns:
        Processed image
    """
    _load_retinex_modules()
    
    if _single_scale_retinex is None:
        raise RuntimeError(f"Single-scale retinex module not available: {_retinex_import_error}")
    
    # Validate sigma
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    try:
        b, g, r = cv2.split(img)
        b_retinex = _single_scale_retinex(b, sigma)
        g_retinex = _single_scale_retinex(g, sigma)
        r_retinex = _single_scale_retinex(r, sigma)
        ssr_output = cv2.merge([b_retinex, g_retinex, r_retinex])
        return ssr_output
    except ZeroDivisionError as e:
        logger.error(f"SSR division by zero error: {e}. Sigma: {sigma}, Image shape: {img.shape}")
        raise ValueError(f"SSR processing failed: division by zero. This may occur with uniform images. Try different parameters.")


def apply_msr_to_image(img: np.ndarray, sigma_list: list = [15, 80, 250]) -> np.ndarray:
    """
    Apply MSR to numpy array image with safe normalization.
    
    Args:
        img: BGR format image (numpy array)
        sigma_list: Sigma values list
    
    Returns:
        Processed image
    """
    _load_retinex_modules()
    
    if _multi_scale_retinex is None:
        raise RuntimeError(f"Multi-scale retinex module not available: {_retinex_import_error}")
    
    # Validate sigma_list
    if not sigma_list or len(sigma_list) == 0:
        raise ValueError("MSR sigma_list cannot be empty")
    
    for sigma in sigma_list:
        if sigma <= 0:
            raise ValueError(f"All sigma values must be positive, got {sigma_list}")
    
    try:
        b, g, r = cv2.split(img)
        b_msr = _multi_scale_retinex(b, sigma_list)
        g_msr = _multi_scale_retinex(g, sigma_list)
        r_msr = _multi_scale_retinex(r, sigma_list)
        msr_output = cv2.merge([b_msr, g_msr, r_msr])
        return msr_output
    except ZeroDivisionError as e:
        logger.error(f"MSR division by zero error: {e}. Sigma list: {sigma_list}, Image shape: {img.shape}")
        raise ValueError(f"MSR processing failed: division by zero. This may occur with uniform images or empty sigma list. Try different parameters.")


def apply_lowlight_lime(
    img: np.ndarray,
    gamma: float = 0.6,
    lambda_: float = 0.15,
    sigma: float = 3.0,
    bc: float = 1.0,
    bs: float = 1.0,
    be: float = 1.0,
) -> np.ndarray:
    """
    Gerçek LIME benzeri low-light iyileştirme.
    python/lowlight_enhancement.py içindeki `enhance_image_exposure` fonksiyonunu
    dual=False ile çağırır.
    """
    _load_lowlight_module()

    if _lowlight_module is None:
        raise RuntimeError(
            f"Low-light LIME/DUAL module not available: {_lowlight_import_error}"
        )

    # LIME: dual=False
    logger.debug(
        "Applying Low-light LIME with params: "
        f"gamma={gamma}, lambda_={lambda_}, sigma={sigma}, bc={bc}, bs={bs}, be={be}"
    )

    # lowlight_enhancement.enhance_image_exposure BGR uint8 bekler, kendi içinde normalize eder
    enhanced = _lowlight_module.enhance_image_exposure(
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
    img: np.ndarray,
    gamma: float = 0.6,
    lambda_: float = 0.15,
    sigma: float = 3.0,
    bc: float = 1.0,
    bs: float = 1.0,
    be: float = 1.0,
) -> np.ndarray:
    """
    Gerçek DUAL benzeri low-light iyileştirme.
    python/lowlight_enhancement.py içindeki `enhance_image_exposure` fonksiyonunu
    dual=True ile çağırır.
    """
    _load_lowlight_module()

    if _lowlight_module is None:
        raise RuntimeError(
            f"Low-light LIME/DUAL module not available: {_lowlight_import_error}"
        )

    logger.debug(
        "Applying Low-light DUAL with params: "
        f"gamma={gamma}, lambda_={lambda_}, sigma={sigma}, bc={bc}, bs={bs}, be={be}"
    )

    enhanced = _lowlight_module.enhance_image_exposure(
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


def apply_sharpen_to_image(
    img: np.ndarray, 
    method: str = 'unsharp', 
    strength: float = 1.0, 
    kernel_size: int = 5
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
    if method == 'unsharp':
        # Unsharp Masking method
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    elif method == 'laplacian':
        # Laplacian filter sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * strength
        sharpened = cv2.filter2D(img, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        sharpened = img.copy()
    
    return sharpened


class EnhancementService:
    """Image enhancement service for business logic"""
    
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
        gray_slice_low: int = 100,
        gray_slice_high: int = 180,
        use_bitplane: bool = False,
        bitplane_bit: int = 7,
        # Low-light seçenekleri
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
                    processed_img = apply_clahe_to_image(
                        processed_img,
                        clip_limit=params['clip_limit'],
                        tile_grid_size=params['tile_size']
                    )
                elif method_name == 'gamma':
                    processed_img = apply_gamma_to_image(
                        processed_img,
                        gamma=params['gamma']
                    )
                elif method_name == 'ssr':
                    processed_img = apply_ssr_to_image(
                        processed_img,
                        sigma=params['sigma']
                    )
                elif method_name == 'msr':
                    processed_img = apply_msr_to_image(
                        processed_img,
                        sigma_list=params['sigmas']
                    )
                elif method_name == 'sharpen':
                    processed_img = apply_sharpen_to_image(
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
                    low, high = params['low'], params['high']
                    mask = (gray >= low) & (gray <= high)
                    result = np.zeros_like(gray, dtype=np.uint8)
                    result[mask] = 255
                    processed_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                elif method_name == 'bitplane':
                    # Bit-plane slicing
                    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                    bit = params['bit']
                    plane = ((gray >> bit) & 1) * 255
                    processed_img = cv2.cvtColor(plane.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif method_name == 'lowlight_lime':
                    processed_img = apply_lowlight_lime(
                        processed_img,
                        gamma=params['gamma'],
                        lambda_=params['lambda_'],
                        sigma=params['sigma'],
                        bc=params['bc'],
                        bs=params['bs'],
                        be=params['be'],
                    )
                elif method_name == 'lowlight_dual':
                    processed_img = apply_lowlight_dual(
                        processed_img,
                        gamma=params['gamma'],
                        lambda_=params['lambda_'],
                        sigma=params['sigma'],
                        bc=params['bc'],
                        bs=params['bs'],
                        be=params['be'],
                    )
            
            # Encode to JPEG bytes
            _, encoded_img = cv2.imencode('.jpg', processed_img)
            return encoded_img.tobytes()
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            raise ValueError(f"Image enhancement failed: {str(e)}")

