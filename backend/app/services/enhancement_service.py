"""
Enhancement service - Image enhancement business logic
"""
import sys
import os
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
_import_error = None


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
    global _single_scale_retinex, _multi_scale_retinex, _import_error
    
    if _single_scale_retinex is not None or _import_error is not None:
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
        
        logger.info("Python enhancement modules loaded successfully")
    except Exception as e:
        _import_error = str(e)
        logger.warning(f"Python enhancement modules could not be imported: {e}")
        _single_scale_retinex = None
        _multi_scale_retinex = None


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
        raise RuntimeError(f"Single-scale retinex module not available: {_import_error}")
    
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
        raise RuntimeError(f"Multi-scale retinex module not available: {_import_error}")
    
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
            
            # Encode to JPEG bytes
            _, encoded_img = cv2.imencode('.jpg', processed_img)
            return encoded_img.tobytes()
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            raise ValueError(f"Image enhancement failed: {str(e)}")

