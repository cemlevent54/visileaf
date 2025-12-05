"""
Enhancement schemas - Request/Response models for image enhancement
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class EnhancementParams(BaseModel):
    """Enhancement parameters model"""
    use_gamma: bool = Field(default=False, description="Use gamma correction")
    gamma: float = Field(default=0.5, description="Gamma value (<1.0 brightens, >1.0 darkens)")
    
    use_clahe: bool = Field(default=False, description="Use CLAHE")
    clahe_clip: float = Field(default=3.0, description="CLAHE clip limit")
    clahe_tile_size: Optional[List[int]] = Field(
        default=[8, 8], 
        description="CLAHE tile grid size [width, height]"
    )
    
    use_ssr: bool = Field(default=False, description="Use Single-Scale Retinex")
    ssr_sigma: int = Field(default=80, description="SSR sigma value")
    
    use_msr: bool = Field(default=False, description="Use Multi-Scale Retinex")
    msr_sigmas: List[int] = Field(default=[15, 80, 250], description="MSR sigma values list")
    
    use_sharpen: bool = Field(default=False, description="Use sharpening")
    sharpen_method: str = Field(
        default="unsharp", 
        description="Sharpening method: 'unsharp' or 'laplacian'"
    )
    sharpen_strength: float = Field(
        default=1.0, 
        description="Sharpening strength (1.0 = normal, 2.0 = strong)"
    )
    sharpen_kernel_size: int = Field(
        default=5, 
        description="Kernel size for unsharp method"
    )
    
    # Eğitimlik temel dönüşümler
    use_negative: bool = Field(
        default=False,
        description="Apply classic negative image filter (invert intensities)"
    )
    use_threshold: bool = Field(
        default=False,
        description="Apply simple binary thresholding on grayscale image"
    )
    threshold_value: int = Field(
        default=128,
        description="Threshold value for binary thresholding (0-255)"
    )
    use_gray_slice: bool = Field(
        default=False,
        description="Apply gray-level slicing to highlight a specific intensity range"
    )
    gray_slice_low: float = Field(
        default=100.0,
        description="Lower bound for gray-level slicing (0.0-255.0)"
    )
    gray_slice_high: float = Field(
        default=180.0,
        description="Upper bound for gray-level slicing (0.0-255.0)"
    )
    use_bitplane: bool = Field(
        default=False,
        description="Apply bit-plane slicing on grayscale image"
    )
    bitplane_bit: int = Field(
        default=7,
        description="Bit index for bit-plane slicing (0-7)"
    )

    # Denoise (gürültü temizleme)
    use_denoise: bool = Field(
        default=False,
        description="Apply denoising to remove color noise (e.g., blue/red speckles)"
    )
    denoise_strength: float = Field(
        default=3.0,
        description="Denoising strength (3.0 = light, 10.0 = strong)"
    )

    # DCP tabanlı low-light modları
    use_dcp: bool = Field(
        default=False,
        description="Enable Dark Channel Prior (DCP) based low-light enhancement as a method in the pipeline"
    )
    use_dcp_guided: bool = Field(
        default=False,
        description="Enable Dark Channel Prior (DCP) + Guided Filter based low-light enhancement as a method in the pipeline"
    )
    
    order: Optional[List[str]] = Field(
        default=None, 
        description="Order of enhancement methods (e.g., ['gamma', 'msr', 'clahe', 'sharpen'])"
    )

    # Low-light özel modları (LIME / DUAL benzeri)
    use_lowlight_lime: bool = Field(
        default=False,
        description="Enable low-light enhancement (LIME-like, illumination-map-based)"
    )
    use_lowlight_dual: bool = Field(
        default=False,
        description="Enable low-light enhancement (DUAL-like, for under- and over-exposed regions)"
    )

    # Low-light parametreleri (Low-light-Image-Enhancement reposundaki argümanlara benzer)
    lowlight_gamma: float = Field(
        default=0.6,
        description="Low-light gamma correction parameter (similar to demo.py --gamma)"
    )
    lowlight_lambda: float = Field(
        default=0.15,
        alias="lowlight_lambda_",
        description="Weight for illumination refinement (similar to demo.py --lambda_)"
    )
    lowlight_sigma: float = Field(
        default=3.0,
        description="Spatial std for Gaussian weights (similar to demo.py --sigma)"
    )
    lowlight_bc: float = Field(
        default=1.0,
        description="Weight for Mertens contrast measure (similar to demo.py -bc)"
    )
    lowlight_bs: float = Field(
        default=1.0,
        description="Weight for Mertens saturation measure (similar to demo.py -bs)"
    )
    lowlight_be: float = Field(
        default=1.0,
        description="Weight for Mertens well-exposedness measure (similar to demo.py -be)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "use_gamma": True,
                "gamma": 0.5,
                "use_msr": True,
                "msr_sigmas": [15, 80, 250],
                "use_clahe": True,
                "clahe_clip": 2.5,
                "clahe_tile_size": [8, 8],
                "use_sharpen": False,
                "sharpen_method": "unsharp",
                "sharpen_strength": 1.0,
                "sharpen_kernel_size": 5,
                "use_negative": False,
                "use_threshold": False,
                "threshold_value": 128,
                "use_gray_slice": False,
                "gray_slice_low": 100,
                "gray_slice_high": 180,
                "use_bitplane": False,
                "bitplane_bit": 7,
                "use_denoise": False,
                "denoise_strength": 3.0,
                "use_dcp": False,
                "use_dcp_guided": False,
                "order": ["gamma", "msr", "clahe"],
                "use_lowlight_lime": True,
                "use_lowlight_dual": False,
                "lowlight_gamma": 0.6,
                "lowlight_lambda": 0.15,
                "lowlight_sigma": 3.0,
                "lowlight_bc": 1.0,
                "lowlight_bs": 1.0,
                "lowlight_be": 1.0
            }
        }

