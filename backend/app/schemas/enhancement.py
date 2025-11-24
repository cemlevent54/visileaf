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
    
    order: Optional[List[str]] = Field(
        default=None, 
        description="Order of enhancement methods (e.g., ['gamma', 'msr', 'clahe', 'sharpen'])"
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
                "order": ["gamma", "msr", "clahe"]
            }
        }

