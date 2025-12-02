"""
Enhancement controller - HTTP request/response handling for image enhancement
"""
from fastapi import HTTPException, Request
from fastapi.responses import Response
from app.services.enhancement_service import EnhancementService
from app.schemas.enhancement import EnhancementParams
from app.config import gettext
import logging

logger = logging.getLogger(__name__)


class EnhancementController:
    """Image enhancement controller for HTTP handling"""
    
    def __init__(self, enhancement_service: EnhancementService, request: Request = None):
        """
        Initialize controller with service
        
        Args:
            enhancement_service: EnhancementService instance
            request: FastAPI Request object for i18n
        """
        self.enhancement_service = enhancement_service
        self.request = request
    
    def enhance_image(
        self,
        image_bytes: bytes,
        params: EnhancementParams
    ) -> Response:
        """
        Handle image enhancement request
        
        Args:
            image_bytes: Image file bytes
            params: Enhancement parameters
        
        Returns:
            Response with enhanced image as downloadable JPEG
        
        Raises:
            HTTPException: If enhancement fails
        """
        try:
            enhanced_image_bytes = self.enhancement_service.enhance_image(
                image_bytes=image_bytes,
                use_gamma=params.use_gamma,
                gamma=params.gamma,
                use_clahe=params.use_clahe,
                clahe_clip=params.clahe_clip,
                clahe_tile_size=params.clahe_tile_size or [8, 8],
                use_ssr=params.use_ssr,
                ssr_sigma=params.ssr_sigma,
                use_msr=params.use_msr,
                msr_sigmas=params.msr_sigmas,
                use_sharpen=params.use_sharpen,
                sharpen_method=params.sharpen_method,
                sharpen_strength=params.sharpen_strength,
                sharpen_kernel_size=params.sharpen_kernel_size,
                # Eğitimlik temel dönüşümler
                use_negative=params.use_negative,
                use_threshold=params.use_threshold,
                threshold_value=params.threshold_value,
                use_gray_slice=params.use_gray_slice,
                gray_slice_low=params.gray_slice_low,
                gray_slice_high=params.gray_slice_high,
                use_bitplane=params.use_bitplane,
                bitplane_bit=params.bitplane_bit,
                # Low-light özel modları
                use_lowlight_lime=params.use_lowlight_lime,
                use_lowlight_dual=params.use_lowlight_dual,
                lowlight_gamma=params.lowlight_gamma,
                lowlight_lambda=params.lowlight_lambda,
                lowlight_sigma=params.lowlight_sigma,
                lowlight_bc=params.lowlight_bc,
                lowlight_bs=params.lowlight_bs,
                lowlight_be=params.lowlight_be,
                order=params.order
            )
            
            return Response(
                content=enhanced_image_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": "attachment; filename=enhanced_image.jpg"
                }
            )
            
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Image enhancement validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Image enhancement failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=gettext("enhancement_controller.enhancement_failed", self.request) if self.request else "Image enhancement failed"
            )

