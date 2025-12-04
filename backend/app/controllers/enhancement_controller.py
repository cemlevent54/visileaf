"""
Enhancement controller - HTTP request/response handling for image enhancement
"""
from fastapi import HTTPException, Request
from fastapi.responses import Response
from app.services.enhancement_service import EnhancementService
from app.schemas.enhancement import EnhancementParams
from app.config import gettext
import logging
from typing import Optional, Dict, Any
from uuid import UUID
from app.services.image_service import ImageService

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
    
    def _enhance_core(
        self,
        image_bytes: bytes,
        params: EnhancementParams
    ) -> bytes:
        """
        Core enhancement call that returns JPEG bytes.
        Separated so that routes can reuse it for persistence.
        """
        return self.enhancement_service.enhance_image(
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
            use_denoise=getattr(params, "use_denoise", False),
            denoise_strength=getattr(params, "denoise_strength", 3.0),
            # DCP tabanlı modlar (pipeline içinden çağrılabilir)
            use_dcp=getattr(params, "use_dcp", False),
            use_dcp_guided=getattr(params, "use_dcp_guided", False),
            # Low-light özel modları
            use_lowlight_lime=params.use_lowlight_lime,
            use_lowlight_dual=params.use_lowlight_dual,
            lowlight_gamma=params.lowlight_gamma,
            lowlight_lambda=params.lowlight_lambda,
            lowlight_sigma=params.lowlight_sigma,
            lowlight_bc=params.lowlight_bc,
            lowlight_bs=params.lowlight_bs,
            lowlight_be=params.lowlight_be,
            order=params.order,
        )

    def enhance_image(
        self,
        image_bytes: bytes,
        params: EnhancementParams
    ) -> Response:
        """
        Handle image enhancement request (without persistence).
        Kept for backward compatibility and tests.
        """
        try:
            enhanced_image_bytes = self._enhance_core(image_bytes, params)

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

    def enhance_image_with_persistence(
        self,
        image_bytes: bytes,
        params: EnhancementParams,
        image_service: ImageService,
        user_id: UUID,
        original_filename: str,
        enhancement_type: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Handle image enhancement and persist input/output + params via ImageService.
        """
        try:
            enhanced_image_bytes = self._enhance_core(image_bytes, params)

            # Persist input/output and params
            full_params: Dict[str, Any] = params.model_dump()
            if extra_params:
                full_params.update(extra_params)

            image_service.save_image(
                user_id=user_id,
                original_filename=original_filename,
                input_bytes=image_bytes,
                output_bytes=enhanced_image_bytes,
                enhancement_type=enhancement_type,
                params=full_params,
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

    def _enhance_dcp_core(
        self,
        image_bytes: bytes,
    ) -> bytes:
        """
        Core DCP enhancement that returns JPEG bytes.
        """
        return self.enhancement_service.enhance_image_with_dcp(
            image_bytes=image_bytes
        )

    def enhance_image_with_dcp(
        self,
        image_bytes: bytes,
    ) -> Response:
        """
        Handle Dark Channel Prior (DCP) based image enhancement request.
        """
        try:
            enhanced_image_bytes = self._enhance_dcp_core(image_bytes)

            return Response(
                content=enhanced_image_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": "attachment; filename=enhanced_image_dcp.jpg"
                }
            )

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"DCP image enhancement validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"DCP image enhancement failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=gettext("enhancement_controller.enhancement_failed", self.request) if self.request else "Image enhancement failed"
            )

    def _enhance_dcp_guided_core(
        self,
        image_bytes: bytes,
    ) -> bytes:
        """
        Core DCP + Guided Filter enhancement that returns JPEG bytes.
        """
        return self.enhancement_service.enhance_image_with_dcp_guided(
            image_bytes=image_bytes
        )

    def enhance_image_with_dcp_guided(
        self,
        image_bytes: bytes,
    ) -> Response:
        """
        Handle Dark Channel Prior + Guided Filter based image enhancement request.
        """
        try:
            enhanced_image_bytes = self._enhance_dcp_guided_core(image_bytes)

            return Response(
                content=enhanced_image_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": "attachment; filename=enhanced_image_dcp_guided.jpg"
                }
            )

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"DCP Guided image enhancement validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"DCP Guided image enhancement failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=gettext("enhancement_controller.enhancement_failed", self.request) if self.request else "Image enhancement failed"
            )

    def enhance_image_with_dcp_persistence(
        self,
        image_bytes: bytes,
        image_service: ImageService,
        user_id: UUID,
        original_filename: str,
        enhancement_type: str = "dcp",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        DCP enhancement + persistence.
        """
        try:
            enhanced_image_bytes = self._enhance_dcp_core(image_bytes)

            image_service.save_image(
                user_id=user_id,
                original_filename=original_filename,
                input_bytes=image_bytes,
                output_bytes=enhanced_image_bytes,
                enhancement_type=enhancement_type,
                params=extra_params or {},
            )

            return Response(
                content=enhanced_image_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": "attachment; filename=enhanced_image_dcp.jpg"
                }
            )

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"DCP image enhancement validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"DCP image enhancement failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=gettext("enhancement_controller.enhancement_failed", self.request) if self.request else "Image enhancement failed"
            )

    def enhance_image_with_dcp_guided_persistence(
        self,
        image_bytes: bytes,
        image_service: ImageService,
        user_id: UUID,
        original_filename: str,
        enhancement_type: str = "dcp_guided",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        DCP + Guided Filter enhancement + persistence.
        """
        try:
            enhanced_image_bytes = self._enhance_dcp_guided_core(image_bytes)

            image_service.save_image(
                user_id=user_id,
                original_filename=original_filename,
                input_bytes=image_bytes,
                output_bytes=enhanced_image_bytes,
                enhancement_type=enhancement_type,
                params=extra_params or {},
            )

            return Response(
                content=enhanced_image_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": "attachment; filename=enhanced_image_dcp_guided.jpg"
                }
            )

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"DCP Guided image enhancement validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"DCP Guided image enhancement failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=gettext("enhancement_controller.enhancement_failed", self.request) if self.request else "Image enhancement failed"
            )

