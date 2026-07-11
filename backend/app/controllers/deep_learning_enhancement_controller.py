"""
Deep learning tabanlı enhancement controller.

Şimdilik EnlightenGAN entegrasyonunu çalıştırır, sonucu DB'ye kaydeder.
"""
from fastapi import HTTPException, Request
from fastapi.responses import Response
import logging
from uuid import UUID
from typing import Optional, Dict, Any
from app.services.image_service import ImageService
from app.services.enhance_with_deep_learning_service import EnhanceWithDeepLearningService

logger = logging.getLogger(__name__)


class DeepLearningEnhancementController:
    """
    Deep learning tabanlı enhancement controller.
    """

    def __init__(
        self,
        dl_service: EnhanceWithDeepLearningService,
        request: Optional[Request] = None,
    ):
        self.request = request
        self.dl_service = dl_service

    def enhance_with_model_and_persistence(
        self,
        *,
        image_bytes: bytes,
        model_name: str,
        image_service: ImageService,
        user_id: UUID,
        original_filename: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """
        Model seçimine göre DL pipeline'ı çalıştırır, sonucu kaydeder ve döndürür.
        """
        try:
            if not model_name:
                raise HTTPException(status_code=400, detail="model_name zorunludur")

            output_bytes = self.dl_service.enhance_with_model(
                image_bytes=image_bytes,
                model_name=model_name,
                original_filename=original_filename,
            )

            if not output_bytes or len(output_bytes) == 0:
                raise ValueError("Deep learning model çıktısı boş")

            params_to_save: Dict[str, Any] = {"model_name": model_name}
            if extra_params:
                params_to_save.update(extra_params)

            enhancement_type = f"dl_{model_name}"
            
            # Persist input/output and params
            image_service.save_image(
                user_id=user_id,
                original_filename=original_filename,
                input_bytes=image_bytes,
                output_bytes=output_bytes,
                enhancement_type=enhancement_type,
                params=params_to_save,
            )

            return Response(
                content=output_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="enhanced_{original_filename}"'
                },
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Deep learning enhancement validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Deep learning enhancement failed: {str(e)}"
            logger.exception(f"Unexpected error during DL enhancement: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

