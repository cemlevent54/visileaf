"""
Enhancement routes - Image enhancement endpoints
"""
import json
import logging
from fastapi import APIRouter, File, Form, UploadFile, Request, Body, HTTPException
from app.controllers.enhancement_controller import EnhancementController
from app.services.enhancement_service import EnhancementService
from app.schemas.enhancement import EnhancementParams
from app.config import get_logger
from pydantic import ValidationError

router = APIRouter(prefix="/api/enhancement", tags=["enhancement"])
logger = get_logger(__name__)


@router.post("/enhance")
async def enhance_image(
    request: Request,
    image: UploadFile = File(..., description="Image file to enhance"),
    params_json: str = Form(..., description="Enhancement parameters as JSON string")
):
    """
    Enhance image with specified methods.
    
    - **image**: Image file (JPEG, PNG, etc.)
    - **params_json**: Enhancement parameters as JSON string
    
    Example JSON:
    ```json
    {
        "use_gamma": true,
        "gamma": 0.5,
        "use_msr": true,
        "msr_sigmas": [15, 80, 250],
        "use_clahe": true,
        "clahe_clip": 2.5,
        "clahe_tile_size": [8, 8],
        "use_ssr": false,
        "ssr_sigma": 80,
        "use_sharpen": false,
        "sharpen_method": "unsharp",
        "sharpen_strength": 1.0,
        "sharpen_kernel_size": 5,
        "order": ["gamma", "msr", "clahe"],
        "use_lowlight_lime": true,
        "use_lowlight_dual": false,
        "lowlight_gamma": 0.6,
        "lowlight_lambda": 0.15,
        "lowlight_sigma": 3.0,
        "lowlight_bc": 1.0,
        "lowlight_bs": 1.0,
        "lowlight_be": 1.0
    }
    ```
    
    Returns:
        Enhanced image as downloadable JPEG file
    """
    # Log request
    logger.info(f"Enhancement request received - Image: {image.filename}, Size: {image.size if hasattr(image, 'size') else 'unknown'}")
    logger.debug(f"Request params_json: {params_json}")
    
    try:
        # Read image bytes
        image_bytes = await image.read()
        logger.debug(f"Image bytes read: {len(image_bytes)} bytes")
        
        # Parse JSON parameters
        try:
            params_dict = json.loads(params_json)
            logger.debug(f"Parsed params_dict: {params_dict}")
            params = EnhancementParams(**params_dict)
            logger.info(f"Enhancement params validated: use_gamma={params.use_gamma}, use_clahe={params.use_clahe}, use_msr={params.use_msr}, use_ssr={params.use_ssr}")
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            logger.error(f"JSON decode error: {error_msg}, params_json: {params_json}")
            raise HTTPException(status_code=400, detail=error_msg)
        except ValidationError as e:
            error_msg = f"Invalid parameters: {str(e)}"
            logger.error(f"Validation error: {error_msg}, errors: {e.errors()}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Create service and controller
        enhancement_service = EnhancementService()
        controller = EnhancementController(enhancement_service, request)
        
        # Process image
        logger.info("Starting image enhancement...")
        response = controller.enhance_image(image_bytes, params)
        logger.info("Image enhancement completed successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Value error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Image enhancement failed: {str(e)}"
        logger.exception(f"Unexpected error during enhancement: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/enhance-json")
async def enhance_image_json(
    request: Request,
    image: UploadFile = File(..., description="Image file to enhance"),
    params: EnhancementParams = Body(..., description="Enhancement parameters")
):
    """
    Enhance image with specified methods (alternative endpoint with JSON body).
    
    - **image**: Image file (JPEG, PNG, etc.) - sent as FormData
    - **params**: Enhancement parameters - sent as JSON body
    
    Note: This endpoint uses multipart/form-data for the image and JSON body for parameters.
    Some clients may have issues with this combination. Use /enhance endpoint for better compatibility.
    
    Returns:
        Enhanced image as downloadable JPEG file
    """
    # Log request
    logger.info(f"Enhancement request (JSON) received - Image: {image.filename}, Size: {image.size if hasattr(image, 'size') else 'unknown'}")
    logger.debug(f"Request params: {params.model_dump_json()}")
    
    try:
        # Read image bytes
        image_bytes = await image.read()
        logger.debug(f"Image bytes read: {len(image_bytes)} bytes")
        
        # Create service and controller
        enhancement_service = EnhancementService()
        controller = EnhancementController(enhancement_service, request)
        
        # Process image
        logger.info("Starting image enhancement...")
        response = controller.enhance_image(image_bytes, params)
        logger.info("Image enhancement completed successfully")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Value error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Image enhancement failed: {str(e)}"
        logger.exception(f"Unexpected error during enhancement: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

