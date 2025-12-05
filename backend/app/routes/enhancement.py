"""
Enhancement routes - Image enhancement endpoints
"""
import json
import logging
from fastapi import APIRouter, File, Form, UploadFile, Request, Body, HTTPException, Header, Depends
from app.controllers.enhancement_controller import EnhancementController
from app.services.enhancement_service import EnhancementService
from app.services.image_service import ImageService
from app.services.auth_service import AuthService
from app.schemas.enhancement import EnhancementParams
from app.config import get_logger, get_session
from pydantic import ValidationError
from sqlmodel import Session

router = APIRouter(prefix="/api/enhancement", tags=["enhancement"])
logger = get_logger(__name__)


@router.post("/enhance")
async def enhance_image(
    request: Request,
    image: UploadFile = File(..., description="Image file to enhance"),
    params_json: str = Form(..., description="Enhancement parameters as JSON string"),
    authorization: str = Header(..., alias="Authorization"),
    session: Session = Depends(get_session),
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
        # Auth & services
        auth_service = AuthService(session)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")
        access_token = authorization.replace("Bearer ", "")
        current_user = auth_service.get_current_user(access_token)
        user_id = current_user["user"]["id"]

        image_service = ImageService(session)

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
        
        # Process image + persist
        logger.info("Starting image enhancement...")
        response = controller.enhance_image_with_persistence(
            image_bytes=image_bytes,
            params=params,
            image_service=image_service,
            user_id=user_id,
            original_filename=image.filename,
            enhancement_type="enhance",
        )
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


@router.post("/enhance-with-dcp")
async def enhance_image_with_dcp(
    request: Request,
    image: UploadFile = File(..., description="Image file to enhance with Dark Channel Prior (DCP)"),
    params_json: str | None = Form(
        None,
        description="Optional enhancement parameters as JSON string (currently not used by DCP algorithm, reserved for future use)",
    ),
    authorization: str = Header(..., alias="Authorization"),
    session: Session = Depends(get_session),
):
    """
    Enhance image using Dark Channel Prior (DCP) based low-light enhancement.

    - **image**: Image file (JPEG, PNG, etc.)
    - **params_json** (optional): Enhancement parameters as JSON string.
    
    Not: `params_json` şu an algoritma içinde kullanılmıyor, ileride DCP parametrelerini
    (ör. patch_size, omega, t_min vb.) kontrol etmek için ayrılmıştır.
    """
    logger.info(
        f"DCP Enhancement request received - Image: {image.filename}, Size: {image.size if hasattr(image, 'size') else 'unknown'}"
    )
    if params_json is not None:
        logger.debug(f"DCP params_json (optional): {params_json}")

    try:
        # Auth & services
        auth_service = AuthService(session)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")
        access_token = authorization.replace("Bearer ", "")
        current_user = auth_service.get_current_user(access_token)
        user_id = current_user["user"]["id"]

        image_service = ImageService(session)

        # Read image bytes
        image_bytes = await image.read()
        logger.debug(f"DCP Image bytes read: {len(image_bytes)} bytes")

        # Parse JSON parameters if provided (optional, same validation as /enhance)
        params = None
        if params_json is not None:
            try:
                params_dict = json.loads(params_json)
                logger.debug(f"DCP parsed params_dict: {params_dict}")
                params = EnhancementParams(**params_dict)
                # DCP endpoint için bayrakları zorunlu kıl
                params.use_dcp = True
                params.use_dcp_guided = False
                logger.info(
                    "DCP params validated "
                    f"use_gamma={params.use_gamma}, use_clahe={params.use_clahe}, "
                    f"use_msr={params.use_msr}, use_ssr={params.use_ssr}, "
                    f"use_lowlight_lime={params.use_lowlight_lime}, use_lowlight_dual={params.use_lowlight_dual}, "
                    f"use_dcp={params.use_dcp}"
                )
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON format for params_json: {str(e)}"
                logger.error(f"DCP JSON decode error: {error_msg}, params_json: {params_json}")
                raise HTTPException(status_code=400, detail=error_msg)
            except ValidationError as e:
                error_msg = f"Invalid parameters for DCP: {str(e)}"
                logger.error(f"DCP validation error: {error_msg}, errors: {e.errors()}")
                raise HTTPException(status_code=400, detail=error_msg)

        # Create service and controller
        enhancement_service = EnhancementService()
        controller = EnhancementController(enhancement_service, request)
        
        # Process image + persist
        if params is not None:
            # /enhance ile aynı pipeline, DCP method'u dahil
            logger.info("Starting DCP-based image enhancement via main pipeline...")
            response = controller.enhance_image_with_persistence(
                image_bytes=image_bytes,
                params=params,
                image_service=image_service,
                user_id=user_id,
                original_filename=image.filename,
                enhancement_type="dcp_pipeline",
            )
            logger.info("DCP-based image enhancement via pipeline completed successfully")
            return response
        else:
            # Eski davranış: sadece DCP uygula (ama artık DB'ye de kaydediyoruz)
            logger.info("Starting DCP-based image enhancement (standalone)...")
            response = controller.enhance_image_with_dcp_persistence(
                image_bytes=image_bytes,
                image_service=image_service,
                user_id=user_id,
                original_filename=image.filename,
                enhancement_type="dcp",
            )
            logger.info("DCP-based image enhancement (standalone) completed successfully")
            return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"DCP Value error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"DCP image enhancement failed: {str(e)}"
        logger.exception(f"Unexpected error during DCP enhancement: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/dcp-guided-filter")
async def enhance_image_with_dcp_guided(
    request: Request,
    image: UploadFile = File(..., description="Image file to enhance with Dark Channel Prior (DCP) + Guided Filter"),
    params_json: str | None = Form(
        None,
        description="Optional enhancement parameters as JSON string (currently not used by DCP+Guided algorithm, reserved for future use)",
    ),
    authorization: str = Header(..., alias="Authorization"),
    session: Session = Depends(get_session),
):
    """
    Enhance image using Dark Channel Prior (DCP) + Guided Filter based advanced low-light enhancement.

    - **image**: Image file (JPEG, PNG, etc.)
    - **params_json** (optional): Enhancement parameters as JSON string.
    
    Not: `params_json` şu an algoritma içinde kullanılmıyor, ileride patch_size, radius,
    eps vb. guided filter / DCP parametrelerini kontrol etmek için ayrılmıştır.
    """
    logger.info(
        f"DCP Guided Enhancement request received - Image: {image.filename}, Size: {image.size if hasattr(image, 'size') else 'unknown'}"
    )
    if params_json is not None:
        logger.debug(f"DCP Guided params_json (optional): {params_json}")

    try:
        # Auth & services
        auth_service = AuthService(session)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")
        access_token = authorization.replace("Bearer ", "")
        current_user = auth_service.get_current_user(access_token)
        user_id = current_user["user"]["id"]

        image_service = ImageService(session)

        # Read image bytes
        image_bytes = await image.read()
        logger.debug(f"DCP Guided Image bytes read: {len(image_bytes)} bytes")

        # Parse JSON parameters if provided (optional, same validation as /enhance)
        params = None
        if params_json is not None:
            try:
                params_dict = json.loads(params_json)
                logger.debug(f"DCP Guided parsed params_dict: {params_dict}")
                params = EnhancementParams(**params_dict)
                # DCP Guided endpoint için bayrakları zorunlu kıl
                params.use_dcp = False
                params.use_dcp_guided = True
                logger.info(
                    "DCP Guided params validated "
                    f"use_gamma={params.use_gamma}, use_clahe={params.use_clahe}, "
                    f"use_msr={params.use_msr}, use_ssr={params.use_ssr}, "
                    f"use_lowlight_lime={params.use_lowlight_lime}, use_lowlight_dual={params.use_lowlight_dual}, "
                    f"use_dcp_guided={params.use_dcp_guided}"
                )
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON format for params_json: {str(e)}"
                logger.error(f"DCP Guided JSON decode error: {error_msg}, params_json: {params_json}")
                raise HTTPException(status_code=400, detail=error_msg)
            except ValidationError as e:
                error_msg = f"Invalid parameters for DCP Guided: {str(e)}"
                logger.error(f"DCP Guided validation error: {error_msg}, errors: {e.errors()}")
                raise HTTPException(status_code=400, detail=error_msg)

        # Create service and controller
        enhancement_service = EnhancementService()
        controller = EnhancementController(enhancement_service, request)
        
        # Process image
        if params is not None:
            logger.info("Starting DCP + Guided Filter enhancement via main pipeline...")
            response = controller.enhance_image_with_persistence(
                image_bytes=image_bytes,
                params=params,
                image_service=image_service,
                user_id=user_id,
                original_filename=image.filename,
                enhancement_type="dcp_guided_pipeline",
            )
            logger.info("DCP + Guided Filter enhancement via pipeline completed successfully")
            return response
        else:
            logger.info("Starting DCP + Guided Filter enhancement (standalone)...")
            response = controller.enhance_image_with_dcp_guided_persistence(
                image_bytes=image_bytes,
                image_service=image_service,
                user_id=user_id,
                original_filename=image.filename,
                enhancement_type="dcp_guided",
            )
            logger.info("DCP + Guided Filter enhancement (standalone) completed successfully")
            return response

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"DCP Guided Value error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"DCP Guided image enhancement failed: {str(e)}"
        logger.exception(f"Unexpected error during DCP Guided enhancement: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/results")
async def list_enhancement_results(
    authorization: str = Header(..., alias="Authorization"),
    session: Session = Depends(get_session),
    limit: int = 100,
):
    """
    List recent enhancement results for the current user.

    Returns:
        List of records containing input/output image info and params.
    """
    try:
        auth_service = AuthService(session)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")
        access_token = authorization.replace("Bearer ", "")
        current_user = auth_service.get_current_user(access_token)
        user_id = current_user["user"]["id"]

        image_service = ImageService(session)
        results = image_service.list_results_for_user(user_id=user_id, limit=limit)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to list enhancement results: {e}")
        raise HTTPException(status_code=500, detail="Failed to list enhancement results")


@router.post("/results/{image_id}/toggle-star")
async def toggle_star_result(
    image_id: str,
    authorization: str = Header(..., alias="Authorization"),
    session: Session = Depends(get_session),
):
    """
    Toggle star status for an enhancement result.
    
    - **image_id**: Output image UUID (the enhancement result to star/unstar)
    
    Returns:
        dict with 'id' and 'is_starred' status
    """
    try:
        from uuid import UUID
        
        auth_service = AuthService(session)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required")
        access_token = authorization.replace("Bearer ", "")
        current_user = auth_service.get_current_user(access_token)
        user_id = current_user["user"]["id"]

        try:
            image_uuid = UUID(image_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image ID format")

        image_service = ImageService(session)
        result = image_service.toggle_star(image_id=image_uuid, user_id=user_id)
        return result
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Toggle star error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to toggle star: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle star")

