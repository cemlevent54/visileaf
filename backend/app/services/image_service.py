"""
Image service - Image persistence (input/output images and enhancement params)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

import cv2
import numpy as np
from sqlmodel import Session

from app.models.image import Image, ImageCreate
from app.repositories.image_repository import ImageRepository
from app.services.audit_log_service import AuditLogService


class ImageService:
    """
    Image service for saving input/output images and enhancement parameters.

    Usage:
        - save_image(...) is called from controllers after enhancement is done.
    """

    def __init__(
        self,
        session: Session,
        upload_root: Optional[str] = None,
        audit_log_service: Optional[AuditLogService] = None,
    ):
        """
        Initialize service with database session and upload root.

        Args:
            session: SQLModel database session
            upload_root: Root directory for uploads (default: 'uploads')
        """
        self.session = session
        self.repository = ImageRepository(session)
        # Upload root relative to project root; can be made configurable later
        self.upload_root = upload_root or "uploads"
        # Audit log service (optional, created lazily if not provided)
        self.audit_log_service = audit_log_service or AuditLogService(session)

    def _ensure_directory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _build_paths(self, original_filename: str) -> Dict[str, str]:
        """
        Build input/output file paths based on current UTC timestamp and original extension.

        Returns:
            dict with keys: 'dir', 'input_path', 'output_path'
        """
        # Timestamp format: yyyy_dd_mm_hh_mm_ss
        ts = datetime.utcnow().strftime("%Y_%d_%m_%H_%M_%S")

        # Extract extension from original filename, default to .jpg if missing
        _, ext = os.path.splitext(original_filename or "")
        ext = ext.lower() if ext else ".jpg"

        directory = os.path.join(self.upload_root, ts)
        self._ensure_directory(directory)

        input_path = os.path.join(directory, f"input{ext}")
        output_path = os.path.join(directory, "output.jpg")  # Output is always JPEG in our pipeline

        return {
            "dir": directory,
            "input_path": input_path,
            "output_path": output_path,
        }

    def save_image(
        self,
        *,
        user_id: UUID,
        original_filename: str,
        input_bytes: bytes,
        output_bytes: bytes,
        enhancement_type: str,
        params: Dict[str, Any],
        parent_image_id: Optional[UUID] = None,
    ) -> Dict[str, Image]:
        """
        Save input/output images to disk and persist metadata to database.

        Args:
            user_id: Owner user UUID
            original_filename: Original uploaded file name (to derive extension)
            input_bytes: Raw uploaded image bytes
            output_img: Processed image as numpy array (BGR)
            enhancement_type: A short string describing the enhancement type (e.g. 'enhance', 'dcp', 'dcp_guided')
            params: Enhancement parameters dict (usually EnhancementParams.model_dump())
            parent_image_id: Optional parent Image UUID (if this is derived from another image)

        Returns:
            dict with 'input' and 'output' Image records
        """
        paths = self._build_paths(original_filename)

        # 1) Save input bytes to disk
        with open(paths["input_path"], "wb") as f:
            f.write(input_bytes)

        # Decode input for metadata (width/height)
        nparr_in = np.frombuffer(input_bytes, np.uint8)
        input_img = cv2.imdecode(nparr_in, cv2.IMREAD_COLOR)
        input_h, input_w = (input_img.shape[:2] if input_img is not None else (None, None))

        # 2) Save output image bytes to disk (JPEG)
        with open(paths["output_path"], "wb") as f:
            f.write(output_bytes)

        # Decode output for metadata
        nparr_out = np.frombuffer(output_bytes, np.uint8)
        output_img = cv2.imdecode(nparr_out, cv2.IMREAD_COLOR)
        output_h, output_w = (output_img.shape[:2] if output_img is not None else (None, None))

        # 3) Create input Image record
        input_image_create = ImageCreate(
            user_id=user_id,
            parent_image_id=parent_image_id,
            file_path=paths["input_path"],
            file_size=len(input_bytes),
            width=input_w,
            height=input_h,
            enhancement_type=None,
            params=None,
        )
        input_image = self.repository.create(input_image_create)

        # 4) Create output Image record with params
        # Approximate file size from disk
        output_file_size = os.path.getsize(paths["output_path"])
        output_image_create = ImageCreate(
            user_id=user_id,
            parent_image_id=input_image.id,
            file_path=paths["output_path"],
            file_size=output_file_size,
            width=output_w,
            height=output_h,
            enhancement_type=enhancement_type,
            params=params,
        )
        output_image = self.repository.create(output_image_create)
        # 5) Audit log
        if self.audit_log_service:
            try:
                # Log a generic image enhancement action
                self.audit_log_service.log_user_action(
                    action="image.enhance",
                    user_id=user_id,
                    details={
                        "enhancement_type": enhancement_type,
                        "input_image_id": str(input_image.id),
                        "output_image_id": str(output_image.id),
                        "input_path": input_image.file_path,
                        "output_path": output_image.file_path,
                        "params": params,
                    },
                )
            except Exception:
                # Audit log hatası ana akışı bozmamalı
                pass

        return {
            "input": input_image,
            "output": output_image,
        }

    def list_results_for_user(
        self,
        user_id: UUID,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List recent enhancement results for a user.

        Returns output images (children) with their input (parent) and params.
        """
        from app.models.image import Image as ImageModel  # avoid circular import hints
        from sqlmodel import select

        statement = (
            select(ImageModel)
            .where(
                ImageModel.user_id == user_id,
                ImageModel.parent_image_id.is_not(None),
            )
            .order_by(ImageModel.created_at.desc())
            .limit(limit)
        )
        results = self.session.exec(statement).all()

        items: List[Dict[str, Any]] = []
        for out_img in results:
            parent = out_img.parent_image
            items.append(
                {
                    "id": str(out_img.id),
                    "enhancement_type": out_img.enhancement_type,
                    "created_at": out_img.created_at.isoformat(),
                    "output_path": out_img.file_path,
                    "output_width": out_img.width,
                    "output_height": out_img.height,
                    "params": out_img.params or {},
                    "input": {
                        "id": str(parent.id) if parent else None,
                        "path": parent.file_path if parent else None,
                        "width": parent.width if parent else None,
                        "height": parent.height if parent else None,
                        "created_at": parent.created_at.isoformat() if parent else None,
                    },
                }
            )

        return items


