"""
Image service - Image persistence (input/output images and enhancement params)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
import io

import cv2
import numpy as np
from sqlmodel import Session
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

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
            .order_by(ImageModel.is_starred.desc(), ImageModel.created_at.desc())
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
                    "is_starred": out_img.is_starred,
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

    def toggle_star(self, image_id: UUID, user_id: UUID) -> Dict[str, Any]:
        """
        Toggle star status for an enhancement result (output image).
        
        Args:
            image_id: Output image UUID (the enhancement result to star/unstar)
            user_id: User UUID (to verify ownership)
            
        Returns:
            dict with 'is_starred' status
        """
        image = self.repository.get_by_id(image_id)
        if not image:
            raise ValueError(f"Image with id {image_id} not found")
        
        # Verify ownership
        if image.user_id != user_id:
            raise ValueError("You can only star your own images")
        
        # Only output images (with parent) can be starred
        if image.parent_image_id is None:
            raise ValueError("Only enhancement results (output images) can be starred")
        
        # Toggle star status
        image.is_starred = not image.is_starred
        self.session.add(image)
        self.session.commit()
        self.session.refresh(image)
        
        return {
            "id": str(image.id),
            "is_starred": image.is_starred
        }

    def delete_result(self, image_id: UUID, user_id: UUID) -> Dict[str, Any]:
        """
        Hard delete an enhancement result (output image) and its associated files.
        
        - Only output images (with parent_image_id) can be deleted.
        - Verifies ownership by user_id.
        - Deletes both DB record(s) and image file(s) from disk.
        """
        image = self.repository.get_by_id(image_id)
        if not image:
            raise ValueError(f"Image with id {image_id} not found")

        # Verify ownership
        if image.user_id != user_id:
            raise ValueError("You can only delete your own images")

        # Only output images (with parent) are shown in results and can be deleted
        if image.parent_image_id is None:
            raise ValueError("Only enhancement results (output images) can be deleted")

        # Collect file paths (output + input) to delete from disk
        file_paths: List[str] = []
        if image.file_path:
            file_paths.append(image.file_path)

        parent = image.parent_image
        if parent and parent.file_path:
            file_paths.append(parent.file_path)

        # Delete files from disk
        for path in file_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                # Dosya silme hatası ana akışı bozmamalı, sadece logla
                # (ör. dosya zaten silinmiş olabilir)
                pass

        # Optionally try to remove now-empty directory for the output image
        try:
            if image.file_path:
                directory = os.path.dirname(image.file_path)
                if directory and os.path.isdir(directory) and not os.listdir(directory):
                    os.rmdir(directory)
        except Exception:
            # Klasör silme hatası ana akışı bozmamalı
            pass

        # Delete DB records: delete output first, then its input image
        self.repository.delete(image_id)
        if parent:
            self.repository.delete(parent.id)

        # Audit log
        if self.audit_log_service:
            try:
                self.audit_log_service.log_user_action(
                    action="image.delete",
                    user_id=user_id,
                    details={
                        "output_image_id": str(image_id),
                        "input_image_id": str(parent.id) if parent else None,
                        "output_path": image.file_path,
                        "input_path": parent.file_path if parent else None,
                    },
                )
            except Exception:
                # Audit log hatası ana akışı bozmamalı
                pass

        return {
            "id": str(image_id),
            "deleted": True,
        }

    def export_result_pdf(self, image_id: UUID, user_id: UUID) -> bytes:
        """
        Export a single enhancement result (output image) as a PDF containing:
        - Input and output images side by side
        - Parameters listed below the images
        """
        image = self.repository.get_by_id(image_id)
        if not image:
            raise ValueError(f"Image with id {image_id} not found")

        # Verify ownership
        if image.user_id != user_id:
            raise ValueError("You can only export your own images")

        # Only output images (with parent) can be exported
        if image.parent_image_id is None:
            raise ValueError("Only enhancement results (output images) can be exported")

        parent = image.parent_image
        if parent is None:
            raise ValueError("Input image record not found for this result")

        # Validate file paths
        if not parent.file_path or not os.path.exists(parent.file_path):
            raise ValueError("Input image file not found on disk")
        if not image.file_path or not os.path.exists(image.file_path):
            raise ValueError("Output image file not found on disk")

        params = image.params or {}

        buffer = io.BytesIO()
        page_width, page_height = landscape(A4)
        c = canvas.Canvas(buffer, pagesize=landscape(A4))

        margin = 40
        available_width = page_width - 2 * margin
        available_height = page_height - 2 * margin

        # Top centered title
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(page_width / 2.0, page_height - margin / 2.0, "Export Results")

        # Allocate ~60% of height for images
        image_area_height = available_height * 0.6
        image_max_height = image_area_height
        gutter = 20.0
        each_width = (available_width - gutter) / 2.0

        def _draw_image(path: str, x_left: float) -> float:
            """Draw an image scaled to fit within each_width x image_max_height.
            Returns bottom y of drawn image."""
            img = cv2.imread(path)
            if img is None:
                return page_height - margin - image_max_height
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                return page_height - margin - image_max_height
            scale = min(each_width / float(w), image_max_height / float(h))
            draw_w = w * scale
            draw_h = h * scale
            x = x_left + (each_width - draw_w) / 2.0
            # Start images a bit lower to leave space under the title
            top_y = page_height - margin - 20
            y = top_y - draw_h
            c.drawImage(
                path,
                x,
                y,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                anchor="sw",
            )
            return y

        # Draw input and output images
        left_x = margin
        right_x = margin + each_width + gutter

        y_input = _draw_image(parent.file_path, left_x)
        y_output = _draw_image(image.file_path, right_x)
        images_bottom_y = min(y_input, y_output)

        # Labels above images
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_x, page_height - margin + 5, "Input")
        c.drawString(right_x, page_height - margin + 5, "Output")

        # Parameters section (with a bit more space under images)
        c.setFont("Helvetica-Bold", 11)
        params_title_y = images_bottom_y - 30
        if params_title_y < margin + 40:
            params_title_y = margin + 40
        c.drawString(margin, params_title_y, "Parameters")

        c.setFont("Helvetica", 9)
        text_y = params_title_y - 14
        line_height = 11
        max_lines = int((text_y - margin) / line_height)

        def _format_value(v: Any) -> str:
            if isinstance(v, (dict, list)):
                import json

                return json.dumps(v, ensure_ascii=False)
            return str(v)

        lines: list[str] = []
        for key, value in params.items():
            value_str = _format_value(value)
            line = f"{key}: {value_str}"
            lines.append(line)

        # Wrap long lines
        max_chars = 110
        wrapped: list[str] = []
        for line in lines:
            if len(line) <= max_chars:
                wrapped.append(line)
            else:
                while len(line) > max_chars:
                    wrapped.append(line[:max_chars])
                    line = line[max_chars:]
                if line:
                    wrapped.append(line)

        for i, line in enumerate(wrapped[:max_lines]):
            c.drawString(margin, text_y - i * line_height, line)

        c.showPage()
        c.save()
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes


