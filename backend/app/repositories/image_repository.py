"""
Image repository - Image model CRUD operations
"""
from typing import Optional
from uuid import UUID
from sqlmodel import Session, select
from app.models.image import Image, ImageCreate, ImageUpdate


class ImageRepository:
    """Image repository for database operations"""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session
        
        Args:
            session: SQLModel database session
        """
        self.session = session
    
    def create(self, image_data: ImageCreate) -> Image:
        """
        Create a new image record.
        
        Args:
            image_data: Image creation data
        
        Returns:
            Created Image object
        """
        image = Image(**image_data.model_dump())
        self.session.add(image)
        self.session.commit()
        self.session.refresh(image)
        return image
    
    def get_by_id(self, image_id: UUID) -> Optional[Image]:
        """
        Get image by ID.
        
        Args:
            image_id: Image UUID
        
        Returns:
            Image object or None if not found
        """
        statement = select(Image).where(Image.id == image_id)
        return self.session.exec(statement).first()
    
    def update(self, image_id: UUID, image_data: ImageUpdate) -> Optional[Image]:
        """
        Update image.
        
        Args:
            image_id: Image UUID
            image_data: Image update data
        
        Returns:
            Updated Image object or None if not found
        """
        image = self.get_by_id(image_id)
        if not image:
            return None
        
        update_data = image_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(image, field, value)
        
        self.session.add(image)
        self.session.commit()
        self.session.refresh(image)
        return image

