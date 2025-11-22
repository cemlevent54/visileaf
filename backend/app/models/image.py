"""
Image model - Görüntü bilgileri ve enhancement parametreleri
"""
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy.types import JSON
from uuid import UUID, uuid4
import enum

if TYPE_CHECKING:
    from .user import User


class EnhancementType(str, enum.Enum):
    """Enhancement yöntem tipleri"""
    CLAHE = "clahe"
    GAMMA = "gamma"
    SSR = "ssr"
    MSR = "msr"
    SHARPEN = "sharpen"
    HYBRID = "hybrid"


class EnhancementParams(SQLModel):
    """Enhancement parametreleri için schema"""
    methods: Optional[List[str]] = None  # Kullanılan yöntemler: ["clahe", "gamma"]
    order: Optional[List[str]] = None  # Uygulama sırası: ["gamma", "clahe"]
    
    # CLAHE parametreleri
    clahe: Optional[Dict[str, Any]] = None
    # Örnek: {"clip_limit": 3.0, "tile_size": [8, 8]}
    
    # Gamma parametreleri
    gamma: Optional[Dict[str, Any]] = None
    # Örnek: {"value": 0.5}
    
    # SSR parametreleri
    ssr: Optional[Dict[str, Any]] = None
    # Örnek: {"sigma": 80}
    
    # MSR parametreleri
    msr: Optional[Dict[str, Any]] = None
    # Örnek: {"sigmas": [15, 80, 250]}
    
    # Sharpening parametreleri
    sharpen: Optional[Dict[str, Any]] = None
    # Örnek: {"method": "unsharp", "strength": 1.0, "kernel_size": 5}


class ImageBase(SQLModel):
    """Base image model"""
    file_path: str = Field(max_length=1000)
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    enhancement_type: Optional[str] = Field(default=None, max_length=50)
    # params field'ı - JSON olarak saklanır
    params: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON)
    )


class Image(ImageBase, table=True):
    """Image tablo modeli"""
    __tablename__ = "images"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id")
    parent_image_id: Optional[UUID] = Field(default=None, foreign_key="images.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    user: "User" = Relationship(back_populates="images")
    parent_image: Optional["Image"] = Relationship(
        back_populates="child_images",
        sa_relationship_kwargs={"remote_side": "Image.id"}
    )
    child_images: list["Image"] = Relationship(back_populates="parent_image")


class ImageCreate(ImageBase):
    """Yeni görüntü oluşturma modeli"""
    user_id: UUID
    parent_image_id: Optional[UUID] = None


class ImageUpdate(SQLModel):
    """Görüntü güncelleme modeli"""
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    enhancement_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class ImageResponse(ImageBase):
    """Görüntü yanıt modeli"""
    id: UUID
    user_id: UUID
    parent_image_id: Optional[UUID] = None
    created_at: datetime

