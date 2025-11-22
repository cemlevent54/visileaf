"""
User model - Kullanıcı bilgileri
"""
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship
from uuid import UUID, uuid4
import enum


class UserRole(str, enum.Enum):
    """Kullanıcı rolleri"""
    USER = "user"
    ADMIN = "admin"


class UserBase(SQLModel):
    """Base user model"""
    first_name: Optional[str] = Field(default=None, max_length=255)
    last_name: Optional[str] = Field(default=None, max_length=255)
    email: str = Field(unique=True, index=True, max_length=255)
    password_hash: str = Field(max_length=500)
    role: UserRole = Field(default=UserRole.USER)


class User(UserBase, table=True):
    """User tablo modeli"""
    __tablename__ = "users"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    images: list["Image"] = Relationship(back_populates="user")
    password_reset_tokens: list["PasswordResetToken"] = Relationship(back_populates="user")
    audit_logs: list["AuditLog"] = Relationship(back_populates="user")


class UserCreate(UserBase):
    """Yeni kullanıcı oluşturma modeli"""
    pass


class UserUpdate(SQLModel):
    """Kullanıcı güncelleme modeli"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None


class UserResponse(UserBase):
    """Kullanıcı yanıt modeli (password_hash hariç)"""
    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

