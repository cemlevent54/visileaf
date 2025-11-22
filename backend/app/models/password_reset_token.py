"""
PasswordResetToken model - Şifre sıfırlama token'ları
"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .user import User


class PasswordResetTokenBase(SQLModel):
    """Base password reset token model"""
    token_hash: str = Field(max_length=500)
    code: str = Field(max_length=6)  # 6-digit verification code
    expires_at: datetime
    used: bool = Field(default=False)


class PasswordResetToken(PasswordResetTokenBase, table=True):
    """PasswordResetToken tablo modeli"""
    __tablename__ = "password_reset_tokens"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="users.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    user: "User" = Relationship(back_populates="password_reset_tokens")


class PasswordResetTokenCreate(PasswordResetTokenBase):
    """Yeni token oluşturma modeli"""
    user_id: UUID


class PasswordResetTokenResponse(PasswordResetTokenBase):
    """Token yanıt modeli"""
    id: UUID
    user_id: UUID
    created_at: datetime

