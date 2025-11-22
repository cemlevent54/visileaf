"""
AuditLog model - Sistem log kayıtları
"""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy.types import JSON
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .user import User


class AuditLogBase(SQLModel):
    """Base audit log model"""
    action: str = Field(max_length=255)
    # details field'ı - JSON olarak saklanır (metadata rezervli olduğu için details kullanıyoruz)
    details: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON)
    )


class AuditLog(AuditLogBase, table=True):
    """AuditLog tablo modeli"""
    __tablename__ = "audit_logs"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: Optional[UUID] = Field(default=None, foreign_key="users.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    user: Optional["User"] = Relationship(back_populates="audit_logs")


class AuditLogCreate(AuditLogBase):
    """Yeni log oluşturma modeli"""
    user_id: Optional[UUID] = None


class AuditLogResponse(AuditLogBase):
    """Log yanıt modeli"""
    id: UUID
    user_id: Optional[UUID] = None
    created_at: datetime

