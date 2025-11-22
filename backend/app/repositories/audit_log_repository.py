"""
Audit log repository - Audit log CRUD operations
"""
from typing import Optional, List
from uuid import UUID
from sqlmodel import Session, select
from app.models.audit_log import AuditLog, AuditLogCreate


class AuditLogRepository:
    """Audit log repository for database operations"""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session
        
        Args:
            session: SQLModel database session
        """
        self.session = session
    
    def create(self, log_data: AuditLogCreate) -> AuditLog:
        """
        Create a new audit log entry
        
        Args:
            log_data: Audit log creation data
        
        Returns:
            Created audit log object
        """
        log = AuditLog(**log_data.model_dump())
        self.session.add(log)
        self.session.commit()
        self.session.refresh(log)
        return log
    
    def get_by_id(self, log_id: UUID) -> Optional[AuditLog]:
        """
        Get audit log by ID
        
        Args:
            log_id: Audit log UUID
        
        Returns:
            Audit log object or None if not found
        """
        statement = select(AuditLog).where(AuditLog.id == log_id)
        return self.session.exec(statement).first()
    
    def get_by_user_id(self, user_id: UUID, limit: int = 100) -> List[AuditLog]:
        """
        Get audit logs by user ID
        
        Args:
            user_id: User UUID
            limit: Maximum number of logs to return
        
        Returns:
            List of audit log objects
        """
        statement = (
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
    
    def get_by_action(self, action: str, limit: int = 100) -> List[AuditLog]:
        """
        Get audit logs by action
        
        Args:
            action: Action type
            limit: Maximum number of logs to return
        
        Returns:
            List of audit log objects
        """
        statement = (
            select(AuditLog)
            .where(AuditLog.action == action)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
    
    def get_all(self, limit: int = 100) -> List[AuditLog]:
        """
        Get all audit logs
        
        Args:
            limit: Maximum number of logs to return
        
        Returns:
            List of audit log objects
        """
        statement = (
            select(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

