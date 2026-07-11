"""
Audit log service - Audit log business logic
"""
from typing import Optional, Dict, Any
from uuid import UUID
from sqlmodel import Session
from app.repositories.audit_log_repository import AuditLogRepository
from app.models.audit_log import AuditLogCreate


class AuditLogService:
    """Audit log service for business logic"""
    
    def __init__(self, session: Session):
        """
        Initialize service with database session
        
        Args:
            session: SQLModel database session
        """
        self.session = session
        self.repository = AuditLogRepository(session)
    
    def log_action(
        self,
        action: str,
        user_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a system action
        
        Args:
            action: Action type (e.g., "user.register", "user.login", "user.logout")
            user_id: User UUID (if action is performed by a user)
            details: Additional details about the action (optional)
        """
        log_data = AuditLogCreate(
            action=action,
            user_id=user_id,
            details=details or {}
        )
        self.repository.create(log_data)
    
    def log_user_action(
        self,
        action: str,
        user_id: UUID,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a user action (shorthand for log_action with user_id)
        
        Args:
            action: Action type
            user_id: User UUID
            details: Additional details about the action (optional)
        """
        self.log_action(action=action, user_id=user_id, details=details)
    
    def log_system_action(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a system action (shorthand for log_action without user_id)
        
        Args:
            action: Action type
            details: Additional details about the action (optional)
        """
        self.log_action(action=action, user_id=None, details=details)
    
    # Convenience methods for common auth actions
    def log_register(self, user_id: UUID, email: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log user registration"""
        log_details = {"email": email}
        if details:
            log_details.update(details)
        self.log_user_action("user.register", user_id, log_details)
    
    def log_login(self, user_id: UUID, email: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log user login"""
        log_details = {"email": email}
        if details:
            log_details.update(details)
        self.log_user_action("user.login", user_id, log_details)
    
    def log_logout(self, user_id: Optional[UUID] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Log user logout"""
        if user_id:
            self.log_user_action("user.logout", user_id, details)
        else:
            self.log_system_action("user.logout", details)
    
    def log_get_me(self, user_id: UUID, details: Optional[Dict[str, Any]] = None) -> None:
        """Log get user info request"""
        self.log_user_action("user.get_me", user_id, details)
    
    def log_refresh_token(self, user_id: UUID, details: Optional[Dict[str, Any]] = None) -> None:
        """Log token refresh"""
        self.log_user_action("user.refresh_token", user_id, details)
    
    def log_forgot_password(self, email: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log forgot password request"""
        log_details = {"email": email}
        if details:
            log_details.update(details)
        self.log_system_action("user.forgot_password", log_details)
    
    def log_reset_password(self, user_id: UUID, details: Optional[Dict[str, Any]] = None) -> None:
        """Log password reset"""
        self.log_user_action("user.reset_password", user_id, details)

