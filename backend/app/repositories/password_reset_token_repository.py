"""
Password reset token repository - Token CRUD operations
"""
from typing import Optional
from uuid import UUID
from datetime import datetime
from sqlmodel import Session, select
from app.models.password_reset_token import PasswordResetToken, PasswordResetTokenCreate
from app.utils.password import verify_password


class PasswordResetTokenRepository:
    """Password reset token repository for database operations"""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session
        
        Args:
            session: SQLModel database session
        """
        self.session = session
    
    def create(self, token_data: PasswordResetTokenCreate) -> PasswordResetToken:
        """
        Create a new password reset token
        
        Args:
            token_data: Token creation data
        
        Returns:
            Created token object
        """
        token = PasswordResetToken(**token_data.model_dump())
        self.session.add(token)
        self.session.commit()
        self.session.refresh(token)
        return token
    
    def get_by_token_hash(self, token_hash: str) -> Optional[PasswordResetToken]:
        """
        Get token by hash
        
        Args:
            token_hash: Hashed token string
        
        Returns:
            Token object or None if not found
        """
        statement = select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash
        )
        return self.session.exec(statement).first()
    
    def get_valid_token_by_hash(self, token_hash: str) -> Optional[PasswordResetToken]:
        """
        Get valid (not used and not expired) token by hash
        
        Note: This method uses direct hash comparison, which works only if
        the hash was created with the same salt. For bcrypt hashes, use
        get_valid_token_by_plain_token instead.
        
        Args:
            token_hash: Hashed token string
        
        Returns:
            Valid token object or None
        """
        token = self.get_by_token_hash(token_hash)
        if not token:
            return None
        
        # Check if token is used
        if token.used:
            return None
        
        # Check if token is expired
        if token.expires_at < datetime.utcnow():
            return None
        
        return token
    
    def get_valid_token_by_plain_token(self, plain_token: str) -> Optional[PasswordResetToken]:
        """
        Get valid (not used and not expired) token by plain token string
        using bcrypt verification
        
        Args:
            plain_token: Plain (unhashed) token string
        
        Returns:
            Valid token object or None
        """
        # Get all non-expired and unused tokens
        statement = select(PasswordResetToken).where(
            PasswordResetToken.used == False,
            PasswordResetToken.expires_at > datetime.utcnow()
        )
        tokens = list(self.session.exec(statement).all())
        
        # Verify each token's hash against the plain token
        for token in tokens:
            try:
                if verify_password(plain_token, token.token_hash):
                    return token
            except Exception:
                # Skip tokens that fail verification
                continue
        
        return None
    
    def mark_as_used(self, token_id: UUID) -> bool:
        """
        Mark token as used
        
        Args:
            token_id: Token UUID
        
        Returns:
            True if marked, False if not found
        """
        statement = select(PasswordResetToken).where(PasswordResetToken.id == token_id)
        token = self.session.exec(statement).first()
        if not token:
            return False
        
        token.used = True
        self.session.add(token)
        self.session.commit()
        return True
    
    def get_by_user_id(self, user_id: UUID) -> list[PasswordResetToken]:
        """
        Get all tokens for a user
        
        Args:
            user_id: User UUID
        
        Returns:
            List of token objects
        """
        statement = select(PasswordResetToken).where(
            PasswordResetToken.user_id == user_id
        )
        return list(self.session.exec(statement).all())

