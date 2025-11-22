"""
User repository - User model CRUD operations
"""
from typing import Optional
from uuid import UUID
from sqlmodel import Session, select
from app.models.user import User, UserCreate, UserUpdate


class UserRepository:
    """User repository for database operations"""
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session
        
        Args:
            session: SQLModel database session
        """
        self.session = session
    
    def create(self, user_data: UserCreate) -> User:
        """
        Create a new user
        
        Args:
            user_data: User creation data
        
        Returns:
            Created user object
        """
        user = User(**user_data.model_dump())
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user
    
    def get_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            user_id: User UUID
        
        Returns:
            User object or None if not found
        """
        statement = select(User).where(
            User.id == user_id,
            User.deleted_at.is_(None)
        )
        return self.session.exec(statement).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email
        
        Args:
            email: User email address
        
        Returns:
            User object or None if not found
        """
        statement = select(User).where(
            User.email == email,
            User.deleted_at.is_(None)
        )
        return self.session.exec(statement).first()
    
    def update(self, user_id: UUID, user_data: UserUpdate) -> Optional[User]:
        """
        Update user
        
        Args:
            user_id: User UUID
            user_data: User update data
        
        Returns:
            Updated user object or None if not found
        """
        user = self.get_by_id(user_id)
        if not user:
            return None
        
        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user
    
    def delete(self, user_id: UUID) -> bool:
        """
        Soft delete user (sets deleted_at timestamp)
        
        Args:
            user_id: User UUID
        
        Returns:
            True if deleted, False if not found
        """
        user = self.get_by_id(user_id)
        if not user:
            return False
        
        from datetime import datetime
        user.deleted_at = datetime.utcnow()
        self.session.add(user)
        self.session.commit()
        return True

