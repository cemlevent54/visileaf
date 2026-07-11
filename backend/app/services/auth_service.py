"""
Auth service - Authentication business logic
"""
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from sqlmodel import Session
from app.repositories.user_repository import UserRepository
from app.repositories.password_reset_token_repository import PasswordResetTokenRepository
from app.models.user import User, UserCreate
from app.utils.password import hash_password, verify_password
from app.utils.jwt import create_access_token, create_refresh_token, verify_token
from app.services.audit_log_service import AuditLogService
from app.services.email_service import EmailService
import secrets
import os
import logging

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for business logic"""
    
    def __init__(
        self,
        session: Session,
        audit_log_service: Optional[AuditLogService] = None,
        email_service: Optional[EmailService] = None
    ):
        """
        Initialize service with database session
        
        Args:
            session: SQLModel database session
            audit_log_service: Optional audit log service
            email_service: Optional email service
        """
        self.session = session
        self.user_repo = UserRepository(session)
        self.token_repo = PasswordResetTokenRepository(session)        
        self.audit_log_service = audit_log_service
        self.email_service = email_service or EmailService()
    
    def register(self, first_name: str, last_name: str, email: str, password: str) -> Dict[str, Any]:
        """
        Register a new user
        
        Args:
            first_name: User's first name
            last_name: User's last name
            email: User's email address
            password: User's password
        
        Returns:
            Dictionary with user data
        
        Raises:
            ValueError: If email already exists
        """
        # Check if user already exists
        existing_user = self.user_repo.get_by_email(email)
        if existing_user:
            raise ValueError("Email already registered")
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Create user data with hashed password
        user_create = UserCreate(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password_hash=hashed_password
        )
        
        # Create user
        user = self.user_repo.create(user_create)
        
        # Log registration
        if self.audit_log_service:
            try:
                self.audit_log_service.log_user_action(
                    action="user.register",
                    user_id=user.id,
                    details={
                        "email": user.email,
                        "first_name": user.first_name,
                        "last_name": user.last_name
                    }
                )
            except Exception:
                pass  # Don't fail if audit log fails
        
        return {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email
            }
        }
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Login user and generate tokens
        
        Args:
            email: User email
            password: User password
        
        Returns:
            Dictionary with user data and tokens
        
        Raises:
            ValueError: If credentials are invalid
        """
        # Get user by email
        user = self.user_repo.get_by_email(email)
        if not user:
            raise ValueError("Invalid email or password")
        
        # Verify password
        if not verify_password(password, user.password_hash):
            raise ValueError("Invalid email or password")
        
        # Generate tokens
        token_data = {
            "user_id": str(user.id),
            "email": user.email
        }
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        # Log successful login
        if self.audit_log_service:
            try:
                self.audit_log_service.log_user_action(
                    action="user.login",
                    user_id=user.id,
                    details={
                        "email": user.email
                    }
                )
            except Exception:
                pass  # Don't fail if audit log fails
        
        return {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email
            },
            "tokens": {
                "access": access_token,
                "refresh": refresh_token
            }
        }
    
    def logout(self, refresh_token: str) -> bool:
        """
        Logout user (invalidate refresh token)
        
        Note: In a stateless JWT system, we typically don't invalidate tokens.
        This is a placeholder for future token blacklist implementation.
        
        Args:
            refresh_token: Refresh token to invalidate
        
        Returns:
            True if successful
        """
        # TODO: Implement token blacklist if needed
        # For now, just verify the token is valid
        payload = verify_token(refresh_token, token_type="refresh")
        
        # Log logout action if token is valid
        if self.audit_log_service and payload:
            try:
                user_id = UUID(payload.get("user_id"))
                self.audit_log_service.log_user_action(
                    action="user.logout",
                    user_id=user_id,
                    details={
                        "refresh_token_jti": payload.get("jti")
                    }
                )
            except Exception:
                pass  # Don't fail if audit log fails
        
        return payload is not None
    
    def get_current_user(self, access_token: str) -> Dict[str, Any]:
        """
        Get current user from access token
        
        Args:
            access_token: JWT access token
        
        Returns:
            Dictionary with user data
        
        Raises:
            ValueError: If token is invalid or user not found
        """
        # Verify token
        payload = verify_token(access_token, token_type="access")
        if not payload:
            raise ValueError("Invalid or expired token")
        
        # Get user
        user_id = UUID(payload.get("user_id"))
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        return {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email
            }
        }
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: JWT refresh token
        
        Returns:
            Dictionary with new access token
        
        Raises:
            ValueError: If refresh token is invalid
        """
        # Verify refresh token
        payload = verify_token(refresh_token, token_type="refresh")
        if not payload:
            raise ValueError("Invalid or expired refresh token")
        
        # Get user
        user_id = UUID(payload.get("user_id"))
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Generate new access token
        token_data = {
            "user_id": str(user.id),
            "email": user.email
        }
        access_token = create_access_token(token_data)
        
        return {
            "access": access_token
        }
    
    def forgot_password(self, email: str, locale: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate password reset token, 6-digit code and send email
        
        Args:
            email: User email
            locale: User locale for email translation (optional, defaults to 'en')
        
        Returns:
            Dictionary with user data
        
        Raises:
            ValueError: If user not found
        """
        # Get user
        user = self.user_repo.get_by_email(email)
        if not user:
            # Don't reveal if user exists or not for security
            # Still log the attempt
            if self.audit_log_service:
                try:
                    self.audit_log_service.log_system_action(
                        action="user.forgot_password",
                        details={"email": email, "user_found": False}
                    )
                except Exception:
                    pass
            return {
                "user": {
                    "id": None,
                    "first_name": None,
                    "last_name": None,
                    "email": email
                }
            }
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        # Hash the token for storage (using password hashing)
        token_hash = hash_password(reset_token)
        
        # Generate 6-digit code
        reset_code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        
        # Create token record with code
        from app.models.password_reset_token import PasswordResetTokenCreate
        token_data = PasswordResetTokenCreate(
            user_id=user.id,
            token_hash=token_hash,
            code=reset_code,  # Store 6-digit code
            expires_at=datetime.utcnow() + timedelta(hours=1),  # 1 hour validity
            used=False
        )
        reset_token_record = self.token_repo.create(token_data)
        
        # Store code in token details (assuming details field exists)
        # If not, we'll need to update the model to include code field
        # For now, we'll store it separately or use a simple mapping
        
        # Determine locale
        if not locale:
            locale = user.language.value if hasattr(user, 'language') and user.language else 'en'
        
        # Get frontend URL from environment
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"
        
        # Prepare template variables
        is_tr = locale == 'tr'
        user_name_display = f' {user.first_name}' if user.first_name else ''
        template_vars = {
            'language': locale,
            'subject': "Şifre Sıfırlama" if is_tr else "Password Reset",
            'greeting': "Merhaba" if is_tr else "Hello",
            'user_name_display': user_name_display,
            'message': "Şifrenizi sıfırlamak için aşağıdaki doğrulama kodunu kullanabilirsiniz:" if is_tr else "You can use the following verification code to reset your password:",
            'code_label': "Doğrulama Kodu" if is_tr else "Verification Code",
            'reset_code': reset_code,
            'reset_link': reset_link,
            'button_text': "Şifreyi Sıfırla" if is_tr else "Reset Password",
            'link_text': "veya aşağıdaki linki kopyalayıp tarayıcınıza yapıştırın:" if is_tr else "or copy and paste the following link in your browser:",
            'expiry_message': "Bu kod 1 saat boyunca geçerlidir." if is_tr else "This code is valid for 1 hour.",
            'footer_message': "Bu e-postayı siz talep etmediyseniz, lütfen görmezden gelin." if is_tr else "If you did not request this email, please ignore it."
        }
        
        # Send email using template
        try:
            html_content = self.email_service._render_template('forgot_password.html', template_vars)
            
            # Plain text version
            text_content = f"""
            {template_vars['subject']}
            
            {template_vars['greeting']}{template_vars['user_name_display']},
            
            {template_vars['message']}
            
            {template_vars['code_label']}: {reset_code}
            
            {template_vars['link_text']}
            {reset_link}
            
            {template_vars['expiry_message']}
            
            {template_vars['footer_message']}
            """
            
            email_result = self.email_service.send_mail_with_resend(
                to=user.email,
                options={
                    'subject': template_vars['subject'],
                    'html': html_content,
                    'text': text_content,
                    'tags': [
                        {'name': 'category', 'value': 'password_reset'}
                    ]
                }
            )
            
            if not email_result.get('success'):
                logger.error('EmailService: Failed to send password reset email', extra={
                    'user_id': str(user.id),
                    'email': user.email,
                    'error': email_result.get('error')
                })
        except Exception as e:
            logger.error('EmailService: Error sending password reset email', extra={
                'user_id': str(user.id),
                'email': user.email,
                'error': str(e)
            }, exc_info=True)
            # Don't fail the request if email fails, but log it
        
        # Log forgot password attempt
        if self.audit_log_service:
            try:
                self.audit_log_service.log_system_action(
                    action="user.forgot_password",
                    details={
                        "email": email,
                        "user_found": True,
                        "user_id": str(user.id),
                        "reset_token_id": str(reset_token_record.id) if reset_token_record else None
                    }
                )
            except Exception:
                pass
        
        # Return user data (don't include token in response)
        return {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email
            }
        }
    
    def reset_password(self, token: str, code: str, new_password: str) -> Dict[str, Any]:
        """
        Reset password using reset token and verification code
        
        Args:
            token: Password reset token (plain/unhashed)
            code: 6-digit verification code
            new_password: New password
        
        Returns:
            Dictionary with user data
        
        Raises:
            ValueError: If token or code is invalid or expired
        """
        # Get valid token using plain token (bcrypt verification)
        reset_token = self.token_repo.get_valid_token_by_plain_token(token)
        if not reset_token:
            raise ValueError("Invalid or expired reset token")
        
        # Verify code
        if reset_token.code != code:
            raise ValueError("Invalid verification code")
        
        # Get user
        user = self.user_repo.get_by_id(reset_token.user_id)
        if not user:
            raise ValueError("User not found")
        
        # Update password
        hashed_password = hash_password(new_password)
        from app.models.user import UserUpdate
        user_update = UserUpdate(password_hash=hashed_password)
        self.user_repo.update(user.id, user_update)
        
        # Mark token as used
        self.token_repo.mark_as_used(reset_token.id)
        
        # Log password reset
        if self.audit_log_service:
            try:
                self.audit_log_service.log_user_action(
                    action="user.reset_password",
                    user_id=user.id,
                    details={
                        "email": user.email,
                        "reset_token_id": str(reset_token.id)
                    }
                )
            except Exception:
                pass
        
        return {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email
            }
        }

