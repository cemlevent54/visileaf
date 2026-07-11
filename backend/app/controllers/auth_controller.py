"""
Auth controller - HTTP request/response handling for authentication
"""
from typing import Dict, Any
from fastapi import HTTPException, Depends, Header, Request
from sqlmodel import Session
from app.config import get_session, gettext
from app.services.auth_service import AuthService
from app.services.audit_log_service import AuditLogService
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    LogoutRequest,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    RegisterResponse,
    LoginResponse,
    LogoutResponse,
    MeResponse,
    RefreshTokenResponse,
    ForgotPasswordResponse,
    ResetPasswordResponse
)


class AuthController:
    """Authentication controller for HTTP handling"""
    
    def __init__(
        self,
        auth_service: AuthService,
        audit_log_service: AuditLogService,
        request: Request = None
    ):
        """
        Initialize controller with services
        
        Args:
            auth_service: AuthService instance
            audit_log_service: AuditLogService instance
            request: FastAPI Request object for i18n
        """
        self.auth_service = auth_service
        self.audit_log_service = audit_log_service
        self.request = request
    
    def register(self, request: RegisterRequest) -> RegisterResponse:
        """
        Handle user registration
        
        Args:
            request: Registration request data
        
        Returns:
            Registration response
        
        Raises:
            HTTPException: If registration fails
        """
        try:
            result = self.auth_service.register(
                first_name=request.first_name,
                last_name=request.last_name,
                email=request.email,
                password=request.password
            )
            
            return RegisterResponse(
                success=True,
                message=gettext("auth_controller.register_success", self.request),
                data=result
            )
        except ValueError as e:
            error_msg = str(e)
            # Translate common error messages
            if "already registered" in error_msg.lower() or "email" in error_msg.lower():
                error_msg = gettext("auth_controller.email_already_registered", self.request)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=gettext("auth_controller.registration_failed", self.request))
    
    def login(self, request: LoginRequest) -> LoginResponse:
        """
        Handle user login
        
        Args:
            request: Login request data
        
        Returns:
            Login response with tokens
        
        Raises:
            HTTPException: If login fails
        """
        try:
            result = self.auth_service.login(request.email, request.password)
            
            return LoginResponse(
                success=True,
                data=result,
                message=gettext("auth_controller.login_success", self.request)
            )
        except ValueError as e:
            error_msg = str(e)
            # Translate common error messages
            if "invalid" in error_msg.lower() or "password" in error_msg.lower():
                error_msg = gettext("auth_controller.invalid_credentials", self.request)
            raise HTTPException(status_code=401, detail=error_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=gettext("auth_controller.login_failed", self.request))
    
    def logout(self, request: LogoutRequest) -> LogoutResponse:
        """
        Handle user logout
        
        Args:
            request: Logout request data
        
        Returns:
            Logout response
        
        Raises:
            HTTPException: If logout fails
        """
        try:
            self.auth_service.logout(request.refresh_token)
            
            return LogoutResponse(
                success=True,
                message=gettext("auth_controller.logout_success", self.request)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=gettext("auth_controller.logout_failed", self.request))
    
    def get_me(self, authorization: str) -> MeResponse:
        """
        Get current user information
        
        Args:
            authorization: Authorization header with Bearer token
        
        Returns:
            User information response
        
        Raises:
            HTTPException: If token is invalid
        """
        try:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(
                    status_code=401, 
                    detail=gettext("auth_controller.authorization_header_required", self.request)
                )
            
            access_token = authorization.replace("Bearer ", "")
            result = self.auth_service.get_current_user(access_token)
            
            return MeResponse(
                success=True,
                data=result,
                message=gettext("auth_controller.me_success", self.request)
            )
        except ValueError as e:
            error_msg = str(e)
            # Translate common error messages
            if "invalid" in error_msg.lower() or "expired" in error_msg.lower() or "token" in error_msg.lower():
                error_msg = gettext("auth_controller.invalid_token", self.request)
            elif "not found" in error_msg.lower():
                error_msg = gettext("auth_controller.user_not_found", self.request)
            raise HTTPException(status_code=401, detail=error_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=gettext("auth_controller.get_user_failed", self.request))
    
    def refresh_token(self, request: RefreshTokenRequest) -> RefreshTokenResponse:
        """
        Refresh access token
        
        Args:
            request: Refresh token request data
        
        Returns:
            New access token response
        
        Raises:
            HTTPException: If refresh fails
        """
        try:
            result = self.auth_service.refresh_access_token(request.refresh_token)
            
            return RefreshTokenResponse(
                success=True,
                message=gettext("auth_controller.refresh_token_success", self.request),
                data=result
            )
        except ValueError as e:
            error_msg = str(e)
            # Translate common error messages
            if "invalid" in error_msg.lower() or "expired" in error_msg.lower() or "refresh" in error_msg.lower():
                error_msg = gettext("auth_controller.invalid_refresh_token", self.request)
            elif "not found" in error_msg.lower():
                error_msg = gettext("auth_controller.user_not_found", self.request)
            raise HTTPException(status_code=401, detail=error_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=gettext("auth_controller.refresh_token_failed", self.request))
    
    def forgot_password(self, request: ForgotPasswordRequest) -> ForgotPasswordResponse:
        """
        Handle forgot password request
        
        Args:
            request: Forgot password request data
        
        Returns:
            Forgot password response
        
        Raises:
            HTTPException: If request fails
        """
        try:
            result = self.auth_service.forgot_password(request.email)
            
            return ForgotPasswordResponse(
                success=True,
                data=result,
                message=gettext("auth_controller.forgot_password_success", self.request)
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=gettext("auth_controller.forgot_password_failed", self.request)
            )
    
    def reset_password(self, request: ResetPasswordRequest) -> ResetPasswordResponse:
        """
        Handle password reset
        
        Args:
            request: Reset password request data
        
        Returns:
            Reset password response
        
        Raises:
            HTTPException: If reset fails
        """
        try:
            result = self.auth_service.reset_password(request.token, request.code, request.password)
            
            return ResetPasswordResponse(
                success=True,
                data=result,
                message=gettext("auth_controller.reset_password_success", self.request)
            )
        except ValueError as e:
            error_msg = str(e)
            # Translate common error messages
            if "invalid verification code" in error_msg.lower() or "code" in error_msg.lower():
                error_msg = gettext("auth_controller.invalid_verification_code", self.request)
            elif "invalid" in error_msg.lower() or "expired" in error_msg.lower() or "reset" in error_msg.lower():
                error_msg = gettext("auth_controller.invalid_reset_token", self.request)
            elif "not found" in error_msg.lower():
                error_msg = gettext("auth_controller.user_not_found", self.request)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=gettext("auth_controller.reset_password_failed", self.request))

