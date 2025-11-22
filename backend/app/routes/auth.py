"""
Auth routes - Authentication endpoints
"""
from fastapi import APIRouter, Depends, Header, Request
from sqlmodel import Session
from app.config import get_session
from app.controllers.auth_controller import AuthController
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

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    request: RegisterRequest,
    http_request: Request,
    session: Session = Depends(get_session)
):
    """
    Register a new user
    
    - **first_name**: User's first name
    - **last_name**: User's last name
    - **email**: User's email address (must be unique)
    - **password**: User's password (min 8 characters)
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.register(request)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    session: Session = Depends(get_session)
):
    """
    Login user and get access/refresh tokens
    
    - **email**: User's email address
    - **password**: User's password
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.login(request)


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: LogoutRequest,
    http_request: Request,
    session: Session = Depends(get_session)
):
    """
    Logout user (invalidate refresh token)
    
    - **refresh_token**: Refresh token to invalidate
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.logout(request)


@router.get("/me", response_model=MeResponse)
async def get_me(
    http_request: Request,
    authorization: str = Header(None, alias="Authorization"),
    session: Session = Depends(get_session)
):
    """
    Get current user information
    
    Requires Bearer token in Authorization header
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.get_me(authorization)


@router.post("/refresh-token", response_model=RefreshTokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    http_request: Request,
    session: Session = Depends(get_session)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.refresh_token(request)


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    http_request: Request,
    session: Session = Depends(get_session)
):
    """
    Request password reset token
    
    - **email**: User's email address
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.forgot_password(request)


@router.post("/reset-password", response_model=ResetPasswordResponse)
async def reset_password(
    request: ResetPasswordRequest,
    http_request: Request,
    session: Session = Depends(get_session)
):
    """
    Reset password using reset token
    
    - **token**: Password reset token
    - **password**: New password (min 8 characters)
    """
    audit_log_service = AuditLogService(session)
    auth_service = AuthService(session, audit_log_service)
    controller = AuthController(auth_service, audit_log_service, http_request)
    return controller.reset_password(request)

