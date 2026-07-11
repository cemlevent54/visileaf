"""
Auth request/response schemas
"""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID
from datetime import datetime


# Request Schemas
class RegisterRequest(BaseModel):
    """Register request schema"""
    first_name: str = Field(..., min_length=1, max_length=255)
    last_name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=100)


class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str


class LogoutRequest(BaseModel):
    """Logout request schema"""
    refresh_token: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema"""
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    """Forgot password request schema"""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Reset password request schema"""
    token: str = Field(..., min_length=1)
    code: str = Field(..., min_length=6, max_length=6, description="6-digit verification code")
    password: str = Field(..., min_length=8, max_length=100)


# Response Schemas
class UserData(BaseModel):
    """User data schema for responses"""
    id: UUID
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokensData(BaseModel):
    """Tokens data schema"""
    access: str
    refresh: str


class RegisterResponse(BaseModel):
    """Register response schema"""
    success: bool
    message: str
    data: dict


class LoginResponse(BaseModel):
    """Login response schema"""
    success: bool
    data: dict
    message: str


class LogoutResponse(BaseModel):
    """Logout response schema"""
    success: bool
    message: str


class MeResponse(BaseModel):
    """Me endpoint response schema"""
    success: bool
    data: dict
    message: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response schema"""
    success: bool
    message: str
    data: Optional[dict] = None


class ForgotPasswordResponse(BaseModel):
    """Forgot password response schema"""
    success: bool
    data: dict
    message: str


class ResetPasswordResponse(BaseModel):
    """Reset password response schema"""
    success: bool
    data: dict
    message: str

