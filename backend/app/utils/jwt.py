"""
JWT Token utility functions
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import uuid4
import jwt
import os

# JWT Secret Key (environment variable'dan al)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Access token oluşturur
    
    Args:
        data: Token payload'ında olacak veriler (user_id, email vb.)
        expires_delta: Token süresi (None ise default kullanılır)
    
    Returns:
        JWT access token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid4()),
        "token_type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Refresh token oluşturur
    
    Args:
        data: Token payload'ında olacak veriler (user_id, email vb.)
    
    Returns:
        JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid4()),
        "token_type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    JWT token'ı doğrular ve payload'ı döndürür
    
    Args:
        token: Doğrulanacak JWT token
        token_type: Token tipi ("access" veya "refresh")
    
    Returns:
        Token payload dict veya None (geçersizse)
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Token type kontrolü
        if payload.get("token_type") != token_type:
            return None
        
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    JWT token'ı decode eder (doğrulama yapmadan)
    
    Args:
        token: Decode edilecek JWT token
    
    Returns:
        Token payload dict veya None
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except jwt.InvalidTokenError:
        return None

