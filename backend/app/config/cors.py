import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_cors_config_from_env() -> dict:
    """
    .env dosyasından CORS yapılandırma bilgilerini okur
    
    Returns:
        CORS yapılandırma bilgilerini içeren dict
    """
    if DOTENV_AVAILABLE:
        # .env dosyasını yükle
        load_dotenv()
    
    env = os.getenv("ENV").lower()
    
    # Production için allowed origins
    allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if allowed_origins_str:
        # Virgülle ayrılmış URL'leri listeye çevir
        allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
    else:
        allowed_origins = []
    
    config = {
        "env": env,
        "allowed_origins": allowed_origins,
    }
    
    return config


def get_cors_origins() -> List[str]:
    """
    CORS allowed origins listesini döndürür
    
    Returns:
        Allowed origins listesi (development'da ["*"], production'da .env'den okunan URL'ler)
    """
    config = get_cors_config_from_env()
    env = config["env"]
    
    if env == "development":
        # Development: Her yerden istek kabul et
        return ["*"]
    else:
        # Production: Sadece belirtilen URL'lerden istek kabul et
        return config["allowed_origins"]


def setup_cors_middleware(app: FastAPI) -> None:
    """
    FastAPI uygulamasına CORS middleware'ini ekler
    
    Args:
        app: FastAPI uygulama instance'ı
    """
    origins = get_cors_origins()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],  # Tüm HTTP metodlarına izin ver
        allow_headers=["*"],  # Tüm header'lara izin ver
    )

