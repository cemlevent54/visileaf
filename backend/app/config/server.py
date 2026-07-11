import os
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_server_config_from_env() -> dict:
    """
    .env dosyasından server yapılandırma bilgilerini okur
    
    Returns:
        Server yapılandırma bilgilerini içeren dict
    """
    if DOTENV_AVAILABLE:
        # .env dosyasını yükle
        load_dotenv()
    
    config = {
        "port": int(os.getenv("PORT", "8000")),
        "host": os.getenv("HOST", "127.0.0.1"),
    }
    
    return config


def get_port() -> int:
    """
    Server port'unu döndürür (.env'den okur, yoksa 8000)
    
    Returns:
        Port numarası
    """
    config = get_server_config_from_env()
    return config["port"]


def get_host() -> str:
    """
    Server host'unu döndürür (.env'den okur, yoksa "127.0.0.1")
    
    Returns:
        Host adresi
    """
    config = get_server_config_from_env()
    return config["host"]

