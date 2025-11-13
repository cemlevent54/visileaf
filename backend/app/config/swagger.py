import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_swagger_config_from_env() -> Dict[str, Any]:
    """
    .env dosyasından Swagger yapılandırma bilgilerini okur
    
    Returns:
        Swagger yapılandırma bilgilerini içeren dict
    """
    if DOTENV_AVAILABLE:
        # .env dosyasını yükle
        load_dotenv()
    
    config = {
        "title": os.getenv("SWAGGER_TITLE", "Visileaf API"),
        "description": os.getenv("SWAGGER_DESCRIPTION", "Visileaf API Documentation"),
        "version": os.getenv("SWAGGER_VERSION", "1.0.0"),
        "docs_url": os.getenv("SWAGGER_DOCS_URL", "/docs"),
        "redoc_url": os.getenv("SWAGGER_REDOC_URL", "/redoc"),
        "openapi_url": os.getenv("SWAGGER_OPENAPI_URL", "/openapi.json"),
    }
    
    return config


def get_swagger_title() -> str:
    """
    Swagger başlığını döndürür (.env'den okur, yoksa "Visileaf API")
    
    Returns:
        Swagger başlığı
    """
    config = get_swagger_config_from_env()
    return config["title"]


def get_swagger_description() -> str:
    """
    Swagger açıklamasını döndürür (.env'den okur)
    
    Returns:
        Swagger açıklaması
    """
    config = get_swagger_config_from_env()
    return config["description"]


def get_swagger_version() -> str:
    """
    API versiyonunu döndürür (.env'den okur, yoksa "1.0.0")
    
    Returns:
        API versiyonu
    """
    config = get_swagger_config_from_env()
    return config["version"]


def get_swagger_docs_url() -> str:
    """
    Swagger UI URL'ini döndürür (.env'den okur, yoksa "/docs")
    
    Returns:
        Swagger UI URL'i
    """
    config = get_swagger_config_from_env()
    return config["docs_url"]


def get_swagger_redoc_url() -> str:
    """
    ReDoc URL'ini döndürür (.env'den okur, yoksa "/redoc")
    
    Returns:
        ReDoc URL'i
    """
    config = get_swagger_config_from_env()
    return config["redoc_url"]


def get_swagger_openapi_url() -> str:
    """
    OpenAPI JSON URL'ini döndürür (.env'den okur, yoksa "/openapi.json")
    
    Returns:
        OpenAPI JSON URL'i
    """
    config = get_swagger_config_from_env()
    return config["openapi_url"]


def get_swagger_tags_metadata() -> list:
    """
    Swagger tag'leri için metadata döndürür
    
    Returns:
        Tag metadata listesi
    """
    return [
        {
            "name": "default",
            "description": "Varsayılan endpoint'ler",
        },
        # Buraya daha fazla tag eklenebilir
        # {
        #     "name": "users",
        #     "description": "Kullanıcı işlemleri",
        # },
    ]


def export_openapi_schema(app, output_path: Optional[str] = None) -> str:
    """
    OpenAPI şemasını JSON dosyasına export eder
    
    Args:
        app: FastAPI uygulama instance'ı
        output_path: Çıktı dosya yolu (None ise "openapi.json" kullanılır)
    
    Returns:
        Export edilen dosya yolu
    """
    if output_path is None:
        output_path = "openapi.json"
    
    # OpenAPI şemasını al
    openapi_schema = app.openapi()
    
    # JSON dosyasına kaydet
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    
    return str(output_file.absolute())

