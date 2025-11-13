import json
import os
from pathlib import Path
from typing import Optional
from fastapi import Request

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_i18n_config_from_env() -> dict:
    """
    .env dosyasından i18n yapılandırma bilgilerini okur
    
    Returns:
        i18n yapılandırma bilgilerini içeren dict
    """
    if DOTENV_AVAILABLE:
        # .env dosyasını yükle
        load_dotenv()
    
    config = {
        "default_locale": os.getenv("DEFAULT_LOCALE", "tr"),
        "locales_dir": os.getenv("LOCALES_DIR", "app/locales"),
    }
    
    return config


def get_default_locale() -> str:
    """
    Varsayılan dili döndürür (.env'den okur, yoksa "tr")
    
    Returns:
        Varsayılan dil kodu (örn: "tr", "en")
    """
    config = get_i18n_config_from_env()
    return config["default_locale"]


def get_locales_dir() -> str:
    """
    Locale dosyalarının bulunduğu dizini döndürür (.env'den okur, yoksa "app/locales")
    
    Returns:
        Locale dosyalarının dizin yolu
    """
    config = get_i18n_config_from_env()
    return config["locales_dir"]


# Çeviri cache'i
_translations_cache: dict = {}


def _load_translations(locale: str) -> dict:
    """
    Belirtilen locale için çeviri dosyasını yükler
    
    Args:
        locale: Dil kodu (örn: "tr", "en")
    
    Returns:
        Çeviri sözlüğü
    """
    if locale in _translations_cache:
        return _translations_cache[locale]
    
    locales_dir = get_locales_dir()
    file_path = Path(locales_dir) / f"{locale}.json"
    
    try:
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                translations = json.load(f)
                _translations_cache[locale] = translations
                return translations
    except Exception:
        pass
    
    return {}


def gettext(key: str, request: Optional[Request] = None, locale: Optional[str] = None) -> str:
    """
    Çeviri anahtarını kullanarak çevrilmiş metni döndürür
    
    Args:
        key: Çeviri anahtarı
        request: FastAPI Request objesi (opsiyonel)
        locale: Dil kodu (opsiyonel, request'ten veya varsayılan değerden alınır)
    
    Returns:
        Çevrilmiş metin veya anahtar (çeviri bulunamazsa)
    """
    # Locale belirleme: önce parametre, sonra request, son olarak varsayılan
    if locale is None:
        if request is not None:
            locale = getattr(request.state, "locale", None)
        if locale is None:
            locale = get_default_locale()
    
    # Çevirileri yükle
    translations = _load_translations(locale)
    
    # Çeviriyi döndür veya anahtarı
    return translations.get(key, key)
