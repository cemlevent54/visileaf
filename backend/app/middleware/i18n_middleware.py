from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi import FastAPI
from app.config.i18n import get_default_locale


class I18nMiddleware(BaseHTTPMiddleware):
    """
    Accept-Language header'ını okuyup request.state.locale'i ayarlayan middleware
    """
    
    def __init__(self, app, default_locale: str = "tr"):
        super().__init__(app)
        self.default_locale = default_locale
    
    async def dispatch(self, request: Request, call_next):
        # Accept-Language header'ını oku
        accept_language = request.headers.get("Accept-Language", "")
        
        # Locale belirleme
        locale = self._parse_locale(accept_language)
        
        # Request state'e locale ekle
        request.state.locale = locale
        
        # İsteği işle
        response = await call_next(request)
        return response
    
    def _parse_locale(self, accept_language: str) -> str:
        """
        Accept-Language header'ından locale'i parse eder
        
        Args:
            accept_language: Accept-Language header değeri
        
        Returns:
            Locale kodu (örn: "tr", "en")
        """
        if not accept_language:
            return self.default_locale
        
        # Accept-Language formatı: "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"
        # İlk dil kodunu al
        languages = accept_language.split(",")
        if languages:
            # İlk dil kodunu al ve quality değerini temizle
            first_lang = languages[0].split(";")[0].strip()
            # Locale kodunu al (örn: "tr-TR" -> "tr")
            locale = first_lang.split("-")[0].lower()
            
            # Desteklenen dilleri kontrol et
            supported_locales = ["tr", "en"]
            if locale in supported_locales:
                return locale
        
        return self.default_locale


def setup_i18n_middleware(app: FastAPI) -> None:
    """
    FastAPI uygulamasına i18n middleware'ini ekler
    
    Args:
        app: FastAPI uygulama instance'ı
    """
    default_locale = get_default_locale()
    
    app.add_middleware(
        I18nMiddleware,
        default_locale=default_locale
    )

