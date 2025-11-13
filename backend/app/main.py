from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.config import (
    setUpLogging,
    get_logger,
    reconfigure_uvicorn_loggers,
    init_db,
    close_db,
    get_engine,
    gettext,
    get_swagger_title,
    get_swagger_description,
    get_swagger_version,
    get_swagger_docs_url,
    get_swagger_redoc_url,
    get_swagger_openapi_url,
    get_swagger_tags_metadata,
    export_openapi_schema,
)
from app.middleware import setup_i18n_middleware
from app.config import setup_cors_middleware

# Logging yapılandırmasını başlat (LOG_LEVEL .env dosyasından okunur)
setUpLogging(use_colorlog=True)

# Logger'ı al
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - startup ve shutdown işlemlerini yönetir
    """
    # Startup
    # Uvicorn başladıktan sonra logger'ları yeniden yapılandır
    reconfigure_uvicorn_loggers()
    
    # Veritabanı bağlantısını test et
    try:
        engine = get_engine()
        # Bağlantıyı test et
        with engine.connect() as conn:
            logger.info(gettext("database_connection_success"))
        
        # Veritabanı tablolarını oluştur (gerekirse)
        # init_db()  # SQLModel modelleri import edildikten sonra çağrılmalı
    except Exception as e:
        logger.error(f"{gettext('database_connection_error')}: {e}")
    
    # OpenAPI şemasını export et
    try:
        output_path = export_openapi_schema(app, "openapi.json")
        logger.info(f"OpenAPI şeması export edildi: {output_path}")
    except Exception as e:
        logger.warning(f"OpenAPI şeması export edilemedi: {e}")
    
    logger.info(gettext("application_starting"))
    
    yield
    
    # Shutdown
    # Veritabanı bağlantısını kapat
    close_db()
    logger.info(gettext("application_shutting_down"))


# Swagger ayarları (.env'den okunur)
app = FastAPI(
    title=get_swagger_title(),
    description=get_swagger_description(),
    version=get_swagger_version(),
    docs_url=get_swagger_docs_url(),
    redoc_url=get_swagger_redoc_url(),
    openapi_url=get_swagger_openapi_url(),
    tags_metadata=get_swagger_tags_metadata(),
    lifespan=lifespan,
)

# CORS Middleware ekle (ayarlar config/cors.py'de)
setup_cors_middleware(app)

# i18n Middleware ekle (ayarlar config/i18n.py'de)
setup_i18n_middleware(app)

@app.get("/")
def read_root(request: Request):
    logger.info(gettext("home_page_request", request))
    return {"message": gettext("hello_world", request)}


# Uvicorn'u programatik olarak çalıştırmak için
if __name__ == "__main__":
    import uvicorn
    from app.config import get_port, get_host
    
    port = get_port()
    host = get_host()
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )

