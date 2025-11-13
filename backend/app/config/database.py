import os
from typing import Optional
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_db_config_from_env() -> dict:
    """
    .env dosyasından veritabanı yapılandırma bilgilerini okur
    
    Returns:
        Veritabanı yapılandırma bilgilerini içeren dict
    """
    if DOTENV_AVAILABLE:
        # .env dosyasını yükle
        load_dotenv()
    
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "name": os.getenv("DB_NAME", "dbname"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }
    
    return config


def get_database_url() -> str:
    """
    PostgreSQL veritabanı bağlantı URL'sini oluşturur
    
    Returns:
        PostgreSQL connection URL
    """
    config = get_db_config_from_env()
    
    # PostgreSQL connection URL formatı:
    # postgresql+psycopg://user:password@host:port/dbname (psycopg3 için)
    # postgresql://user:password@host:port/dbname (psycopg2 için)
    database_url = (
        f"postgresql+psycopg://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['name']}"
    )
    
    return database_url


# Global engine değişkeni
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """
    Veritabanı engine'ini döndürür (singleton pattern)
    
    Returns:
        SQLModel Engine instance
    """
    global _engine
    
    if _engine is None:
        database_url = get_database_url()
        _engine = create_engine(
            database_url,
            echo=False,  # SQL sorgularını loglamak için True yapılabilir
            pool_pre_ping=True,  # Bağlantı sağlığını kontrol et
            pool_size=5,  # Connection pool boyutu
            max_overflow=10,  # Maksimum ekstra bağlantı sayısı
        )
    
    return _engine


def init_db() -> None:
    """
    Veritabanı tablolarını oluşturur
    SQLModel modelleri import edildikten sonra çağrılmalıdır
    """
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """
    Veritabanı session'ı döndürür (dependency injection için)
    
    Yields:
        SQLModel Session instance
    """
    with Session(get_engine()) as session:
        yield session


def close_db() -> None:
    """
    Veritabanı bağlantısını kapatır
    """
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None

