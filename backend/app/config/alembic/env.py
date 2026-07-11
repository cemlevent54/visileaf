"""
Alembic migration environment yapılandırması
SQLModel modellerini otomatik olarak algılar ve migration oluşturur
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
from pathlib import Path

# Backend klasörünü Python path'ine ekle
# env.py -> app/config/alembic/env.py
# backend_dir -> backend klasörü (3 seviye yukarı)
backend_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

# SQLModel ve database yapılandırmasını import et
from app.config.database import get_database_url, get_engine
from sqlmodel import SQLModel

# Tüm modelleri import et (migration'lar için gerekli)
# Modelleri buraya ekleyin ki Alembic onları algılayabilsin
# Örnek: from app.models.user import User
# Örnek: from app.models.example import Example

# Tüm modelleri otomatik import etmek için:
try:
    from app.models import *  # Tüm modelleri import et
except ImportError:
    # Modeller henüz oluşturulmamışsa hata verme
    pass

# Alembic Config object
config = context.config

# Logging yapılandırmasını yükle
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Migration dosyalarının oluşturulacağı klasörü ayarla
# app/migrations klasörüne migration dosyaları oluşturulacak
migrations_dir = backend_dir / "app" / "migrations"
migrations_dir.mkdir(exist_ok=True)

# SQLModel metadata'sını Alembic'e bildir
target_metadata = SQLModel.metadata

# Database URL'i env.py'den al
def get_url():
    """
    Veritabanı URL'ini app.config.database'den alır
    """
    return get_database_url()


def run_migrations_offline() -> None:
    """
    Offline mode'da migration çalıştırır (SQL dosyası oluşturur)
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Online mode'da migration çalıştırır (doğrudan veritabanına uygular)
    """
    # Engine'i mevcut yapılandırmadan al veya yeni oluştur
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    try:
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection, 
                target_metadata=target_metadata
            )

            with context.begin_transaction():
                context.run_migrations()
    except Exception as e:
        error_msg = str(e).lower()
        # Revision/migration hataları Alembic'in kendi mesajlarına bırakılmalı
        if "revision" in error_msg or "migration" in error_msg or "didn't produce" in error_msg:
            # Bu hatalar Alembic tarafından zaten gösteriliyor, tekrar gösterme
            raise
        # Veritabanı bağlantı hatası durumunda daha açıklayıcı hata mesajı
        elif "connection" in error_msg or "connect" in error_msg:
            print(f"Hata: Veritabanı bağlantısı kurulamadı: {e}")
            print("Lütfen .env dosyasındaki veritabanı ayarlarını kontrol edin.")
            raise
        else:
            # Diğer hatalar
            raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

