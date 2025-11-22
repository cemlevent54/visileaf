# Alembic Migration Kılavuzu

Bu proje **SQLModel** ve **Alembic** kullanarak veritabanı migration'larını yönetir.

## Kurulum

1. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Migration Komutları

### İlk Migration'ı Oluşturma

Yeni bir migration oluşturmak için:

```bash
# Backend klasörüne gidin
cd backend

# Yeni migration oluştur
alembic revision --autogenerate -m "migration açıklaması"

# Örnek:
alembic revision --autogenerate -m "create_example_table"
```

### Migration'ları Uygulama

Migration'ları veritabanına uygulamak için:

```bash
# Son migration'a kadar tüm migration'ları uygula
alembic upgrade head

# Belirli bir revision'a kadar uygula
alembic upgrade <revision_id>

# Bir adım ileri git
alembic upgrade +1
```

### Migration'ları Geri Alma

```bash
# Son migration'ı geri al
alembic downgrade -1

# Belirli bir revision'a geri dön
alembic downgrade <revision_id>

# Tüm migration'ları geri al
alembic downgrade base
```

### Migration Durumunu Kontrol Etme

```bash
# Mevcut migration durumunu göster
alembic current

# Migration geçmişini göster
alembic history

# Belirli bir revision'ın detaylarını göster
alembic history <revision_id>
```

## Model Oluşturma

1. `backend/app/models/` klasörüne yeni bir model dosyası oluşturun:

```python
# backend/app/models/user.py
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

2. Modeli `backend/app/models/__init__.py` dosyasına ekleyin:

```python
from .user import User

__all__ = ["User"]
```

3. Migration oluşturun:

```bash
alembic revision --autogenerate -m "create_user_table"
```

4. Migration'ı uygulayın:

```bash
alembic upgrade head
```

## Önemli Notlar

- **SQLModel** kullanıldığı için modeller `table=True` parametresi ile tanımlanmalıdır
- Migration oluşturmadan önce tüm modellerin `app/models/__init__.py` dosyasına import edildiğinden emin olun
- `.env` dosyasında veritabanı bağlantı bilgilerinin doğru olduğundan emin olun:
  ```
  DB_HOST=localhost
  DB_PORT=5432
  DB_NAME=your_database_name
  DB_USER=postgres
  DB_PASSWORD=your_password
  ```

## Sorun Giderme

### Migration'lar modelleri algılamıyor

`backend/alembic/env.py` dosyasında modellerin import edildiğinden emin olun:

```python
from app.models import *  # Tüm modelleri import et
```

### Veritabanı bağlantı hatası

`.env` dosyasındaki veritabanı bilgilerini kontrol edin ve PostgreSQL'in çalıştığından emin olun.

