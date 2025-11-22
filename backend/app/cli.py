"""
CLI komutları - npm start ve npm run db:migrate benzeri komutlar
"""
import sys
import subprocess
from pathlib import Path


def _get_backend_dir():
    """Backend klasörünün yolunu döndürür"""
    return Path(__file__).parent.parent


def _run_alembic_command(command: list[str]):
    """Alembic komutunu çalıştırır"""
    import os
    
    backend_dir = _get_backend_dir()
    alembic_cmd = ["alembic"] + command
    
    # Python path'ini ayarla (Alembic'in app modüllerini bulabilmesi için)
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    if python_path:
        env["PYTHONPATH"] = f"{str(backend_dir)}{os.pathsep}{python_path}"
    else:
        env["PYTHONPATH"] = str(backend_dir)
    
    # Çıktıyı gerçek zamanlı göster ve hataları yakala
    try:
        result = subprocess.run(
            alembic_cmd,
            cwd=str(backend_dir),
            check=False,
            env=env,
            stdout=None,  # stdout'u doğrudan terminale yazdır
            stderr=subprocess.STDOUT  # stderr'ı da stdout'a yönlendir
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından iptal edildi.")
        sys.exit(130)
    except Exception as e:
        print(f"Hata: Alembic komutu çalıştırılamadı: {e}")
        sys.exit(1)


def start():
    """Projeyi başlatır (npm start benzeri)"""
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


def dev():
    """Development modunda projeyi başlatır"""
    start()  # Şimdilik aynı, ileride farklılaştırılabilir


def migrate():
    """Migration'ları uygular (npm run db:migrate benzeri)"""
    _run_alembic_command(["upgrade", "head"])


def migrate_create(message: str = None):
    """Yeni migration oluşturur"""
    if not message:
        print("Hata: Migration mesajı gerekli")
        print("Kullanım: python -m app migrate create 'migration mesajı'")
        sys.exit(1)
    
    _run_alembic_command(["revision", "--autogenerate", "-m", message])


def migrate_upgrade(revision: str = "head"):
    """Migration'ları uygular
    
    Args:
        revision: Uygulanacak migration revision
                 - "head": Tüm migration'ları uygula (varsayılan)
                 - "+1": Bir sonraki migration'ı uygula (tek tek)
                 - "+2": İki sonraki migration'ı uygula
                 - Belirli ID: O migration'a kadar uygula (örn: "abc123")
    """
    _run_alembic_command(["upgrade", revision])


def migrate_downgrade(revision: str = None):
    """Migration'ları geri alır
    
    Args:
        revision: Geri alınacak migration revision ID'si
                 - Belirtilmezse: Son migration'ı geri al (-1)
                 - Belirli ID: O migration'a geri dön (örn: "10418ef861b7")
                 - "base": Tüm migration'ları geri al
                 - "-1": Son migration'ı geri al
    """
    if revision is None:
        revision = "-1"
    _run_alembic_command(["downgrade", revision])


def migrate_history():
    """Migration geçmişini gösterir"""
    _run_alembic_command(["history"])


def migrate_current():
    """Mevcut migration durumunu gösterir"""
    _run_alembic_command(["current"])

