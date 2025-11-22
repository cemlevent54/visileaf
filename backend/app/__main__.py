"""
Python modülü olarak çalıştırıldığında CLI komutlarını yürütür
Kullanım: python -m app start
         python -m app migrate
"""
from app.cli import *

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanılabilir komutlar:")
        print("  python -m app start              - Projeyi başlat")
        print("  python -m app dev                 - Development modunda başlat")
        print("  python -m app migrate             - Tüm migration'ları uygula")
        print("  python -m app migrate create 'mesaj' - Yeni migration oluştur")
        print("  python -m app migrate upgrade [revision] - Migration uygula")
        print("    Örnek: python -m app migrate upgrade head  (tüm migration'lar)")
        print("    Örnek: python -m app migrate upgrade +1  (tek tek - bir sonraki)")
        print("    Örnek: python -m app migrate upgrade +2  (iki sonraki)")
        print("    Örnek: python -m app migrate upgrade abc123  (belirli migration'a kadar)")
        print("  python -m app migrate downgrade [revision] - Migration geri al")
        print("    Örnek: python -m app migrate downgrade -1  (son migration)")
        print("    Örnek: python -m app migrate downgrade -2  (iki önceki)")
        print("    Örnek: python -m app migrate downgrade 10418ef861b7  (belirli migration)")
        print("    Örnek: python -m app migrate downgrade base  (tüm migration'lar)")
        print("  python -m app migrate history     - Migration geçmişi")
        print("  python -m app migrate current      - Mevcut migration durumu")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "start":
        start()
    elif command == "dev":
        dev()
    elif command == "migrate":
        # Alt komut kontrolü
        if len(sys.argv) > 2:
            subcommand = sys.argv[2]
            if subcommand == "create":
                if len(sys.argv) < 4:
                    print("Hata: Migration mesajı gerekli")
                    print("Kullanım: python -m app migrate create 'migration mesajı'")
                    sys.exit(1)
                migrate_create(sys.argv[3])
            elif subcommand == "upgrade":
                revision = sys.argv[3] if len(sys.argv) > 3 else "head"
                migrate_upgrade(revision)
            elif subcommand == "downgrade":
                # Revision belirtilmediyse -1 (son migration) kullan
                revision = sys.argv[3] if len(sys.argv) > 3 else None
                migrate_downgrade(revision)
            elif subcommand == "history":
                migrate_history()
            elif subcommand == "current":
                migrate_current()
            else:
                print(f"Bilinmeyen migrate alt komutu: {subcommand}")
                sys.exit(1)
        else:
            migrate()
    else:
        print(f"Bilinmeyen komut: {command}")
        sys.exit(1)

