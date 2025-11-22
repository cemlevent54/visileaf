# Virtual environment oluşturma
python -m venv .venv
.venv\Scripts\activate.bat

# Projeyi başlatma (npm start benzeri)
python -m app start

# Development modunda başlatma
python -m app dev

# Migration komutları (npm run db:migrate benzeri)
python -m app migrate                    # Tüm migration'ları uygula
python -m app migrate create "mesaj"    # Yeni migration oluştur

# Migration'ları uygulama (tek tek veya toplu)
python -m app migrate upgrade head      # Tüm migration'ları uygula
python -m app migrate upgrade +1        # Tek tek - bir sonraki migration'ı uygula
python -m app migrate upgrade +2        # İki sonraki migration'ı uygula
python -m app migrate upgrade <rev>    # Belirli migration'a kadar uygula

# Migration'ları geri alma
python -m app migrate downgrade         # Son migration'ı geri al
python -m app migrate downgrade -1      # Son migration'ı geri al
python -m app migrate downgrade -2      # İki önceki migration'a geri dön
python -m app migrate downgrade <rev>  # Belirli migration'ı geri al (örn: 10418ef861b7)
python -m app migrate downgrade base    # Tüm migration'ları geri al

# Migration bilgileri
python -m app migrate history           # Migration geçmişi
python -m app migrate current           # Mevcut migration durumu

# Eski komutlar (hala çalışır)
uvicorn app.main:app --reload
python -m app.main
