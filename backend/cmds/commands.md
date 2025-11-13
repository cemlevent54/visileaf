python -m venv .venv
.venv\Scripts\activate.bat
uvicorn app.main:app --reload

# Port .env dosyasından okunur (PORT=8000)
# Alternatif: python -m app.main
