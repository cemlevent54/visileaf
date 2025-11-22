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

tamamdır, endpoint implementasyonlarına geçiyoruz auth kısmı için.
öncelikle kod yapsıından bahsedeceğim. routes/ klasörü altında auth.py da 
auth route ları olacak /api/auth/register gibi mesela. user.py da da user 
route ları olacak /api/user gibi, user ı anlaman için örnek verdim, 
biz auth kısmını istiyoruz sadece. ardından controllers/ klasörüne geliyoruz, 
bu klasörde de http request response lar yapılacak sadece ve implementasyon 
kesinlikle servis e bırakılacak.
services/ klasöründe de metot implementasyonları olacak, tüm iş mantığı burada olacak.
bundan sonra da repositories/ klasörüne geldik, bu klasörde mesela userrepository.py
da kullanıcı oluşturma, güncelleme, silme, getirme gibi metotlar olacak. db sorgusu 
yazmak yerine db işlemleri models/ altındaki modellerden de yararlanarak 
burada yapılacak. kesinlikle sql sorgusu yok, modeller üzerinden yapılacak.

istediğim endpointler:

POST /api/auth/register
Request Body:
{
    "first_name": "",
    "last_name": "",
    "email": "",
    "password":""
}

Response json: 
{
    "success": true,
    "message": "User registered successfully",
    "data": {
        "user": {
            "id": "",
            "first_name": "",
            "last_name": "",
            "email": ""
        }
    }
}

POST /api/auth/login
Request Body:
{
    "email": "",
    "password":""
}

Response json: 
{
    "success": true,
    "data": {
        "user": {
            "id": "",
            "first_name": "",
            "last_name": "",
            "email": ""
        },
        "tokens": {
            "access": "",
            "refresh": ""
        }
    },
    "message": "User logged in successfully",
}

POST /api/auth/logout
Request Body:
{
    "refresh_token": ""
}

Response json: 
{
    "success": true,
    "message": "User logged out successfully",
}

GET /api/auth/me
Request Body:
{
    "access_token": ""
}

Response json: 
{
    "success": true,
    "data": {
        "user": {
            "id": "",
            "first_name": "",
            "last_name": "",
            "email": ""
        }
    },
    "message": "User me information",
}

POST /api/auth/refresh-token
Request Body:
{
    "refresh_token": ""
}

Response json: 
{
    "success": true,
    "message": "User refreshed token successfully",
}

POST /api/auth/forgot-password
Request Body:
{
    "email": ""
}

Response json: 
{
    "success": true,
    "data": {
        "user": {
            "id": "",
            "first_name": "",
            "last_name": "",
            "email": ""
        }
    },
    "message": "User forgot password successfully",
}

POST /api/auth/reset-password
Request Body:
{
    "token": "",
    "password":""
}

Response json: 
{
    "success": true,
    "data": {
        "user": {
            "id": "",
            "first_name": "",
            "last_name": "",
            "email": ""
        }
    },
    "message": "User reset password successfully",
}