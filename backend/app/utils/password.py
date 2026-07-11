"""
Password hashing utility functions
"""
import bcrypt


def hash_password(password: str) -> str:
    """
    Şifreyi hash'ler
    
    Args:
        password: Hash'lenecek şifre
    
    Returns:
        Hash'lenmiş şifre (UTF-8 string olarak)
    """
    # Şifreyi bytes'a çevir
    password_bytes = password.encode('utf-8')
    
    # bcrypt ile hash'le (salt otomatik oluşturulur)
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    
    # UTF-8 string olarak döndür
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Şifreyi doğrular
    
    Args:
        plain_password: Düz metin şifre
        hashed_password: Hash'lenmiş şifre
    
    Returns:
        Şifre doğruysa True, değilse False
    """
    try:
        # Şifreleri bytes'a çevir
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        # bcrypt ile doğrula
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False

