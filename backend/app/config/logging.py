import logging
import os
import sys
from datetime import datetime
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_log_level_from_env() -> str:
    """
    .env dosyasından LOG_LEVEL değerini okur
    Eğer .env dosyası yoksa veya LOG_LEVEL tanımlı değilse "INFO" döner
    
    Returns:
        Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if DOTENV_AVAILABLE:
        # .env dosyasını yükle
        load_dotenv()
    
    # LOG_LEVEL'ı oku, yoksa "INFO" kullan
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Geçerli log seviyelerini kontrol et
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        # Geçersiz seviye ise INFO kullan
        log_level = "INFO"
    
    return log_level


class ColoredFormatter(logging.Formatter):
    """Renkli log formatter - colorlog yoksa ANSI renk kodları kullanır"""
    
    # ANSI renk kodları
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Tarih formatı: YYYY-MM-DD HH:MM:SS
        log_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Log seviyesine göre renk
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format: [TARİH] SEVİYE: Mesaj
        if COLORLOG_AVAILABLE:
            # colorlog kullanılıyorsa, formatı ona bırak
            return super().format(record)
        else:
            # ANSI renk kodları ile format
            level_name = f"{level_color}{record.levelname:8s}{reset_color}"
            message = record.getMessage()
            return f"[{log_time}] {level_name} | {message}"


def setup_logging(
    level: str = "INFO",
    use_colorlog: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Logging yapılandırmasını ayarlar (tüm logger'lar için - Uvicorn dahil)
    
    Args:
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_colorlog: colorlog kütüphanesi kullanılsın mı (True ise ve yüklüyse kullanır)
        log_file: Log dosyası yolu (None ise sadece konsola yazar)
    
    Returns:
        Yapılandırılmış logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Formatter oluştur
    if use_colorlog and COLORLOG_AVAILABLE:
        # colorlog kullan
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'magenta',
            }
        )
    else:
        # Custom renkli formatter kullan
        console_formatter = ColoredFormatter()
    
    # Konsol handler oluştur (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Root logger'ı yapılandır
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.propagate = False
    
    # Uvicorn logger'larını yapılandır
    # Uvicorn başlamadan önce bu logger'ları yapılandırmak önemli
    uvicorn_loggers = [
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
    ]
    
    for uvicorn_logger in uvicorn_loggers:
        uvicorn_logger.setLevel(log_level)
        # Tüm handler'ları temizle
        uvicorn_logger.handlers.clear()
        # Root logger'a propagate et - bu sayede tüm loglar aynı formatter'ı kullanır
        uvicorn_logger.propagate = True
        # Ayrıca direkt handler da ekle (propagate bazen çalışmayabilir)
        uvicorn_logger.addHandler(console_handler)
    
    # Uvicorn'un gelecekte ekleyeceği handler'ları da yakalamak için
    # logging modülünün getLogger fonksiyonunu wrap edebiliriz
    # Ancak şimdilik yukarıdaki yapılandırma yeterli olmalı
    
    # Dosya handler (eğer belirtilmişse)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Uvicorn logger'larına da dosya handler ekle
        for uvicorn_logger in uvicorn_loggers:
            uvicorn_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Belirli bir isimle logger döndürür
    
    Args:
        name: Logger ismi (genellikle __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def reconfigure_uvicorn_loggers():
    """
    Uvicorn logger'larını yeniden yapılandırır
    Uvicorn başladıktan sonra handler'larını override etmek için kullanılır
    """
    log_level = logging.INFO
    
    # Root logger'dan handler'ı al
    root_logger = logging.getLogger()
    
    # Root logger'ın handler'ını kontrol et
    if not root_logger.handlers:
        # Handler yoksa oluştur
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if COLORLOG_AVAILABLE:
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'magenta',
                }
            )
        else:
            console_formatter = ColoredFormatter()
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    else:
        # Handler varsa, onu kullan
        console_handler = root_logger.handlers[0]
    
    # Uvicorn logger'larını yeniden yapılandır
    uvicorn_loggers = [
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
    ]
    
    for uvicorn_logger in uvicorn_loggers:
        uvicorn_logger.setLevel(log_level)
        # Mevcut handler'ları temizle
        for handler in uvicorn_logger.handlers[:]:
            uvicorn_logger.removeHandler(handler)
        # Root logger'a propagate et - bu sayede tüm loglar aynı formatter'ı kullanır
        uvicorn_logger.propagate = True
        # Ayrıca direkt handler da ekle (propagate bazen çalışmayabilir)
        uvicorn_logger.addHandler(console_handler)


def setUpLogging(
    level: Optional[str] = None,
    use_colorlog: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Tüm logging yapılandırmasını başlatır
    Bu fonksiyon hem ilk yapılandırmayı hem de Uvicorn başladıktan sonra
    yapılacak yapılandırmayı içerir.
    
    Args:
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              Eğer None ise .env dosyasından LOG_LEVEL okunur
        use_colorlog: colorlog kütüphanesi kullanılsın mı (True ise ve yüklüyse kullanır)
        log_file: Log dosyası yolu (None ise sadece konsola yazar)
    """
    # Log level'ı belirle: önce parametre, sonra .env, son olarak "INFO"
    if level is None:
        level = get_log_level_from_env()
    
    # İlk yapılandırmayı yap
    setup_logging(level=level, use_colorlog=use_colorlog, log_file=log_file)
    
    # Uvicorn başladıktan sonra logger'ları yeniden yapılandır
    # (Uvicorn kendi handler'larını eklemiş olabilir)
    reconfigure_uvicorn_loggers()

