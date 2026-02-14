"""
Elite Sniper v2.0 - Configuration Module
Enhanced with proxy support, timing thresholds, and category mappings
"""

import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv("config.env")


class Config:
    """Centralized configuration for Elite Sniper v2.0"""
    
    # ==================== Telegram ====================
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    # ==================== Manual Captcha Settings ====================
    # When OCR fails, send captcha to Telegram for manual solving
    MANUAL_CAPTCHA_ENABLED = os.getenv("MANUAL_CAPTCHA", "true").lower() == "true"
    MANUAL_CAPTCHA_TIMEOUT = int(os.getenv("MANUAL_CAPTCHA_TIMEOUT", "60"))  # seconds

    # ==================== CapSolver Settings ====================
    CAPSOLVER_API_KEY = os.getenv("CAPSOLVER_API_KEY")
    CAPSOLVER_ENABLED = os.getenv("CAPSOLVER_ENABLED", "true").lower() == "true"
    
    # ==================== User Data ====================
    LAST_NAME = os.getenv("LAST_NAME")
    FIRST_NAME = os.getenv("FIRST_NAME")
    EMAIL = os.getenv("EMAIL")
    PASSPORT = os.getenv("PASSPORT")
    PHONE = os.getenv("PHONE")
    
    # ==================== Target ====================
    TARGET_URL = os.getenv("TARGET_URL")
    TIMEZONE = "Asia/Aden"  # GMT+3
    
    # ==================== Proxies (Disabled) ====================
    PROXIES = []
    # PROXIES = [
    #     os.getenv("PROXY_1"),
    #     os.getenv("PROXY_2"),
    #     os.getenv("PROXY_3"),
    # ]
    
    # ==================== Session Thresholds ====================
    SESSION_MAX_AGE = 300          # Maximum session age in seconds (REDUCED from 60 - server times out faster!)
    SESSION_MAX_IDLE = 12         # Maximum idle time before refresh (REDUCED from 15)
    HEARTBEAT_INTERVAL = 8        # Keep-alive interval in seconds (REDUCED from 10)
    MAX_CAPTCHA_ATTEMPTS = 5      # Per session before rebirth
    MAX_CONSECUTIVE_ERRORS = 3    # Before forced rebirth
    
    # ==================== Resilience & Performance ====================
    CIRCUIT_BREAKER_THRESHOLD = 2  # Max API failures before cooling down
    CIRCUIT_BREAKER_TIMEOUT = 300  # Cool-down period in seconds (5 mins)
    PARALLEL_SOLVING_ENABLED = True # Enable concurrent local + API solving
    
    # ==================== Booking Purpose ====================
    # Valid values: study, student, work, family, tourism, other
    PURPOSE = os.getenv("PURPOSE", "study")
    
    # ==================== Timing Configuration ====================
    # Default is 2 (2:00 AM Aden). Set ATTACK_HOUR in config.env to override (e.g. 19 for 7 PM).
    ATTACK_HOUR = int(os.getenv("ATTACK_HOUR", "2"))
    PRE_ATTACK_MINUTE = 59        # Pre-attack minute (1:59 AM)
    PRE_ATTACK_SECOND = 30        # Pre-attack second (1:59:30 AM)
    ATTACK_WINDOW_MINUTES = 60    # Full hour duration
    
    # ==================== Sleep Intervals ====================
    PATROL_SLEEP_MIN = 10.0       # Normal patrol minimum sleep
    PATROL_SLEEP_MAX = 20.0       # Normal patrol maximum sleep
    WARMUP_SLEEP = 5.0            # Warmup mode sleep
    ATTACK_SLEEP_MIN = 0.5        # Attack mode minimum sleep
    ATTACK_SLEEP_MAX = 1.5        # Attack mode maximum sleep
    PRE_ATTACK_SLEEP = 0.5        # Pre-attack ready state
    
    # ==================== Smart Targeting Configuration ====================
    # The bot will scan dropdown options for these keywords in order.
    # Priority 1 is checked first. If found, it's selected immediately.
    TARGET_KEYWORDS = [
        "Yemeni national",       # Priority 1: (The Specific Target)
        "national language",     # Priority 2
        "student visa",          # Priority 3
        "Student",               # Priority 4
        "Studium",               # Priority 5
        "Sprachkurs",            # Priority 6
        "University",            # Priority 7
        "Course"                 # Priority 8 (Fallback)
    ]
    
    # ==================== NTP Servers ====================
    NTP_SERVERS = [
        "pool.ntp.org",
        "time.google.com",
        "time.windows.com",
        "time.nist.gov"
    ]
    NTP_SYNC_INTERVAL = 300  # Re-sync every 5 minutes
    
    # ==================== Browser Configuration ====================
    HEADLESS = True
    BROWSER_ARGS = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--no-first-run",
        "--disable-extensions"
    ]
    
    # ==================== Evidence Configuration ====================
    EVIDENCE_DIR = "evidence"
    MAX_EVIDENCE_AGE_HOURS = 48  # Auto-cleanup after 48 hours

    # ==================== Development ====================
    DRY_RUN = True # FORCE TRUE FOR TESTING
    
    # ==================== Execution Mode ====================
    # AUTO, MANUAL, HYBRID
    # Allow explicit override via CAPTCHA_MANUAL_ONLY
    CAPTCHA_MANUAL_ONLY = os.getenv("CAPTCHA_MANUAL_ONLY", "false").lower() == "true"
    
    if CAPTCHA_MANUAL_ONLY:
        EXECUTION_MODE = "MANUAL"
    else:
        EXECUTION_MODE = os.getenv("EXECUTION_MODE", "HYBRID").upper()

