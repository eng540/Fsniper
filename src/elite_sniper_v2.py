"""
Elite Sniper v2.0 - Production-Grade Multi-Session Appointment Booking System

Integrates best features from:
- Elite Sniper: Multi-session architecture, Scout/Attacker pattern, Scheduled activation
- KingSniperV12: State Machine, Soft Recovery, Safe Captcha Check, Debug utilities

Architecture:
- 3 Parallel Sessions (1 Scout + 2 Attackers)
- 24/7 Operation with 2:00 AM Aden time activation
- Intelligent session lifecycle management
- Production-grade error handling and recovery

Version: 2.0.0
"""

import time
import random
import datetime
import logging
import os
import sys
import re
from typing import List, Tuple, Optional, Dict, Any
from threading import Thread, Event, Lock
from dataclasses import asdict

import pytz
from playwright.sync_api import sync_playwright, Page, BrowserContext, Browser

# Internal imports
try:
    from .config import Config
except ImportError:
    from config import Config
from .ntp_sync import NTPTimeSync
from .session_state import (
    SessionState, SessionStats, SystemState, SessionHealth, 
    SessionRole, Incident, IncidentManager, IncidentType, IncidentSeverity
)
from .captcha import EnhancedCaptchaSolver
from .notifier import send_alert, send_photo, send_success_notification, send_status_update
from .debug_utils import DebugManager
from .page_flow import PageFlowDetector

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elite_sniper_v2.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger("EliteSniperV2")


class EliteSniperV2:
    """
    Production-Grade Multi-Session Appointment Booking System
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, run_mode: str = "AUTO"):
        """Initialize Elite Sniper v2.0"""
        self.run_mode = run_mode
        
        logger.info("=" * 70)
        logger.info(f"[INIT] ELITE SNIPER V{self.VERSION} - INITIALIZING")
        logger.info(f"[MODE] Running Mode: {self.run_mode}")
        logger.info("=" * 70)
        
        # Validate configuration
        self._validate_config()
        
        # Session management
        self.session_id = f"elite_v2_{int(time.time())}_{random.randint(1000, 9999)}"
        self.start_time = datetime.datetime.now()
        
        # System state
        self.system_state = SystemState.STANDBY
        self.stop_event = Event()      # Global kill switch
        self.slot_event = Event()      # Scout → Attacker signal
        self.target_url: Optional[str] = None  # Discovered appointment URL
        self.lock = Lock()              # Thread-safe coordination
        self.screenshot_requested = Event()  # Flag for screenshot request

        
        
        # Initialize Telegram C2 FIRST so we can pass it
        try:
            from .telegram_c2 import TelegramCommander
            self.c2 = TelegramCommander(bot_instance=self)
            self.c2.start()
        except Exception as e:
            logger.error(f"[C2] Failed to start Telegram Commander: {e}")
            self.c2 = None
            
        # Components
        # EXECUTION MODES: AUTO, MANUAL, HYBRID
        self.mode = Config.EXECUTION_MODE
        logger.info(f"[MODE] Execution Strategy: {self.mode}")
        
        # Pass C2 to solver
        self.solver = EnhancedCaptchaSolver(mode=self.mode, c2_instance=self.c2)
        
        self.debug_manager = DebugManager(self.session_id, Config.EVIDENCE_DIR)
        self.incident_manager = IncidentManager()
        self.ntp_sync = NTPTimeSync(Config.NTP_SERVERS, Config.NTP_SYNC_INTERVAL)
        self.page_flow = PageFlowDetector()  # For accurate page type detection
        self.paused = Event() # Pause control
        
        # Configuration
        self.base_url = self._prepare_base_url(Config.TARGET_URL)
        self.timezone = pytz.timezone(Config.TIMEZONE)
        
        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ]
        
        # Proxies (optional)
        self.proxies = self._load_proxies()
        
        # Global statistics
        self.global_stats = SessionStats()
        
        # Start background NTP sync
        self.ntp_sync.start_background_sync()
        
        logger.info(f"[ID] Session ID: {self.session_id}")
        logger.info(f"[URL] Base URL: {self.base_url[:60]}...")
        logger.info(f"[TZ] Timezone: {self.timezone}")
        logger.info(f"[NTP] NTP Offset: {self.ntp_sync.offset:.4f}s")
        logger.info(f"[DIR] Evidence Dir: {self.debug_manager.session_dir}")
        logger.info(f"[PROXY] Proxies: {len([p for p in self.proxies if p])} configured")
        logger.info(f"[OK] Initialization complete")
    
    # ==================== Configuration ====================
    
    def request_screenshot(self):
        """Set flag to request screenshot on next loop cycle"""
        self.screenshot_requested.set()

    # ==================== C2 Interface Methods ====================
    def set_mode(self, new_mode: str): 
        """Update execution mode correctly""" 
        valid_modes = ["AUTO", "MANUAL", "HYBRID"] 
        if new_mode.upper() in valid_modes: 
            self.mode = new_mode.upper() 
            self.run_mode = new_mode.upper() # Sync both
            # Update solver mode as well 
            if hasattr(self, 'solver'): 
                self.solver.mode = self.mode 
            logger.info(f"[MODE] Switched to {self.mode}") 
            return True 
        return False

    def pause_execution(self): 
        """Pause the scanning loop""" 
        if not self.paused.is_set(): 
            self.paused.set() 
            logger.info("⏸️ System PAUSED by user")

    def resume_execution(self): 
        """Resume the scanning loop""" 
        if self.paused.is_set(): 
            self.paused.clear() 
            logger.info("▶️ System RESUMED by user")

    def get_status_report(self) -> str:
        """Generate status report for C2"""
        stats = self.global_stats
        status = "🟢 Running" if not self.paused.is_set() else "⏸ Paused"
        
        return (
            f"📊 <b>System Status</b>\n"
            f"Mode: {self.run_mode}\n"
            f"State: {status}\n\n"
            f"📉 <b>Statistics</b>:\n"
            f"Days Found: {stats.days_found}\n"
            f"Slots Found: {stats.slots_found}\n"
            f"Forms Filled: {stats.forms_filled}\n"
            f"Captchas: {stats.captchas_solved}/{stats.captchas_solved + stats.captchas_failed}\n"
        )
    
    def force_screenshot(self) -> Optional[str]:
        """Take immediate screenshot for C2"""
        try:
            # But we can store a reference or use a shared variable
            # IMPROVEMENT: _run_single_session writes 'self.current_page'
            
            if hasattr(self, 'current_page') and self.current_page:
                try:
                    timestamp = int(time.time())
                    filename = f"c2_shot_{timestamp}.jpg"
                    path = os.path.join(Config.EVIDENCE_DIR, filename)
                    self.current_page.screenshot(path=path)
                    return path
                except Exception as e:
                    logger.error(f"[C2] Screenshot failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[C2] Force screenshot error: {e}")
            return None

    def _validate_config(self):
        """Validate required configuration"""
        required = [
            'TARGET_URL', 'LAST_NAME', 'FIRST_NAME', 
            'EMAIL', 'PASSPORT', 'PHONE'
        ]
        
        missing = [field for field in required if not getattr(Config, field, None)]
        
        if missing:
            raise ValueError(f"[ERR] Missing configuration: {', '.join(missing)}")
        
        logger.info("[OK] Configuration validated")
    
    def cleanup(self):
        """
        Robust resource cleanup (Anti-Zombie)
        Ensures all threads, browsers, and syncs are terminated
        """
        logger.info("[CLEANUP] Initiating robust shutdown...")
        
        # 1. Signal Stop
        self.stop_event.set()
        
        # 2. Stop NTP
        try:
            if hasattr(self, 'ntp_sync'):
                self.ntp_sync.stop_background_sync()
        except: pass
        
        # 3. Close Browser (if managed here)
        # Note: In current architecture, browser is passed IN, but we should close contexts
        # The main run method handles browser.close(), but we can add safety checks here
        
        # 4. Stop C2
        try:
            if hasattr(self, 'c2') and self.c2:
                self.c2.stop()
        except: pass
        
        logger.info("[CLEANUP] Resources released")
    
    def _prepare_base_url(self, url: str) -> str:
        """Prepare base URL with locale"""
        if "request_locale" not in url:
            separator = "&" if "?" in url else "?"
            return f"{url}{separator}request_locale=en"
        return url
    
    def _load_proxies(self) -> List[Optional[str]]:
        """Load proxies from config or file"""
        proxies = []
        
        # From Config.PROXIES
        if hasattr(Config, 'PROXIES') and Config.PROXIES:
            proxies.extend([p for p in Config.PROXIES if p])
        
        # From proxies.txt
        try:
            if os.path.exists("proxies.txt"):
                with open("proxies.txt") as f:
                    file_proxies = [line.strip() for line in f if line.strip()]
                    proxies.extend(file_proxies)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load proxies.txt: {e}")
        
        # Ensure we have at least 3 slots (None = direct connection)
        while len(proxies) < 3:
            proxies.append(None)
        
        return proxies[:3]  # Only use first 3
    
    # ==================== Time Management ====================
    
    def get_current_time_aden(self) -> datetime.datetime:
        """Get current time in Aden timezone with NTP correction"""
        corrected_utc = self.ntp_sync.get_corrected_time()
        aden_time = corrected_utc.replace(tzinfo=pytz.UTC).astimezone(self.timezone)
        return aden_time
    
    def is_pre_attack(self) -> bool:
        """Check if in pre-attack window (1:59:30 - 1:59:59 Aden time)"""
        now = self.get_current_time_aden()
        return (now.hour == 1 and 
                now.minute == Config.PRE_ATTACK_MINUTE and 
                now.second >= Config.PRE_ATTACK_SECOND)
    
    def is_attack_time(self) -> bool:
        """Check if in Golden Hour (2:00:00 - 3:00:00 Aden time)"""
        now = self.get_current_time_aden()
        # Full hour attack window (2:00 - 2:59)
        return now.hour == Config.ATTACK_HOUR
    
    def get_sleep_interval(self) -> float:
        """
        Calculate dynamic sleep interval based on current mode
        
        Strategies:
        1. Golden Hour (Attack): Sleep 30s (Persistent Session Strategy)
        2. Pre-Attack: Sleep 0.5s (Ready State)
        3. Patrol (Normal): Sleep until next 20-minute mark (:00, :20, :40)
        """
        if self.is_attack_time():
            # Persistent Session Strategy: Wait 30s inside the session
            return 30.0
            
        elif self.is_pre_attack():
            return Config.PRE_ATTACK_SLEEP
            
        else:
            now = self.get_current_time_aden()
            
            # WARMUP (1:45 - 2:00)
            if now.hour == 1 and now.minute >= 45:
                return Config.WARMUP_SLEEP
            
            # PATROL: 20-minute alignment logic
            # Calculate next target minute (0, 20, 40, 60)
            current_minute = now.minute
            current_second = now.second
            
            if current_minute < 20:
                target_minute = 20
            elif current_minute < 40:
                target_minute = 40
            else:
                target_minute = 60 # Next hour :00
                
            minutes_to_wait = target_minute - current_minute - 1
            seconds_to_wait = 60 - current_second
            total_wait = (minutes_to_wait * 60) + seconds_to_wait
            
            # Subtract buffer (5 seconds) to be ready slightly before
            final_wait = max(10.0, total_wait - 5.0)
            
            logger.info(f"⏳ [SMART SCHEDULE] Aligning to :{(target_minute%60):02d} - Sleeping {final_wait:.1f}s")
            return final_wait
    
    def get_mode(self) -> str:
        """Get current operational mode"""
        if self.is_attack_time():
            return "ATTACK"
        elif self.is_pre_attack():
            return "PRE_ATTACK"
        else:
            now = self.get_current_time_aden()
            if now.hour == 1 and now.minute >= 45:
                return "WARMUP"
            return "PATROL"
    
    # ==================== Session Management ====================
    
    def create_context(
        self, 
        browser: Browser, 
        worker_id: int,
        proxy: Optional[str] = None
    ) -> Tuple[BrowserContext, Page, SessionState]:
        """
        Create browser context with session state
        
        Args:
            browser: Playwright browser instance
            worker_id: Worker ID (1-3)
            proxy: Optional proxy server
        
        Returns:
            (context, page, session_state)
        """
        try:
            # Determine role
            role = SessionRole.SCOUT if worker_id == 1 else SessionRole.ATTACKER
            
            # Select user agent
            user_agent = random.choice(self.user_agents)
            
            # Randomize viewport slightly for fingerprint variation
            viewport_width = 1366 + random.randint(0, 50)
            viewport_height = 768 + random.randint(0, 30)
            
            # Context arguments
            context_args = {
                "user_agent": user_agent,
                "viewport": {"width": viewport_width, "height": viewport_height},
                "locale": "en-US",
                "timezone_id": "Asia/Aden",
                "ignore_https_errors": True
            }
            
            # Add proxy if provided
            if proxy:
                context_args["proxy"] = {"server": proxy}
                logger.info(f"[PROXY] [W{worker_id}] Using proxy: {proxy[:30]}...")
            
            # Create context
            context = browser.new_context(**context_args)
            page = context.new_page()
            
            # Anti-detection + Keep-Alive script
            page.add_init_script(f"""
                // Hide webdriver
                Object.defineProperty(navigator, 'webdriver', {{ 
                    get: () => undefined 
                }});
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {{
                    get: () => [1, 2, 3, 4, 5]
                }});
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {{
                    get: () => ['en-US', 'en']
                }});
                
                // Session keep-alive heartbeat (every {Config.HEARTBEAT_INTERVAL}s)
                setInterval(() => {{
                    fetch(location.href, {{ method: 'HEAD' }}).catch(()=>{{}});
                }}, {Config.HEARTBEAT_INTERVAL * 1000});
            """)
            
            # Timeouts
            context.set_default_timeout(45000)
            context.set_default_navigation_timeout(120000)
            
            # Resource blocking for performance
            def route_handler(route):
                resource_type = route.request.resource_type
                if resource_type in ["image", "media", "font", "stylesheet"]:
                    route.abort()
                else:
                    route.continue_()
            
            page.route("**/*", route_handler)
            
            # Create session state with config limits
            session_state = SessionState(
                session_id=f"{self.session_id}_w{worker_id}",
                role=role,
                worker_id=worker_id,
                max_age=300,  # FORCE 5 MINUTES (Essential for multi-month sequential scan)
                max_idle=Config.SESSION_MAX_IDLE,
                max_failures=Config.MAX_CONSECUTIVE_ERRORS,
                max_captcha_attempts=Config.MAX_CAPTCHA_ATTEMPTS
            )
            
            logger.info(f"[CTX] [W{worker_id}] Context created - Role: {role.value}")
            
            with self.lock:
                self.global_stats.rebirths += 1
            
            return context, page, session_state
            
        except Exception as e:
            logger.error(f"[ERR] [W{worker_id}] Context creation failed: {e}")
            raise
    
    def validate_session_health(
        self, 
        page: Page, 
        session: SessionState, 
        location: str = "UNKNOWN"
    ) -> bool:
        """
        Validate session health with strict kill rules
        
        Returns:
            True if session is healthy, False if should be terminated
        """
        worker_id = session.worker_id
        
        # Rule 1: Session expired (age > 60s or idle > 15s)
        if session.is_expired():
            age = session.age()
            idle = session.idle_time()
            logger.critical(
                f"[EXP] [W{worker_id}][{location}] "
                f"Session EXPIRED - Age: {age:.1f}s, Idle: {idle:.1f}s"
            )
            self.incident_manager.create_incident(
                session.session_id, IncidentType.SESSION_EXPIRED,
                IncidentSeverity.CRITICAL,
                f"Session expired: age={age:.1f}s, idle={idle:.1f}s"
            )
            return False
        
        # Rule 2: Too many failures
        if session.should_terminate():
            logger.critical(
                f"💀 [W{worker_id}][{location}] "
                f"Session POISONED - Failures: {session.failures}"
            )
            self.incident_manager.create_incident(
                session.session_id, IncidentType.SESSION_POISONED,
                IncidentSeverity.CRITICAL,
                f"Session poisoned: failures={session.failures}"
            )
            return False
        
        # Rule 3: Double captcha detection (captcha appears twice in same flow)
        if session.captcha_solved:
            has_captcha, _ = self.solver.safe_captcha_check(page, location)
            if has_captcha:
                logger.critical(
                    f"💀 [W{worker_id}][{location}] "
                    f"DOUBLE CAPTCHA detected - Session INVALID"
                )
                session.health = SessionHealth.POISONED
                self.incident_manager.create_incident(
                    session.session_id, IncidentType.DOUBLE_CAPTCHA,
                    IncidentSeverity.CRITICAL,
                    "Double captcha in same flow - session poisoned"
                )
                return False
        
        # Rule 4: Silent rejection (form still visible after submit)
        if location == "POST_SUBMIT":
            try:
                if page.locator("input[name='lastname']").count() > 0:
                    logger.critical(
                        f"🔁 [W{worker_id}][{location}] "
                        f"Silent rejection - Form reappeared"
                    )
                    self.incident_manager.create_incident(
                        session.session_id, IncidentType.FORM_REJECTED,
                        IncidentSeverity.ERROR,
                        "Form reappeared after submit - silent rejection"
                    )
                    return False
            except:
                pass
        
        # Rule 5: Bounce detection (month captcha in form view)
        if location == "FORM":
            try:
                if page.locator("form#appointment_captcha_month").count() > 0:
                    logger.critical(
                        f"↩️ [W{worker_id}][{location}] "
                        f"Bounced to month captcha"
                    )
                    return False
            except:
                pass
        
        # Session is healthy
        session.touch()
        return True
    
    def soft_recovery(self, session: SessionState, reason: str):
        """
        Soft recovery without full context recreation
        From KingSniperV12
        """
        logger.info(f"🔄 [W{session.worker_id}] Soft recovery: {reason}")
        
        # Reset counters
        session.consecutive_errors = 0
        session.failures = max(0, session.failures - 1)  # Forgive one failure
        
        # Update health
        if session.health == SessionHealth.DEGRADED:
            session.health = SessionHealth.WARNING
        elif session.health == SessionHealth.WARNING:
            session.health = SessionHealth.CLEAN
        
        session.touch()
        
        logger.info(f"✅ [W{session.worker_id}] Soft recovery completed")
    
    # ==================== Navigation & Form Filling ====================
    
    def generate_month_urls(self) -> List[str]:
        """Generate priority month URLs - STRICT SEQUENTIAL ORDER [2, 3, 4, 5]"""
        try:
            today = datetime.datetime.now().date()
            # Ensure base URL is clean
            base_clean = self.base_url.split("&dateStr=")[0] if "&dateStr=" in self.base_url else self.base_url
            
            urls = []
            # MANDATORY ORDER: Month 4, 5, 2, 3 (Focus on Aprile/May first as requested)
            # Scan April, May, February, March sequentially
            priority_offsets = [4, 5, 2, 3] 
            
            for offset in priority_offsets:
                future_date = today + datetime.timedelta(days=30 * offset)
                # Set to 15th to ensure we land in the middle of the target month
                date_str = f"15.{future_date.month:02d}.{future_date.year}" 
                url = f"{base_clean}&dateStr={date_str}"
                urls.append(url)
            
            logger.info(f"📋 Generated {len(urls)} month URLs: [4, 5, 2, 3] months ahead")
            return urls
            
        except Exception as e:
            logger.error(f"❌ Month URL generation failed: {e}")
            return []
    
    def fast_inject(self, page: Page, selector: str, value: str) -> bool:
        """
        Inject value into form field using Playwright native methods first,
        then JavaScript fallback for reliability.
        """
        try:
            locator = page.locator(selector)
            if locator.count() == 0:
                logger.warning(f"[INJECT] Selector not found: {selector}")
                return False
            
            # Method 1: Use Playwright's native fill() - most reliable
            try:
                locator.first.fill(value, timeout=2000)
                logger.debug(f"[INJECT] Filled via Playwright: {selector}")
                return True
            except Exception as e1:
                logger.debug(f"[INJECT] Playwright fill failed for {selector}: {e1}")
            
            # Method 2: Click then type
            try:
                locator.first.click(timeout=1000)
                locator.first.fill(value, timeout=2000)
                return True
            except Exception as e2:
                logger.debug(f"[INJECT] Click+fill failed for {selector}: {e2}")
            
            # Method 3: JavaScript injection as fallback
            try:
                escaped_value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
                page.evaluate(f"""
                    const el = document.querySelector("{selector}");
                    if(el) {{ 
                        el.value = "{escaped_value}"; 
                        el.dispatchEvent(new Event('input', {{ bubbles: true }})); 
                        el.dispatchEvent(new Event('change', {{ bubbles: true }})); 
                        el.dispatchEvent(new Event('blur', {{ bubbles: true }})); 
                    }}
                """)
                logger.debug(f"[INJECT] Filled via JS: {selector}")
                return True
            except Exception as e3:
                logger.warning(f"[INJECT] JS injection failed for {selector}: {e3}")
            
            return False
            
        except Exception as e:
            logger.warning(f"[INJECT] All methods failed for {selector}: {e}")
            return False
    
    def find_input_id_by_label(self, page: Page, label_text: str) -> Optional[str]:
        """Find input ID by label text"""
        try:
            return page.evaluate(f"""
                () => {{
                    const labels = Array.from(document.querySelectorAll('label'));
                    const target = labels.find(l => l.innerText.toLowerCase().includes("{label_text.lower()}"));
                    return target ? target.getAttribute('for') : null;
                }}
            """)
        except:
            return None
    
    def select_category_by_value(self, page: Page) -> bool:
        """
        Smart Targeting: Select category using keyword priority search.
        Scans dropdown options for TARGET_KEYWORDS in order of priority.
        First match wins - immediately selects and triggers change events.
        """
        try:
            # Find all select elements
            selects = page.locator("select").all()
            
            if not selects:
                logger.warning("[CATEGORY] No select elements found on page")
                return False
            
            # Collect all options from all selects with their metadata
            all_options = []
            for select in selects:
                try:
                    options = select.locator("option").all()
                    for option in options:
                        text = option.inner_text().strip()
                        value = option.get_attribute("value")
                        if text and value:  # Skip empty options
                            all_options.append({
                                "select": select,
                                "option": option,
                                "text": text,
                                "text_lower": text.lower(),
                                "value": value
                            })
                except Exception:
                    continue
            
            logger.info(f"[CATEGORY] Found {len(all_options)} dropdown options to scan")
            
            # Priority-based keyword search
            for priority, keyword in enumerate(Config.TARGET_KEYWORDS, start=1):
                keyword_lower = keyword.lower()
                
                for opt in all_options:
                    if keyword_lower in opt["text_lower"]:
                        # MATCH FOUND! Select immediately
                        try:
                            opt["select"].select_option(value=opt["value"])
                            logger.info(f"[CATEGORY] Priority {priority} MATCH: '{keyword}' -> '{opt['text']}' (value={opt['value']})")
                            
                            # Trigger change and input events for server-side detection
                            page.evaluate("""
                                const selects = document.querySelectorAll('select');
                                selects.forEach(s => {
                                    s.dispatchEvent(new Event('input', { bubbles: true }));
                                    s.dispatchEvent(new Event('change', { bubbles: true }));
                                });
                            """)
                            return True
                        except Exception as e:
                            logger.warning(f"[CATEGORY] Selection failed for '{opt['text']}': {e}")
                            continue
            
            # No keyword match found - fallback to 2nd option (Index 1)
            logger.warning("[CATEGORY] No keyword match found, using fallback (Option 2)")
            
            # Filter out empty/placeholder options
            valid_options = [opt for opt in all_options if opt["value"]]
            
            if len(valid_options) >= 2:
                fallback_opt = valid_options[1] # Index 1 is the second option
                try:
                    fallback_opt["select"].select_option(value=fallback_opt["value"])
                    logger.info(f"[CATEGORY] Fallback selected: '{fallback_opt['text']}' (value={fallback_opt['value']})")
                    
                    page.evaluate("""
                        const selects = document.querySelectorAll('select');
                        selects.forEach(s => {
                            s.dispatchEvent(new Event('input', { bubbles: true }));
                            s.dispatchEvent(new Event('change', { bubbles: true }));
                        });
                    """)
                    return True
                except Exception as e:
                    logger.warning(f"[CATEGORY] Fallback selection failed: {e}")
            else:
                logger.warning("[CATEGORY] Not enough options for fallback selection")
            
            return False
            
        except Exception as e:
            logger.warning(f"[CATEGORY] Smart targeting error: {e}")
            return False
    
    def fill_booking_form(self, page: Page, session: SessionState) -> bool:
        """
        Fill the booking form with user data
        Uses Surgeon's Injection for reliability
        """
        worker_id = session.worker_id
        logger.info(f"📝 [W{worker_id}] Filling booking form...")
        
        try:
            # 1. Standard Fields
            self.fast_inject(page, "input[name='lastname']", Config.LAST_NAME)
            self.fast_inject(page, "input[name='firstname']", Config.FIRST_NAME)
            self.fast_inject(page, "input[name='email']", Config.EMAIL)
            
            # Email repeat (try both variants)
            if not self.fast_inject(page, "input[name='emailrepeat']", Config.EMAIL):
                self.fast_inject(page, "input[name='emailRepeat']", Config.EMAIL)
            
            # 2. Dynamic Fields (Passport, Phone)
            phone_value = Config.PHONE.replace("+", "00").strip()
            
            # Try finding by label first
            passport_id = self.find_input_id_by_label(page, "Passport")
            if passport_id:
                self.fast_inject(page, f"#{passport_id}", Config.PASSPORT)
            else:
                self.fast_inject(page, "input[name='fields[0].content']", Config.PASSPORT)
            
            phone_id = self.find_input_id_by_label(page, "Telephone")
            if phone_id:
                self.fast_inject(page, f"#{phone_id}", phone_value)
            else:
                self.fast_inject(page, "input[name='fields[1].content']", phone_value)
            
            # 3. Category Selection
            self.select_category_by_value(page)
            
            with self.lock:
                self.global_stats.forms_filled += 1
            
            # Save debug evidence
            self.debug_manager.save_debug_html(page, "form_filled", worker_id)
            
            logger.info(f"✅ [W{worker_id}] Form filled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ [W{worker_id}] Form fill error: {e}")
            return False
    
    def submit_form(self, page: Page, session: SessionState) -> bool:
        """
        AGGRESSIVE FORM SUBMISSION - NO SESSION CHECK!
        
        CRITICAL: When we reach this point, we HAVE the slot and MUST submit.
        Session validation would kill our chance - SKIP IT!
        
        Uses hybrid approach: Enter on captcha + Click submit button
        """
        worker_id = session.worker_id
        logger.info(f"[W{worker_id}] === AGGRESSIVE SUBMIT STARTED ===")
        
        # DRY RUN CHECK
        if Config.DRY_RUN:
            logger.warning(f"🛑 [W{worker_id}] [DRY_RUN] Skipping actual submission!")
            logger.info(f"[W{worker_id}] [DRY_RUN] Taking proof of concept screenshot...")
            
            # Save evidence
            self.debug_manager.save_critical_screenshot(page, "DRY_RUN_SUCCESS", worker_id)
            self.debug_manager.save_debug_html(page, "DRY_RUN_SUCCESS", worker_id)
            
            logger.info(f"[W{worker_id}] [DRY_RUN] Simulating successful booking...")
            
            # Simulate success state
            with self.lock:
                self.global_stats.success = True
            self.stop_event.set()
            return True
        
        max_attempts = 15  # Increased retries
        
        for attempt in range(1, max_attempts + 1):
            try:
                # ===============================================
                # CRITICAL: DO NOT CHECK SESSION HEALTH!
                # We have the slot, we MUST try to submit anyway
                # ===============================================
                
                # Method 1: Check if captcha input exists and press Enter
                captcha_input = page.locator("input[name='captchaText']").first
                if captcha_input.is_visible(timeout=1000):
                    # Focus and press Enter on captcha field
                    captcha_input.focus()
                    page.keyboard.press("Enter")
                    logger.info(f"[W{worker_id}] [SUBMIT {attempt}] Pressed Enter on captcha field")
                    
                    # Wait briefly for response
                    try:
                        page.wait_for_load_state("networkidle", timeout=3000)
                    except:
                        pass
                    
                    # Check result immediately
                    if self._check_submission_success(page, worker_id):
                        return True
                    
                    # Check if still on form (silent reject)
                    if self._is_on_form_page(page):
                        logger.warning(f"[W{worker_id}] [SUBMIT {attempt}] Still on form - trying click...")
                
                # Method 2: Click submit button directly
                submit_selectors = [
                    "#appointment_newAppointmentForm_appointment_addAppointment",
                    "input[name='action:appointment_addAppointment']",
                    "input[value='Submit']",
                    "input[type='submit'][value='Submit']",
                ]
                
                for selector in submit_selectors:
                    try:
                        btn = page.locator(selector).first
                        if btn.is_visible(timeout=500):
                            btn.click(timeout=2000)
                            logger.info(f"[W{worker_id}] [SUBMIT {attempt}] Clicked: {selector}")
                            
                            try:
                                page.wait_for_load_state("networkidle", timeout=3000)
                            except:
                                pass
                            
                            if self._check_submission_success(page, worker_id):
                                return True
                            break
                    except:
                        continue
                
                # Method 3: JavaScript native submit (FACT-BASED TARGETING)
                try:
                    # Submit SPECIFIC form ID verified from RK-Termin form.html
                    logger.info(f"[W{worker_id}] [SUBMIT {attempt}] JS form.submit() targeting 'appointment_newAppointmentForm'...")
                    page.evaluate("""
                        const form = document.getElementById('appointment_newAppointmentForm');
                        if(form) {
                            // FACT: The HTML checkKey function explicitly sets this action
                            form.action = "extern/appointment_addAppointment.do"; 
                            form.submit();
                        } else {
                            // Fallback to name if ID fails
                            document.getElementsByName('appointment_newAppointmentForm')[0]?.submit();
                        }
                    """)
                    
                    try:
                        page.wait_for_load_state("networkidle", timeout=3000)
                    except:
                        pass
                    
                    if self._check_submission_success(page, worker_id):
                        return True
                except Exception as e:
                    logger.warning(f"[W{worker_id}] [SUBMIT {attempt}] JS Submit error: {e}")
                
                # Check for captcha errors (need new captcha)
                content = page.content().lower()
                if "incorrect" in content or "wrong" in content or "falsch" in content:
                    logger.warning(f"[W{worker_id}] [SUBMIT {attempt}] Captcha rejected - re-solving...")
                    
                    # Try to solve new captcha
                    has_captcha, _ = self.solver.safe_captcha_check(page, f"RESUBMIT_{attempt}")
                    if has_captcha:
                        success, code, status = self.solver.solve_from_page(page, f"RESUBMIT_{attempt}")
                        if success:
                            logger.info(f"[W{worker_id}] New captcha solved: '{code}'")
                            self.global_stats.captchas_solved += 1
                            continue
                    
                    self.global_stats.captchas_failed += 1
                    continue
                
                # If we're back on month/day page, slot was taken
                if "appointment_showMonth" in page.url or "appointment_showDay" in page.url:
                    logger.error(f"[W{worker_id}] [SUBMIT {attempt}] Slot lost - redirected to calendar")
                    return False
                
            except Exception as e:
                logger.error(f"[W{worker_id}] [SUBMIT {attempt}] Error: {e}")
                continue
        
        logger.warning(f"[W{worker_id}] Max submit attempts ({max_attempts}) reached")
        return False
    
    def _check_submission_success(self, page: Page, worker_id: int) -> bool:
        """Check if submission was successful"""
        try:
            content = page.content().lower()
            
            # Success indicators
            success_terms = [
                "appointment number",
                "confirmation",
                "successfully",
                "termin wurde gebucht",
                "ihre buchung",
                "booking confirmed",
                "appointment confirmed",
            ]
            
            for term in success_terms:
                if term in content:
                    logger.critical(f"[W{worker_id}] *** SUCCESS! Found: '{term}' ***")
                    
                    # Save evidence
                    self.debug_manager.save_critical_screenshot(page, "SUCCESS_FINAL", worker_id)
                    self.debug_manager.save_debug_html(page, "SUCCESS_FINAL", worker_id)
                    
                    # Notify
                    try:
                        send_success_notification(self.session_id, worker_id, None)
                    except:
                        pass
                    
                    with self.lock:
                        self.global_stats.success = True
                    
                    self.stop_event.set()
                    return True
                    
            # If we get here, no success terms found
            logger.warning(f"[W{worker_id}] Submission verification failed - checking for specific errors...")
            
            # Save snapshot of what we see
            try:
                self.debug_manager.save_debug_html(page, "submission_failed_snapshot", worker_id)
                self.debug_manager.save_critical_screenshot(page, "submission_failed_snapshot", worker_id)
            except:
                pass
                
            return False
        except Exception as e:
            logger.error(f"[W{worker_id}] Error in success check: {e}")
            return False
    
    def _is_on_form_page(self, page: Page) -> bool:
        """Check if still on form page (silent reject)"""
        try:
            return page.locator("input[name='lastname']").count() > 0
        except:
            return False
    
    # ==================== Scout Behavior ====================
    
    def _scout_behavior(self, page: Page, session: SessionState, worker_logger):
        """
        Scout behavior: Fast discovery, signals attackers
        Does NOT book - purely for finding slots
        """
        worker_id = session.worker_id
        
        try:
            # Get month URLs to scan
            month_urls = self.generate_month_urls()
            
            for url in month_urls:
                if self.stop_event.is_set():
                    return
                
                # Navigate to month page
                try:
                    page.goto(url, timeout=20000, wait_until="domcontentloaded")
                    session.current_url = url
                    session.touch()
                    
                    with self.lock:
                        self.global_stats.pages_loaded += 1
                        self.global_stats.months_scanned += 1
                        self.global_stats.scans += 1
                        
                except Exception as e:
                    worker_logger.warning(f"Navigation error: {e}")
                    with self.lock:
                        self.global_stats.navigation_errors += 1
                    continue
                
                # Check session health
                if not self.validate_session_health(page, session, "SCOUT_MONTH"):
                    return
                
                # Handle captcha if present
                has_captcha, _ = self.solver.safe_captcha_check(page, "SCOUT_MONTH")
                if has_captcha:
                    success, code, captcha_status = self.solver.solve_from_page(page, "SCOUT_MONTH")
                    if success and code:
                        self.solver.submit_captcha(page, "enter")
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except:
                            pass
                        
                        with self.lock:
                            self.global_stats.captchas_solved += 1
                        session.mark_captcha_solved()
                    else:
                        with self.lock:
                            self.global_stats.captchas_failed += 1
                        continue
                
                # Check for "no appointments" message
                content = page.content().lower()
                if "no appointments" in content or "keine termine" in content:
                    continue
                
                # Look for available days
                day_links = page.locator("a.arrow[href*='appointment_showDay']").all()
                
                if day_links:
                    num_days = len(day_links)
                    worker_logger.critical(f"🔥 SCOUT FOUND {num_days} DAYS!")
                    
                    with self.lock:
                        self.global_stats.days_found += num_days
                    
                    # Get the first day URL
                    first_href = day_links[0].get_attribute("href")
                    if first_href:
                        # Build full URL for attackers
                        base_domain = self.base_url.split("/extern")[0]
                        self.target_url = f"{base_domain}/{first_href}"
                        
                        # Signal attackers!
                        worker_logger.critical(f"🟢 SIGNALING ATTACKERS! URL: {self.target_url[:50]}...")
                        send_alert(
                            f"🟢 <b>SCOUT: SLOTS DETECTED!</b>\n"
                            f"📅 Days found: {num_days}\n"
                            f"⏰ Attackers engaging..."
                        )
                        
                        self.incident_manager.create_incident(
                            session.session_id, IncidentType.SLOT_DETECTED,
                            IncidentSeverity.INFO,
                            f"Found {num_days} available days"
                        )
                        
                        # Signal the event
                        self.slot_event.set()
                        
                        # Scout doesn't proceed to booking - let attackers handle it
                        return
                
        except Exception as e:
            worker_logger.error(f"Scout behavior error: {e}")
            session.increment_failure(str(e))
    
    # ==================== Attacker Behavior ====================
    
    def _attacker_behavior(self, page: Page, session: SessionState, worker_logger):
        """
        Attacker behavior: Wait for scout signal or scan independently
        Executes booking when slots are found
        """
        worker_id = session.worker_id
        
        try:
            # In attack mode, scan independently
            mode = self.get_mode()
            
            # If not attack mode and no signal, do light scanning
            if mode not in ["ATTACK", "PRE_ATTACK"] and not self.slot_event.is_set():
                # Light patrol - don't overwhelm server
                time.sleep(random.uniform(2, 5))
                
                # Check for scout signal
                if self.slot_event.wait(timeout=1.0):
                    worker_logger.info("📡 Received scout signal!")
            
            # If signal received and we have a target URL, go directly there
            if self.slot_event.is_set() and self.target_url:
                worker_logger.info(f"🎯 Attacking target: {self.target_url[:50]}...")
                try:
                    page.goto(self.target_url, timeout=15000, wait_until="domcontentloaded")
                    session.touch()
                except Exception as e:
                    worker_logger.warning(f"Target navigation failed: {e}")
                    self.slot_event.clear()  # Clear and retry
                    return
            else:
                # Independent scanning
                month_urls = self.generate_month_urls()
                
                # Attackers scan fewer months to stay ready
                for url in month_urls[:3]:
                    if self.stop_event.is_set():
                        return
                    
                    try:
                        page.goto(url, timeout=20000, wait_until="domcontentloaded")
                        session.current_url = url
                        session.touch()
                        
                        with self.lock:
                            self.global_stats.pages_loaded += 1
                            self.global_stats.scans += 1
                            
                    except Exception as e:
                        worker_logger.warning(f"Navigation error: {e}")
                        continue
                    
                    # Handle captcha
                    has_captcha, _ = self.solver.safe_captcha_check(page, f"ATK_MONTH")
                    if has_captcha:
                        success, code, captcha_status = self.solver.solve_from_page(page, f"ATK_MONTH")
                        if success and code:
                            self.solver.submit_captcha(page, "enter")
                            try:
                                page.wait_for_load_state("domcontentloaded", timeout=4000)
                            except:
                                pass
                            
                            with self.lock:
                                self.global_stats.captchas_solved += 1
                            session.mark_captcha_solved()
                        else:
                            continue
                    
                    # Look for days
                    day_links = page.locator("a.arrow[href*='appointment_showDay']").all()
                    if day_links:
                        break
                else:
                    # No days found in any month
                    return
            
            # Check session health
            if not self.validate_session_health(page, session, "ATK_DAY"):
                return
            
            # Click on first available day (or we're already there from target_url)
            day_links = page.locator("a.arrow[href*='appointment_showDay']").all()
            if day_links:
                target_day = random.choice(day_links)
                href = target_day.get_attribute("href")
                
                worker_logger.info(f"📅 Clicking day: {href[:40] if href else 'N/A'}...")
                
                try:
                    target_day.click(timeout=5000)
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception as e:
                    # Fallback: direct navigation
                    if href:
                        base_domain = self.base_url.split("/extern")[0]
                        page.goto(f"{base_domain}/{href}", timeout=15000)
                
                session.reset_for_new_flow()
            
            # Handle day captcha
            has_captcha, _ = self.solver.safe_captcha_check(page, "ATK_DAY")
            if has_captcha:
                success, code, captcha_status = self.solver.solve_from_page(page, "ATK_DAY")
                if success and code:
                    self.solver.submit_captcha(page, "enter")
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=4000)
                    except:
                        pass
                    session.mark_captcha_solved()
                else:
                    return
            
            # Look for time slots
            time_links = page.locator("a.arrow[href*='appointment_showForm']").all()
            
            if time_links:
                with self.lock:
                    self.global_stats.slots_found += len(time_links)
                
                worker_logger.critical(f"⏰ [W{worker_id}] {len(time_links)} TIME SLOTS FOUND!")
                
                # Click first time slot
                target_time = random.choice(time_links)
                href = target_time.get_attribute("href")
                
                try:
                    target_time.click(timeout=5000)
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception as e:
                    if href:
                        base_domain = self.base_url.split("/extern")[0]
                        page.goto(f"{base_domain}/{href}", timeout=15000)
                
                session.reset_for_new_flow()
                
                # Handle form captcha
                has_captcha, _ = self.solver.safe_captcha_check(page, "ATK_FORM")
                if has_captcha:
                    success, code, captcha_status = self.solver.solve_from_page(page, "ATK_FORM")
                    if success and code:
                        self.solver.submit_captcha(page, "enter")
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=4000)
                        except:
                            pass
                        session.mark_captcha_solved()
                    else:
                        return
                
                # Validate we're on the form
                if not self.validate_session_health(page, session, "FORM"):
                    return
                
                # Check if form is visible
                if page.locator("input[name='lastname']").count() == 0:
                    worker_logger.warning("Form not found after navigation")
                    return
                
                # FILL AND SUBMIT FORM!
                self.incident_manager.create_incident(
                    session.session_id, IncidentType.BOOKING_ATTEMPT,
                    IncidentSeverity.INFO,
                    "Attempting to book appointment"
                )
                
                if self.fill_booking_form(page, session):
                    if self.submit_form(page, session):
                        # SUCCESS!
                        return
                
        except Exception as e:
            worker_logger.error(f"Attacker behavior error: {e}")
            session.increment_failure(str(e))
    
    # ==================== Worker Thread ====================
    
    def session_worker(self, browser: Browser, worker_id: int):
        """
        Worker thread for one browser session
        Implements Scout or Attacker behavior based on worker_id
        """
        worker_logger = logging.getLogger(f"EliteSniperV2.W{worker_id}")
        
        try:
            # Get proxy for this worker
            proxy = self.proxies[worker_id - 1] if len(self.proxies) >= worker_id else None
            
            # Create initial context
            context, page, session = self.create_context(browser, worker_id, proxy)
            
            role = "SCOUT" if worker_id == 1 else "ATTACKER"
            worker_logger.info(f"👤 Worker started - Role: {role}")
            
            cycle = 0
            last_status_update = 0
            
            while not self.stop_event.is_set():
                cycle += 1
                
                try:
                    current_time = time.time()
                    mode = self.get_mode()
                    
                    # Periodic status update (every 5 minutes)
                    if current_time - last_status_update > 300:
                        send_status_update(
                            self.session_id,
                            f"Cycle {cycle}",
                            self.global_stats.to_dict(),
                            mode
                        )
                        last_status_update = current_time
                    
                    # Pre-attack reset - fresh session before attack window
                    if self.is_pre_attack() and not session.pre_attack_reset_done:
                        worker_logger.warning("⚙️ PRE-ATTACK: Fresh session reset")
                        try:
                            context.close()
                        except:
                            pass
                        context, page, session = self.create_context(browser, worker_id, proxy)
                        session.pre_attack_reset_done = True
                        
                        # Pre-solve captcha while waiting
                        for _ in range(3):
                            try:
                                page.goto(self.base_url, timeout=30000, wait_until="domcontentloaded")
                                self.solver.pre_solve(page, "PRE_ATTACK")
                                break
                            except:
                                time.sleep(2)
                        
                        continue
                    
                    # Check session health
                    if session.should_terminate() or session.is_expired():
                        worker_logger.warning("💀 Session unhealthy - Rebirth!")
                        try:
                            context.close()
                        except:
                            pass
                        context, page, session = self.create_context(browser, worker_id, proxy)
                        continue
                    
                    # Route to appropriate behavior based on role
                    if session.role == SessionRole.SCOUT:
                        self._scout_behavior(page, session, worker_logger)
                    else:
                        self._attacker_behavior(page, session, worker_logger)
                    
                    # Reset slot event after processing (attackers will re-wait)
                    if session.role == SessionRole.ATTACKER and self.slot_event.is_set():
                        # Small delay before clearing to let other attackers see it
                        time.sleep(0.5)
                        self.slot_event.clear()
                
                except Exception as e:
                    worker_logger.error(f"❌ Session crashed: {e}")
                    # In single session mode, we might want to restart?
                    # For V3/V2.0.0 persistence, _run_single_session has its own loop
                    # If it returns, it's either success or fatal stop
                    if self.global_stats.success:
                        worker_logger.info("🏆 Application Success via Session")
                        break
                    time.sleep(5) # Cooldown before restart attempt
                    
                    # Hard reset context on crash
                    try:
                        context.close()
                    except:
                        pass
                    # Recreate context loop continues naturally
                    continue
        
        except Exception as e:
            worker_logger.error(f"[FATAL] Worker error: {e}", exc_info=True)
        
        finally:
            try:
                context.close()
            except:
                pass
            worker_logger.info("[END] Worker terminated")
    
    # ==================== Single Session Mode ====================
    
    def _run_single_session(self, browser: Browser, worker_id: int = 1):
        """
        Run a single robust blocking session
        """
        worker_logger = logging.getLogger(f"Worker-{worker_id}")
        
        # Create Browser Context
        try:
            proxy = Config.PROXIES[worker_id % len(Config.PROXIES)] if Config.PROXIES else None
        except: proxy = None
        
        context, page, session = self.create_context(self.browser, worker_id, proxy)
        self.current_page = page # Expose for C2 Screenshot
        
        session.role = SessionRole.SCOUT
        
        worker_logger.info(f"[START] Robust Single Session Mode")
        
        try:
            max_cycles = 1000  # Persistent runner
            
            for cycle in range(max_cycles):
                if self.stop_event.is_set(): break
                
                # PAUSE CHECK (C2)
                if self.paused.is_set():
                    worker_logger.info("⏸️ Session PAUSED by C2")
                    while self.paused.is_set() and not self.stop_event.is_set():
                        time.sleep(1)
                    worker_logger.info("▶️ Session RESUMED")
                
                worker_logger.info(f"🔄 [CYCLE {cycle+1}] Scanning...")
                
                try:
                    # 1. GENERATE TARGETS
                    month_urls = self.generate_month_urls()
                    
                    # 2. SCAN PHASE
                    for url in month_urls:
                        if self.stop_event.is_set(): break
                        
                        # Process Month (Returns True if slot found and booked specific to this flow)
                        if self._process_month_page(page, session, url, worker_logger):
                            return  # SUCCESS or CRITICAL STOP
                        
                        # Circuit Breaker: Fail Fast on Network Issues
                        if getattr(session, 'consecutive_network_failures', 0) >= 2:
                             worker_logger.warning("⚡ Circuit Breaker Triggered: Network Unstable. Resetting session...")
                             break
                        
                        # POISON CHECK: If session died (Gate Detected), reset IMMEDIATELY (Don't sleep!)
                        if session.health == SessionHealth.POISONED:
                             worker_logger.warning("☠️ Session POISONED (Gate Detected). Triggering IMMEDIATE Rebirth!")
                             break 

                        # Small delay between months
                        time.sleep(random.uniform(1, 2))
                    
                    # 3. SLEEP PHASE
                    # Only sleep if session is HEALTHY
                    if session.health != SessionHealth.POISONED:
                        sleep_time = self.get_sleep_interval()
                        worker_logger.info(f"💤 Sleeping {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                    else:
                        worker_logger.info("⏩ Skipping sleep due to POISONED session.")
                    
                    # 4. MAINTENANCE PHASE (GARBAGE COLLECTION)
                    
                    # DYNAMIC SESSION AGE: 
                    # During Golden Hour (Attack), we allow session to live longer (up to 45 mins)
                    # to support "Persistent Session" strategy.
                    current_max_age = 2700 if self.is_attack_time() else Config.SESSION_MAX_AGE
                    
                    if session.age() > current_max_age or getattr(session, 'consecutive_network_failures', 0) >= 2 or session.health == SessionHealth.POISONED:
                        if session.health == SessionHealth.POISONED:
                             reason = "POISONED/GATE"
                        else:
                             reason = "Age" if session.age() > current_max_age else "Network Instability"
                        
                        worker_logger.info(f"♻️ Session Reset triggered ({reason}) - Recreating Context...")
                        
                        # STRICT CLEANUP BEFORE REBIRTH
                        try: 
                            page.close()
                            context.close()
                        except: pass
                        
                        # FRESH CONTEXT
                        context, page, session = self.create_context(browser, worker_id, proxy)
                        session.role = SessionRole.SCOUT
                        
                except Exception as cycle_error:
                    worker_logger.error(f"⚠️ Cycle error: {cycle_error}")
                    # Force reset on error to prevent zombie state
                    try:
                        page.close()
                        context.close()
                    except: pass
                    context, page, session = self.create_context(browser, worker_id, proxy)
                    session.role = SessionRole.SCOUT

        except Exception as e:
            worker_logger.error(f"❌ Critical Session Error: {e}", exc_info=True)
        finally:
            # ULTIMATE CLEANUP
            worker_logger.info("🧹 Final cleanup of single session...")
            try: 
                page.close()
                context.close()
            except: pass

    def _analyze_page_state(self, page: Page, logger) -> str:
        """
        Analyzes the current page HTML to determine the exact state.
        Based on actual website HTML structure.
        
        PRIORITY ORDER (most important first):
        1. SLOTS_FOUND - Success! Days available
        2. EMPTY_CALENDAR - No appointments this month
        3. WRONG_CODE - Captcha was wrong
        4. CAPTCHA - Need to solve captcha
        5. UNKNOWN - Fallback
        
        Returns: 'CAPTCHA', 'WRONG_CODE', 'EMPTY_CALENDAR', 'SLOTS_FOUND', 'UNKNOWN'
        """
        try:
            # Wait for page to stabilize first
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except:
                pass
            
            time.sleep(0.3)  # Small buffer for dynamic content
            
            content = page.content().lower()
            
            # 1. SLOTS_FOUND - Check first! This is SUCCESS
            # HTML: <a href="...appointment_showDay..." class="arrow">Appointments are available</a>
            try:
                slot_count = page.locator("a[href*='appointment_showDay']").count()
                if slot_count > 0:
                    logger.info(f"🎯 Detected {slot_count} available day(s)!")
                    return "SLOTS_FOUND"
            except:
                pass
            
            # 2. EMPTY_CALENDAR - Check second
            # HTML: "Unfortunately, there are no appointments available"
            if "unfortunately, there are no appointments available" in content:
                return "EMPTY_CALENDAR"
            if "keine termine" in content:
                return "EMPTY_CALENDAR"
            if "no appointments" in content and "appointment_showDay" not in content:
                return "EMPTY_CALENDAR"
            
            # 3. WRONG_CODE - Check third
            # HTML: <div id="message" class="global-error"><p>The entered text was wrong</p></div>
            if "entered text was wrong" in content:
                return "WRONG_CODE"
            try:
                if page.locator("div.global-error").is_visible(timeout=300):
                    return "WRONG_CODE"
            except:
                pass
            
            # 4. CAPTCHA - Check if we're on captcha page
            # HTML: <form id="appointment_captcha_month">
            try:
                if page.locator("#appointment_captcha_month").is_visible(timeout=300):
                    return "CAPTCHA"
            except:
                pass
            
            # Fallback: check for captcha input
            try:
                if page.locator("input[name='captchaText']").is_visible(timeout=300):
                    return "CAPTCHA"
            except:
                pass

            return "UNKNOWN"
            
        except Exception as e:
            logger.warning(f"⚠️ Page state analysis error: {e}")
            return "UNKNOWN"

    def check_session_health(self, page: Page, session: SessionState, logger) -> bool:
        """
        [FACT-BASED] Check if session is alive using selectors from sniberhtmel.
        Returns False if we are kicked out to a gate/captcha page.
        """
        try:
            # 1. Check for Month Captcha Page (The Gate)
            # 1. Check for Month Captcha Page (The Gate)
            # Source: debug_CRITICAL_slot_disappeared...html -> <form id="appointment_captcha_month">
            if page.locator("#appointment_captcha_month").is_visible(timeout=500):
                 # [CRITICAL UPDATE] This is NOT a dead session. It's a Gated Session.
                 # Let _process_month_page handle solving it.
                 # Do NOT return False here.
                 pass

            # 2. Check for Booking Form Captcha (The Inner Gate)
            # Source: debug_verification_step...html -> #appointment_newAppointmentForm_captchaText
            if page.locator("#appointment_newAppointmentForm_captchaText").is_visible(timeout=500):
                 # This is actually GOOD if we are booking, but bad if we are scanning
                 # Context matters, but for safety, if we see this unexpected, it might be a flow error.
                 # For now, we assume if we are scanning months, this shouldn't be here.
                 pass

            # 3. Check for Global Error
            # Source: common error pages -> div.global-error
            if page.locator("div.global-error").is_visible(timeout=500):
                logger.warning("⚠️ Session Check: Detected Global Error")
                # Don't kill immediately, but warn.
                
            return True
            
        except:
            return True # Assume alive on error to avoid false positives

    def _process_month_page(self, page: Page, session: SessionState, url: str, logger) -> bool:
        """
        Smart Month Page Handler with HTML-Based State Analysis
        """
        try:
            # A. ROBUST NAVIGATION
            # ... (Navigation logic remains similar) ...
            
            # [Added] Pre-check Session Health
            if not self.check_session_health(page, session, logger):
                return False

            # ... (Navigation code) ...
            logger.info(f"🌐 Navigating to: {url[:60]}...")
            max_nav_retries = 2
            for nav_attempt in range(max_nav_retries):
                try:
                    page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    break
                except Exception as nav_e:
                    logger.warning(f"⚠️ Nav failed ({nav_attempt+1}/{max_nav_retries}): {nav_e}")
                    if nav_attempt < max_nav_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return False
            
            session.touch()
            self.global_stats.pages_loaded += 1
            
            # [Added] Post-Navigation Health Check
            if not self.check_session_health(page, session, logger):
                logger.warning("⚠️ Session died after navigation.")
                return False

            # B. SMART CAPTCHA SOLVING LOOP
            max_attempts = 5  # increased from 3 to 5 (USER REQUIREMENT)
            consecutive_failures = 0
            
            for attempt in range(max_attempts):
                # Analyze State
                state = self._analyze_page_state(page, logger)
                
                if state == "SLOTS_FOUND":
                    # ... (Success logic) ...
                    logger.critical("🎯 SUCCESS: Slots detected!")
                    # ... (Click logic) ...
                    day_links = page.locator("a[href*='appointment_showDay']").all()
                    if day_links:
                         # ...
                         target_day = day_links[0]
                         day_href = target_day.get_attribute("href")
                         if day_href:
                            base_domain = self.base_url.split("/extern")[0]
                            day_url = f"{base_domain}/{day_href}" if not day_href.startswith("http") else day_href
                            return self._process_day_page(page, session, day_url, logger)
                    return False

                elif state == "EMPTY_CALENDAR":
                    logger.info("📅 Calendar Empty.")
                    return False

                elif state == "WRONG_CODE" or state == "CAPTCHA":
                    # Handle both WRONG_CODE and CAPTCHA states
                    is_wrong_code = (state == "WRONG_CODE")
                    
                    if is_wrong_code:
                        logger.warning(f"❌ Server said: 'Wrong captcha'. Retrying... [{attempt+1}/{max_attempts}]")
                        consecutive_failures += 1
                        self.global_stats.captchas_failed += 1
                        
                        if consecutive_failures >= 5:
                             logger.critical("🛑 [ANTI-BAN] 5 Consecutive WRONG CODES! Sleeping 2 minutes...")
                             time.sleep(120)
                             return False
                    else:
                        logger.info(f"🧩 Captcha detected (Attempt {attempt+1}/{max_attempts})")

                    # Solve
                    success, code, status = self.solver.solve_from_page(page, f"MONTH_{attempt}")
                    
                    if not success:
                        if not is_wrong_code: # Don't double count failure if it was already wrong code
                            consecutive_failures += 1
                        
                        logger.warning(f"⚠️ Solve failed. Failures: {consecutive_failures}")
                        
                        if consecutive_failures >= 5:
                             logger.critical("🛑 [ANTI-BAN] 5 Consecutive Failures! Sleeping 2 minutes...")
                             time.sleep(120)
                             return False # Exist session to force rebirth
                        
                        self._refresh_captcha(page)
                        continue
                        
                    # Submit
                    self.solver.submit_captcha(page, code)
                    # Loop continues to check result
                    
                else:
                    # Unknown state
                    return False
            
            return False

        except Exception as e:
            logger.error(f"Error processing month: {e}")
            return False


    def _process_day_page(self, page: Page, session: SessionState, url: str, logger) -> bool:
        """Handle Day Page: Scan Slots -> Navigate Form"""
        try:
            logger.info("📆 Analyzing Day Page...")
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            
            slot_links = page.locator("a.arrow[href*='appointment_showForm']").all()
            if not slot_links:
                logger.info("⚠️ Days shown but no slots active.")
                return False
                
            logger.critical(f"⏰ {len(slot_links)} SLOTS FOUND! ENGAGING!")
            self.global_stats.slots_found += len(slot_links)
            
            # Pick first slot
            target_slot = slot_links[0]
            slot_href = target_slot.get_attribute("href")
            
            if slot_href:
                base_domain = self.base_url.split("/extern")[0]
                form_url = f"{base_domain}/{slot_href}" if not slot_href.startswith("http") else slot_href
                return self._process_booking_form(page, session, form_url, logger)
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Day processing error: {e}")
            return False

    def _process_booking_form(self, page: Page, session: SessionState, url: str, logger) -> bool:
        """Handle Booking: Fill -> Captcha -> Smart Submit"""
        try:
            logger.info("📝 Entering Booking Phase...")
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            
            # 1. FAST FILL (Humanized)
            if not self._fill_booking_form(page, session, logger):
                return False
                
            # 2. CAPTCHA SOLVE
            captcha_code = None
            has_captcha, _ = self.solver.safe_captcha_check(page, "FORM")
            if has_captcha:
                success, code, _ = self.solver.solve_form_captcha_with_retry(page, "BOOKING")
                if not success:
                    logger.warning("❌ Booking Captcha Failed")
                    return False
                captcha_code = code
            
            # 3. DRY RUN CHECK
            if Config.DRY_RUN:
                logger.critical("🛑 DRY RUN TRIGGERED - NOT SUBMITTING!")
                self.debug_manager.save_critical_screenshot(page, "DRY_RUN_SUCCESS", session.worker_id)
                time.sleep(5) # Let user see it
                return True # Treat as success
                
            # 4. SMART SUBMIT
            return self._submit_form(page, session, logger, initial_code=captcha_code)
            
        except Exception as e:
            logger.error(f"❌ Booking error: {e}")
            return False
    
    # ==================== Main Entry Point ====================
    
    def run(self) -> bool:
        """
        Main execution entry point
        
        Returns:
            True if booking successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info(f"[ELITE SNIPER V{self.VERSION}] - STARTING EXECUTION")
        # Single session mode - multi-session architecture preserved for future
        logger.info("[MODE] Single Session (Multi-session ready for expansion)")
        logger.info(f"[ATTACK TIME] {Config.ATTACK_HOUR}:00 AM {Config.TIMEZONE}")
        logger.info(f"[CURRENT TIME] Aden: {self.get_current_time_aden().strftime('%H:%M:%S')}")
        logger.info("=" * 70)
        
        try:
            # Send startup notification
            send_alert(
                f"[Elite Sniper v{self.VERSION} Started]\n"
                f"Session: {self.session_id}\n"
                f"Mode: Single Session\n"
                f"Attack: {Config.ATTACK_HOUR}:00 AM Aden\n"
                f"NTP Offset: {self.ntp_sync.offset:.4f}s"
            )
            
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(
                    headless=Config.HEADLESS,
                    args=Config.BROWSER_ARGS,
                    timeout=60000
                )
                
                logger.info("[BROWSER] Launched successfully")
                
                # ========================================
                # SINGLE SESSION MODE (Direct execution)
                # Architecture preserved for 3 sessions later
                # ========================================
                self.browser = browser  # Assign to instance for shared access
                worker_id = 1  # Scout role for single session
                
                try:
                    # Run single session directly (no threads)
                    self._run_single_session(browser, worker_id=worker_id)
                except Exception as e:
                    logger.error(f"[SESSION ERROR] {e}")
                
                # Stop NTP sync
                self.ntp_sync.stop_background_sync()
                
                # Cleanup
                browser.close()
                
                # Save final stats
                final_stats = self.global_stats.to_dict()
                self.debug_manager.save_stats(final_stats, "final_stats.json")
                self.debug_manager.create_session_report(final_stats)
                
                if self.global_stats.success:
                    self._handle_success()
                    return True
                else:
                    self._handle_completion()
                    return False
                
        except KeyboardInterrupt:
            logger.info("\n[STOP] Manual stop requested")
            self.stop_event.set()
            self.ntp_sync.stop_background_sync()
            send_alert("⏸️ Elite Sniper stopped manually")
            return False
            
        except Exception as e:
            logger.error(f"💀 Critical error: {e}", exc_info=True)
            send_alert(f"🚨 Critical error: {str(e)[:200]}")
            return False
            
        finally:
            self.cleanup()
    
    def _scout_behavior(self, page: Page, session: SessionState, worker_logger):
        """
        Scout behavior: Fast discovery without booking
        Scans months for available days and signals Attackers
        """
        worker_logger.info("🔍 Scout scanning...")
        
        try:
            month_urls = self.generate_month_urls()
            
            for url in month_urls[:4]:  # First 4 priority months
                if self.stop_event.is_set():
                    return
                
                try:
                    # Navigate to month page
                    page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    session.pages_loaded += 1
                    self.global_stats.pages_loaded += 1
                    
                    # Save debug HTML
                    self.debug_manager.save_debug_html(page, "scout_month", session.worker_id)
                    
                    # Handle captcha if present
                    success, code, captcha_status = self.solver.solve_from_page(page, "SCOUT_MONTH")
                    if success and code:
                        session.mark_captcha_solved()
                        self.global_stats.captchas_solved += 1
                        self.solver.submit_captcha(page)
                        time.sleep(1)
                    
                    # Check for available days
                    day_selectors = [
                        "a.arrow[href*='appointment_showDay']",
                        "td.buchbar a",
                        "a[href*='showDay']"
                    ]
                    
                    for selector in day_selectors:
                        try:
                            days = page.locator(selector).all()
                            if days:
                                worker_logger.critical(f"🟢 SCOUT FOUND {len(days)} DAYS!")
                                self.global_stats.days_found += len(days)
                                
                                # Signal attackers
                                with self.lock:
                                    self.target_url = url
                                
                                self.slot_event.set()
                                send_alert(f"🎯 Days found! Signaling attackers. URL: {url[:60]}...")
                                
                                time.sleep(2)  # Give attackers time to react
                                break
                        except:
                            continue
                    
                except Exception as e:
                    worker_logger.warning(f"⚠️ Month scan error: {e}")
                    session.increment_failure(str(e))
                    continue
            
            self.global_stats.scans += 1
            
        except Exception as e:
            worker_logger.error(f"❌ Scout behavior error: {e}")
            session.increment_failure(str(e))
    
    def _attacker_behavior(self, page: Page, session: SessionState, worker_logger):
        """
        Attacker behavior: Wait for Scout signal, then execute booking
        Pre-positioned with solved captcha for instant action
        """
        # If no signal yet, stay ready on first month page
        if not self.slot_event.is_set():
            try:
                if session.pages_loaded == 0:
                    # Get positioned on first month
                    month_urls = self.generate_month_urls()
                    if month_urls:
                        page.goto(month_urls[0], wait_until="domcontentloaded", timeout=20000)
                        session.pages_loaded += 1
                        
                        # Pre-solve captcha
                        success, code, captcha_status = self.solver.solve_from_page(page, "ATTACKER_READY")
                        if success and code:
                            session.mark_captcha_solved()
                            self.global_stats.captchas_solved += 1
                            self.solver.submit_captcha(page)
                            worker_logger.info("✅ Attacker ready with pre-solved captcha")
                
                # Wait for signal with timeout
                self.slot_event.wait(timeout=5)
                return
                
            except Exception as e:
                worker_logger.warning(f"⚠️ Attacker standby error: {e}")
                return
        
        # Got signal - ATTACK!
        worker_logger.warning("🔥 ATTACKER ENGAGING!")
        
        try:
            target = self.target_url
            if not target:
                return
            
            # Navigate to target month
            page.goto(target, wait_until="domcontentloaded", timeout=15000)
            
            # Handle captcha
            success, _ = self.solver.solve_from_page(page, "ATTACK_MONTH")
            if success:
                self.solver.submit_captcha(page)
                time.sleep(0.5)
            
            # Find and click day
            day_links = page.locator("a.arrow[href*='appointment_showDay']").all()
            if not day_links:
                day_links = page.locator("a[href*='showDay']").all()
            
            if not day_links:
                worker_logger.warning("⚠️ No days found at target")
                return
            
            # Click first available day
            target_day = day_links[0]
            day_href = target_day.get_attribute("href")
            worker_logger.info(f"📅 Clicking day: {day_href}")
            target_day.click()
            
            time.sleep(1)
            
            # Handle day page captcha
            success, _ = self.solver.solve_from_page(page, "ATTACK_DAY")
            if success:
                self.solver.submit_captcha(page)
                time.sleep(0.5)
            
            # Find and click time slot
            time_links = page.locator("a.arrow[href*='appointment_showForm']").all()
            if not time_links:
                time_links = page.locator("a[href*='showForm']").all()
            
            if not time_links:
                worker_logger.warning("⚠️ No time slots found")
                self.global_stats.slots_found = 0
                return
            
            self.global_stats.slots_found += len(time_links)
            
            # Click first available time
            target_time = time_links[0]
            time_href = target_time.get_attribute("href")
            worker_logger.info(f"⏰ Clicking time: {time_href}")
            target_time.click()
            
            time.sleep(1)
            
            # Handle form page captcha
            success, _ = self.solver.solve_from_page(page, "ATTACK_FORM")
            if success:
                self.solver.submit_captcha(page)
                time.sleep(0.5)
            
            # Save form page for debugging
            self.debug_manager.save_debug_html(page, "form_page", session.worker_id)
            
            # Fill form
            if self._fill_booking_form(page, session, worker_logger):
                # Submit form
                if self._submit_form(page, session, worker_logger):
                    # SUCCESS!
                    self.global_stats.success = True
                    self.stop_event.set()
                    return
            
        except Exception as e:
            worker_logger.error(f"❌ Attacker error: {e}")
            session.increment_failure(str(e))
    
    def _fill_booking_form(self, page: Page, session: SessionState, worker_logger) -> bool:
        """
        [PATCHED] Fill booking form using HUMAN TYPING to trigger validation scripts.
        Avoids JS injection unless absolutely necessary.
        """
        try:
            worker_logger.info("📝 Filling form (Human Mode)...")
            
            # تعريف الحقول والقيم
            fields = [
                ("input[name='lastname']", Config.LAST_NAME),
                ("input[name='firstname']", Config.FIRST_NAME),
                ("input[name='email']", Config.EMAIL),
                ("input[name='emailrepeat']", Config.EMAIL),
                ("input[name='emailRepeat']", Config.EMAIL), # Case sensitive check
                # الحقول الديناميكية (جواز السفر والهاتف)
                ("input[name='fields[0].content']", Config.PASSPORT),
                ("input[name='fields[1].content']", Config.PHONE.replace("+", "00").strip())
            ]
            
            for selector, value in fields:
                try:
                    # التحقق من وجود الحقل
                    if page.locator(selector).count() > 0:
                        # 1. التركيز (Focus) - مهم جداً لتفعيل السكربتات
                        page.focus(selector)
                        # 2. مسح المحتوى القديم (إن وجد)
                        page.fill(selector, "")
                        # 3. الكتابة البشرية (Typing)
                        page.type(selector, value, delay=10) # تأخير بسيط جداً (10ms) للمحاكاة
                        # 4. الخروج من الحقل (Blur) لتثبيت القيمة
                        page.evaluate(f"document.querySelector(\"{selector}\").blur()")
                except Exception as e:
                    worker_logger.debug(f"Field fill error ({selector}): {e}")
                    continue

            # اختيار الفئة (Category) - استخدام Smart Targeting
            if not self.select_category_by_value(page):
                worker_logger.warning("Category selection failed via Smart Targeting - attempting fallback...")
                
                # Fallback: Try selecting by index 1
                try:
                    page.evaluate("""
                        const s = document.querySelector('select');
                        if(s) { s.selectedIndex = 1; s.dispatchEvent(new Event('change')); }
                    """)
                    worker_logger.info("Category selected via JS Fallback (Index 1)")
                except Exception as e:
                    worker_logger.error(f"Category selection fallback failed: {e}")

            self.global_stats.forms_filled += 1
            worker_logger.info("✅ Form filled (Humanized)")
            return True
            
        except Exception as e:
            worker_logger.error(f"❌ Form fill error: {e}")
            return False
    
    def _fast_inject(self, page: Page, selector: str, value: str) -> bool:
        """Fast DOM injection bypassing events"""
        try:
            page.evaluate(f"""
                const el = document.querySelector("{selector}");
                if(el) {{
                    el.value = "{value}";
                    el.dispatchEvent(new Event('input', {{bubbles: true}}));
                    el.dispatchEvent(new Event('change', {{bubbles: true}}));
                }}
            """)
            return True
        except:
            return False
    
    def _submit_form(self, page: Page, session: SessionState, worker_logger, initial_code: Optional[str] = None) -> bool:
        """
        [HUMAN-LIKE SUBMISSION] Smart submit with proper waiting and validation.
        Prevents race conditions by waiting for server response.
        Optimization: Accepts initial_code to skip redundant solving on first attempt.
        """
        max_attempts = 5  # Reduced from 15 (Quality over Quantity)
        worker_logger.info(f"🚀 STARTING SMART SUBMISSION SEQUENCE...")

        for attempt in range(1, max_attempts + 1):
            try:
                # 1. FIND CAPTCHA INPUT
                captcha_input = page.locator("input[name='captchaText']").first
                if not captcha_input.is_visible():
                    # Check if we already succeeded (race win)
                    if self._check_success(page, worker_logger): return True
                    worker_logger.warning("⚠️ Form/Captcha not visible - checking state...")
                    time.sleep(1)
                    continue

                # 2. SOLVE CAPTCHA (Or use pre-solved code)
                code = None
                
                # Attempt 1 Optimization: Use initial_code if available
                if attempt == 1 and initial_code:
                    worker_logger.info(f"⚡ [SPEED] Using pre-filled captcha code: '{initial_code}'")
                    code = initial_code
                else:
                    # Standard behavior: Solve from scratch
                    # Also applies if initial_code failed (Attempt > 1)
                    success, code, status = self.solver.solve_from_page(page, f"SUBMIT_{attempt}")
                    
                    # [CRITICAL FIX] ABORT ON BLACK CAPTCHA
                    # If we detect a black captcha, it means we are IP banned.
                    # Retrying will only prolong the ban. We must DIE immediately.
                    if status == "BLACK_IMAGE" or status == "POISONED":
                         worker_logger.critical(f"💀 [BLACK CAPTCHA] Session POISONED on attempt {attempt} - ABORTING!")
                         session.health = SessionHealth.POISONED
                         return False

                    if not success or not code:
                        worker_logger.warning("🔄 Captcha solve failed, refreshing...")
                        self._refresh_captcha(page)
                        continue

                # 3. INTERACT (Human Timing)
                worker_logger.info(f"⌨️ Attempt {attempt}: Entering code '{code}'...")
                captcha_input.click()
                captcha_input.fill("")
                time.sleep(random.uniform(0.1, 0.3)) # Micro-delay
                captcha_input.type(code, delay=50)   # Human typing
                time.sleep(random.uniform(0.3, 0.7)) # Hesitation before submit

                # 4. SUBMIT WITH NAVIGATION WAIT
                # This is the Anti-Race Condition Logic
                worker_logger.info("⚡ Submitting and WAITING for response...")
                
                try:
                    # We expect either a navigation (success/redirect) or a reload (failure)
                    # We trigger the submit and wait for the action to resolve
                    with page.expect_navigation(timeout=15000):
                        page.keyboard.press("Enter")
                except Exception as e:
                    worker_logger.debug(f"Navigation wait timeout/error: {e}")
                    # If timeout, page might have updated in-place via AJAX
                
                # 5. VALIDATE STATE
                time.sleep(1) # Settling time
                
                # Case A: Success
                if self._check_success(page, worker_logger):
                    return True
                
                # Case B: Soft Fail (Wrong Captcha) - Back on form
                if page.locator("input[name='lastname']").count() > 0:
                    worker_logger.warning(f"❌ Rejected (Soft) - Back on form. Retrying...")
                    self._refresh_captcha(page)
                    
                    # Ensure fields are still filled (sometimes they clear)
                    if page.locator("input[name='lastname']").input_value() == "":
                         worker_logger.info("📝 Re-filling cleared fields...")
                         self._fill_booking_form(page, session, worker_logger)
                    continue
                    
                # Case C: Hard Fail (Session Error)
                content = page.content().lower()
                if "ref-id" in content or "beginnen sie" in content:
                    worker_logger.error("💀 Hard Failure: Session invalid.")
                    return False

            except Exception as e:
                worker_logger.error(f"⚠️ Submit exception: {e}")
                time.sleep(1)
        
        return False

    def _check_success(self, page: Page, logger) -> bool:
        """Helper to scan for success indicators"""
        content = page.content().lower()
        success_terms = ["appointment number", "termin nummer", "successfully", "erfolgreich"]
        
        for term in success_terms:
            if term in content:
                logger.critical(f"🏆 VICTORY! Found marker: '{term}'")
                self.global_stats.success = True
                self.debug_manager.save_critical_screenshot(page, "VICTORY", 1)
                self.stop_event.set()
                return True
        return False

    def _refresh_captcha(self, page: Page):
        """Helper to refresh captcha safely"""
        try:
            refresh = page.locator("#appointment_newAppointmentForm_form_newappointment_refreshcaptcha")
            if refresh.is_visible(): refresh.click()
            else: self.solver.reload_captcha(page)
            time.sleep(1.5)
        except: pass
    
    def _handle_success(self):
        """Handle successful booking"""
        logger.info("\n" + "=" * 70)
        logger.info("[SUCCESS] MISSION ACCOMPLISHED - BOOKING SUCCESSFUL!")
        logger.info("=" * 70)
        
        runtime = (datetime.datetime.now() - self.start_time).total_seconds()
        
        send_alert(
            f"ELITE SNIPER V2.0 - SUCCESS!\n"
            f"[+] Appointment booked!\n"
            f"Session: {self.session_id}\n"
            f"Runtime: {runtime:.0f}s\n"
            f"Stats: {self.global_stats.get_summary()}"
        )
    
    def _handle_completion(self):
        """Handle completion without success"""
        logger.info("\n" + "=" * 70)
        logger.info("[STOP] Session completed without booking")
        logger.info("=" * 70)
        
        runtime = (datetime.datetime.now() - self.start_time).total_seconds()
        logger.info(f"[TIME] Runtime: {runtime:.0f}s")
        logger.info(f"[STATS] Final stats: {self.global_stats.get_summary()}")


# Entry point
if __name__ == "__main__":
    sniper = EliteSniperV2()
    success = sniper.run()
    sys.exit(0 if success else 1)