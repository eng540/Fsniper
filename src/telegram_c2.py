
import time
import threading
import logging
import requests
import queue
import json
from typing import Optional, Dict, Any, Callable
from .config import Config

logger = logging.getLogger("EliteSniperV2.C2")

class TelegramCommander(threading.Thread):
    """
    Centralized Telegram Command & Control System.
    Single Source of Truth for all incoming Telegram updates.
    """
    
    # UI Layout
    KEYBOARD_LAYOUT = {
        "keyboard": [
            [{"text": "📸 Screenshot"}, {"text": "📊 Status"}],
            [{"text": "▶ Resume"}, {"text": "⏸ Pause"}],
            [{"text": "🤖 Auto"}, {"text": "⚖️ Hybrid"}, {"text": "👤 Manual"}]
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "one_time_keyboard": False
    }
    
    def __init__(self, bot_instance=None):
        super().__init__()
        self.bot = bot_instance  # Reference to EliteSniperV2 instance
        self.daemon = True
        self.running = False
        self.last_update_id = 0
        self.captcha_reply_queue = queue.Queue()
        self.cmd_lock = threading.Lock()
        
    def run(self):
        """Main polling loop"""
        if not Config.TELEGRAM_TOKEN:
            logger.warning("[C2] Telegram disabled (no token)")
            return
            
        logger.info("[C2] Telegram Commander started")
        self.running = True
        self._send_message("📡 <b>Electronic Warfare Suite Online</b>", with_keyboard=True)
        
        while self.running:
            try:
                updates = self._get_updates(timeout=10)
                for update in updates:
                    self._process_update(update)
                
                time.sleep(1) # Small delay to be nice to API
            except Exception as e:
                logger.error(f"[C2] Loop error: {e}")
                time.sleep(5)
                
    def stop(self):
        self.running = False
        
    def _get_updates(self, timeout: int = 30) -> list:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/getUpdates"
        params = {
            "offset": self.last_update_id + 1,
            "timeout": timeout,
            "allowed_updates": ["message"]
        }
        try:
            response = requests.get(url, params=params, timeout=timeout + 5)
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    updates = result.get("result", [])
                    if updates:
                        self.last_update_id = updates[-1]["update_id"]
                    return updates
            return []
        except Exception:
            return []

    def _process_update(self, update: dict):
        """Route update to Command or Captcha Queue"""
        message = update.get("message", {})
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        
        # Security check
        if chat_id != str(Config.TELEGRAM_CHAT_ID):
            return

        if not text:
            return

        logger.info(f"[C2] Received: '{text}'")

        # Normalize text (handle emojis)
        cmd_trigger = text.lower()
        
        # 1. Command Mapping
        if "screenshot" in cmd_trigger or cmd_trigger == "/screenshot":
            self._handle_command("/screenshot")
        elif "status" in cmd_trigger or cmd_trigger == "/status":
            self._handle_command("/status")
        elif "resume" in cmd_trigger or cmd_trigger == "/resume":
            self._handle_command("/resume")
        elif "pause" in cmd_trigger or cmd_trigger == "/pause":
            self._handle_command("/pause")
        elif "auto" in cmd_trigger or cmd_trigger == "/auto":
            self._handle_command("/auto")
        elif "manual" in cmd_trigger or cmd_trigger == "/manual":
            self._handle_command("/manual")
        elif "hybrid" in cmd_trigger or cmd_trigger == "/hybrid":
            self._handle_command("/hybrid")
        elif text.startswith("/"):
            self._handle_command(text)
        # 2. Captcha Reply (fallback)
        else:
            self.captcha_reply_queue.put(text)

    def _handle_command(self, cmd_text: str):
        """Execute commands"""
        if not self.bot:
            self._send_message("⚠️ Bot instance not connected")
            return
            
        cmd = cmd_text.lower().split()[0]
        
        if cmd == "/status":
            report = self.bot.get_status_report()
            self._send_message(report)
            
        elif cmd == "/auto":
            self.bot.set_mode("AUTO")
            self._send_message("🚀 Switched to <b>AUTO MODE</b>")
            
        elif cmd == "/manual":
            self.bot.set_mode("MANUAL")
            self._send_message("🛡️ Switched to <b>MANUAL MODE</b>")
            
        elif cmd == "/hybrid":
            self.bot.set_mode("HYBRID")
            self._send_message("⚖️ Switched to <b>HYBRID MODE</b>")
            
        elif cmd == "/pause":
            self.bot.pause_execution()
            self._send_message("⏸️ <b>PAUSED</b>. Scanning suspended.")
            
        elif cmd == "/resume":
            self.bot.resume_execution()
            self._send_message("▶️ <b>RESUMED</b>. Scanning active.")
            
        elif cmd == "/screenshot":
            self._send_message("📸 Requesting screenshot (Flag Set)...")
            self.bot.request_screenshot() # Set flag
        
        else:
            self._send_message(f"❓ Unknown command: {cmd}")

    def wait_for_captcha(self, timeout=60) -> Optional[str]:
        """Thread-safe method for Captcha Handler"""
        # Clear old items
        with self.captcha_reply_queue.mutex:
            self.captcha_reply_queue.queue.clear()
            
        try:
            return self.captcha_reply_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def _send_message(self, text, with_keyboard=False):
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            data = {
                "chat_id": Config.TELEGRAM_CHAT_ID, 
                "text": text,
                "parse_mode": "HTML"
            }
            
            if with_keyboard:
                data["reply_markup"] = json.dumps(self.KEYBOARD_LAYOUT)
                
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def send_photo(self, path, caption):
        """Public method for bot to send photos"""
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendPhoto"
            data = {
                "chat_id": Config.TELEGRAM_CHAT_ID, 
                "caption": caption[:1024]
            }
            with open(path, "rb") as image_file:
                files = {"photo": image_file}
                requests.post(url, data=data, files=files, timeout=30)
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
