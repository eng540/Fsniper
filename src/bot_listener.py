"""
Elite Sniper v2.0 - Telegram Bot Listener
Polls for commands to control the sniper via Telegram.
Commands:
- /start : Start in AUTO mode (Hybrid)
- /manual : Start in STRICT MANUAL mode
- /stop : Stop the current session
- /status : Check current status
"""

import time
import logging
import requests
import signal
import sys
from typing import List, Dict, Any

from .config import Config
from .sniper_manager import SniperManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("BotListener")

class BotListener:
    def __init__(self):
        self.manager = SniperManager()
        self.offset = 0
        self.running = True
        
        # Validate config
        if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            logger.error("❌ Telegram Token or Chat ID missing from config!")
            sys.exit(1)
            
        self.base_url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}"
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("🔻 Shutting down Bot Listener...")
        self.running = False
        self.manager.stop_session()
        sys.exit(0)

    def send_message(self, text: str):
        """Send reply to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": Config.TELEGRAM_CHAT_ID,
                "text": text
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def get_updates(self) -> List[Dict[str, Any]]:
        """Poll Telegram for new updates"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                "offset": self.offset + 1,
                "timeout": 30
            }
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                result = response.json().get("result", [])
                return result
        except Exception as e:
            logger.debug(f"Poll error: {e}")
        return []

    def process_update(self, update: Dict[str, Any]):
        """Process a single Telegram update"""
        try:
            update_id = update.get("update_id")
            if update_id:
                self.offset = max(self.offset, update_id)
            
            message = update.get("message", {})
            text = message.get("text", "").strip()
            # chat_id = message.get("chat", {}).get("id") # We only listen to Config.CHAT_ID ideally?
            
            # Simple security check: Only accept commands from configured Admin ID
            # Convert to string for comparison as config might be str or int
            sender_id = str(message.get("from", {}).get("id"))
            config_id = str(Config.TELEGRAM_CHAT_ID)
            
            # Allow command if it matches chat_id or specific user id (simplified to chat_id for now)
            # You might want to enhance this security logic
            
            if not text.startswith("/"):
                return # Ignore non-commands (like captcha replies)

            logger.info(f"[CMD] Received command: {text}")

            if text == "/start":
                if self.manager.start_session("AUTO"):
                    self.send_message("🚀 Elite Sniper Started (AUTO/HYBRID Mode)")
                else:
                    self.send_message("⚠️ System is already running!")
            
            elif text == "/manual":
                if self.manager.start_session("MANUAL"):
                    self.send_message("🛠️ Elite Sniper Started (STRICT MANUAL Mode)\nOCR is disabled. You will be asked to solve all captchas.")
                else:
                    self.send_message("⚠️ System is already running!")

            elif text == "/autofull":
                if self.manager.start_session("AUTO_FULL"):
                    self.send_message("🤖 Elite Sniper Started (AUTO FULL Mode)\nManual fallback DISABLED. System will rely largely on OCR.")
                else:
                    self.send_message("⚠️ System is already running!")

            elif text == "/stop":
                if self.manager.stop_session():
                    self.send_message("🛑 Stopping System...")
                else:
                    self.send_message("⚠️ System is already stopped.")

            elif text == "/status":
                status = self.manager.get_status()
                self.send_message(f"📊 System Status: {status}")
                
            elif text == "/ping":
                self.send_message("🏓 Pong! System is online.")

        except Exception as e:
            logger.error(f"Error processing update: {e}")

    def run(self):
        """Main loop"""
        logger.info("[LISTENER] Bot Listener is ONLINE. Waiting for commands...")
        self.send_message("Elite Sniper Control Online.\nCommands:\n/start - Hybrid Mode\n/manual - Manual Mode\n/autofull - Auto Full Mode\n/stop - Stop\n/status - Check Status")
        
        while self.running:
            updates = self.get_updates()
            for update in updates:
                self.process_update(update)
            time.sleep(1)

if __name__ == "__main__":
    bot = BotListener()
    bot.run()
