"""
SniperManager - Orchestrates the EliteSniperV2 lifecycle
Responsible for starting, stopping, and monitoring the sniper thread.
"""
import threading
import time
import logging
from typing import Optional
from .elite_sniper_v2 import EliteSniperV2

logger = logging.getLogger("SniperManager")

class SniperManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(SniperManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.current_sniper: Optional[EliteSniperV2] = None
        self.sniper_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start_session(self, mode: str = "AUTO") -> bool:
        """Start a new sniper session in the specified mode"""
        with self._lock:
            if self.is_running:
                logger.warning("[!] Sniper is already running!")
                return False
            
            logger.info(f"[START] Starting Sniper Session (Mode: {mode})")
            
            try:
                # Initialize sniper with mode
                self.current_sniper = EliteSniperV2(run_mode=mode)
                
                # Run in separate thread to not block the listener
                self.sniper_thread = threading.Thread(target=self._run_wrapper)
                self.sniper_thread.daemon = True
                self.sniper_thread.start()
                self.is_running = True
                return True
            except Exception as e:
                logger.error(f"[ERROR] Failed to start sniper: {e}")
                self.is_running = False
                return False

    def stop_session(self) -> bool:
        """Stop the current session"""
        with self._lock:
            if not self.is_running or not self.current_sniper:
                logger.warning("[!] No active session to stop")
                return False
            
            logger.info("[STOP] Stopping Sniper Session...")
            self.current_sniper.stop_event.set()
            # We don't join here to keep the listener responsive
            return True

    def get_status(self) -> str:
        """Get current system status"""
        if not self.is_running:
            return "IDLE"
        return f"RUNNING ({self.current_sniper.run_mode})"

    def _run_wrapper(self):
        """Internal wrapper to run the sniper and handle lifecycle"""
        try:
            if self.current_sniper:
                success = self.current_sniper.run()
                
                if success:
                    logger.info("[DONE] Sniper finished successfully (Booking made).")
                else:
                     logger.info("[END] Sniper stopped or failed.")
        except Exception as e:
            logger.error(f"[CRASH] Sniper crashed: {e}")
        finally:
            with self._lock:
                self.is_running = False
                self.current_sniper = None
                logger.info("[RESET] Manager reset to IDLE")
