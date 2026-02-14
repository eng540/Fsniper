"""
Elite Sniper v2.0 - NTP Time Synchronization Module
Provides sub-second accuracy for Zero-Hour precision attacks
"""

import time
import logging
import datetime
from typing import Optional
from threading import Thread, Event

logger = logging.getLogger("EliteSniperV2.NTP")


class NTPTimeSync:
    """
    NTP Time Synchronization with multiple server fallback
    Ensures sub-second accuracy for Zero-Hour attacks
    """
    
    def __init__(self, servers: list = None, sync_interval: int = 300):
        """
        Initialize NTP sync
        
        Args:
            servers: List of NTP servers to try
            sync_interval: Re-sync interval in seconds
        """
        self.servers = servers or [
            "pool.ntp.org",
            "time.google.com", 
            "time.windows.com",
            "time.nist.gov"
        ]
        self.sync_interval = sync_interval
        self.offset = 0.0
        self.last_sync = 0.0
        self.sync_count = 0
        self.stop_event = Event()
        self._sync_thread: Optional[Thread] = None
        
        # Initial sync
        self.sync()
    
    def sync(self) -> bool:
        """
        Synchronize time with NTP servers
        
        Returns:
            True if sync successful, False otherwise
        """
        try:
            import ntplib
            client = ntplib.NTPClient()
            
            for server in self.servers:
                try:
                    response = client.request(server, version=3, timeout=5)
                    self.offset = response.offset
                    self.last_sync = time.time()
                    self.sync_count += 1
                    
                    logger.info(
                        f"[NTP] Sync OK [{server}] - "
                        f"Offset: {self.offset:.4f}s"
                    )
                    return True
                    
                except Exception as e:
                    logger.debug(f"NTP {server} failed: {e}")
                    continue
            
            logger.warning("[WARN] All NTP servers failed, using local time")
            return False
            
        except ImportError:
            logger.error("[ERROR] ntplib not installed - run: pip install ntplib")
            return False
        except Exception as e:
            logger.error(f"[ERROR] NTP sync error: {e}")
            return False
    
    def get_corrected_time(self) -> datetime.datetime:
        """
        Get current UTC time with NTP correction
        
        Returns:
            Corrected UTC datetime
        """
        utc_now = datetime.datetime.utcnow()
        corrected = utc_now + datetime.timedelta(seconds=self.offset)
        return corrected
    
    def get_offset_ms(self) -> float:
        """Get current offset in milliseconds"""
        return self.offset * 1000
    
    def time_since_sync(self) -> float:
        """Get seconds since last successful sync"""
        if self.last_sync == 0:
            return float('inf')
        return time.time() - self.last_sync
    
    def needs_resync(self) -> bool:
        """Check if re-synchronization is needed"""
        return self.time_since_sync() > self.sync_interval
    
    def start_background_sync(self):
        """Start background sync thread"""
        if self._sync_thread and self._sync_thread.is_alive():
            return
        
        self.stop_event.clear()
        self._sync_thread = Thread(
            target=self._background_sync_loop,
            daemon=True,
            name="NTP-Sync"
        )
        self._sync_thread.start()
        logger.info("[NTP] Background NTP sync started")
    
    def stop_background_sync(self):
        """Stop background sync thread"""
        self.stop_event.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        logger.info("[NTP] Background NTP sync stopped")
    
    def _background_sync_loop(self):
        """Background sync loop"""
        while not self.stop_event.is_set():
            # Wait for interval or stop event
            if self.stop_event.wait(timeout=self.sync_interval):
                break
            
            # Perform sync
            self.sync()
    
    def get_status(self) -> dict:
        """Get sync status information"""
        return {
            "offset_seconds": self.offset,
            "offset_ms": self.get_offset_ms(),
            "last_sync": self.last_sync,
            "time_since_sync": self.time_since_sync(),
            "sync_count": self.sync_count,
            "needs_resync": self.needs_resync()
        }
