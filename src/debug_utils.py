"""
Elite Sniper v2.0 - Debug & Evidence Utilities
Enhanced debugging with Telegram integration and incident reporting
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from playwright.sync_api import Page

logger = logging.getLogger("EliteSniperV2.Debug")


class DebugManager:
    """
    Debug and evidence management system
    From KingSniperV12 with Telegram integration
    """
    
    def __init__(self, session_id: str, base_dir: str = "evidence"):
        """
        Initialize debug manager
        
        Args:
            session_id: Unique session identifier
            base_dir: Base directory for evidence
        """
        self.session_id = session_id
        self.base_dir = base_dir
        
        # Create directory structure
        self.session_dir = os.path.join(base_dir, session_id)
        self.debug_dir = os.path.join(self.session_dir, "debug")
        self.screenshots_dir = os.path.join(self.session_dir, "screenshots")
        self.logs_dir = os.path.join(self.session_dir, "logs")
        
        for directory in [self.session_dir, self.debug_dir, self.screenshots_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"[DIR] Evidence directory: {self.session_dir}")
    
    def save_debug_html(
        self, 
        page: Page, 
        stage: str, 
        worker_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Save page HTML for debugging
        From KingSniperV12
        
        Args:
            page: Playwright page object
            stage: Stage identifier (e.g., "month_scan", "form_fill")
            worker_id: Optional worker ID for multi-session
        
        Returns:
            Path to saved HTML file
        """
        try:
            timestamp = int(time.time())
            
            # Build filename
            if worker_id:
                filename = f"w{worker_id}_{stage}_{timestamp}.html"
            else:
                filename = f"{stage}_{timestamp}.html"
            
            filepath = os.path.join(self.debug_dir, filename)
            
            # Get HTML content
            html_content = page.content()
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.debug(f"📄 Saved HTML: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to save HTML: {e}")
            return None
    
    def save_screenshot(
        self, 
        page: Page, 
        name: str,
        worker_id: Optional[int] = None,
        send_telegram: bool = False,
        telegram_caption: str = ""
    ) -> Optional[str]:
        """
        Save page screenshot with optional Telegram notification
        
        Args:
            page: Playwright page object
            name: Screenshot name
            worker_id: Optional worker ID
            send_telegram: Whether to send to Telegram
            telegram_caption: Caption for Telegram message
        
        Returns:
            Path to saved screenshot
        """
        try:
            timestamp = int(time.time())
            
            if worker_id:
                filename = f"w{worker_id}_{name}_{timestamp}.png"
            else:
                filename = f"{name}_{timestamp}.png"
            
            filepath = os.path.join(self.screenshots_dir, filename)
            
            # Take screenshot
            page.screenshot(path=filepath, full_page=True)
            
            logger.debug(f"📸 Saved screenshot: {filename}")
            
            # Send to Telegram if requested
            if send_telegram:
                try:
                    from .notifier import send_photo
                    caption = telegram_caption or f"📸 {name} - W{worker_id or 0}"
                    send_photo(filepath, caption)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send screenshot to Telegram: {e}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to save screenshot: {e}")
            return None
    
    def save_critical_screenshot(
        self, 
        page: Page, 
        event_name: str,
        worker_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Save and send critical screenshot to Telegram
        Used for success, errors, and important events
        
        Args:
            page: Playwright page object
            event_name: Name of the critical event
            worker_id: Optional worker ID
        
        Returns:
            Path to saved screenshot
        """
        caption = f"🚨 {event_name.upper()}"
        if worker_id:
            caption += f" [W{worker_id}]"
        caption += f" | {datetime.now().strftime('%H:%M:%S')}"
        
        return self.save_screenshot(
            page=page,
            name=f"critical_{event_name}",
            worker_id=worker_id,
            send_telegram=True,
            telegram_caption=caption
        )
    
    def save_forensic_state(
        self,
        page: Page,
        step_name: str,
        worker_id: Optional[int] = None,
        extra_data: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Save complete forensic state for debugging flow issues.
        Captures HTML, screenshot, URL, and page state.
        
        Args:
            page: Playwright page object
            step_name: Name of the current step (e.g., "month_captcha", "form_submit")
            worker_id: Optional worker ID
            extra_data: Additional data to save (e.g., captcha result, form values)
        
        Returns:
            Dictionary with paths to saved files
        """
        evidence = {}
        timestamp = int(time.time())
        prefix = f"w{worker_id}_" if worker_id else ""
        
        try:
            # 1. Save HTML
            html_path = self.save_debug_html(page, f"forensic_{step_name}", worker_id)
            if html_path:
                evidence['html'] = html_path
            
            # 2. Save screenshot
            screenshot_path = self.save_screenshot(page, f"forensic_{step_name}", worker_id)
            if screenshot_path:
                evidence['screenshot'] = screenshot_path
            
            # 3. Save page state (URL, title, hidden fields)
            state = {
                "step_name": step_name,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "worker_id": worker_id,
                "url": page.url,
                "title": page.title(),
            }
            
            # Extract hidden form fields
            try:
                hidden_fields = page.evaluate("""
                    () => {
                        const fields = {};
                        document.querySelectorAll('input[type="hidden"]').forEach(el => {
                            fields[el.name] = el.value;
                        });
                        return fields;
                    }
                """)
                state["hidden_fields"] = hidden_fields
            except:
                pass
            
            # Extract visible form values
            try:
                form_values = page.evaluate("""
                    () => {
                        const values = {};
                        document.querySelectorAll('input[type="text"], input[type="email"], select').forEach(el => {
                            if (el.name) {
                                values[el.name] = el.value;
                            }
                        });
                        return values;
                    }
                """)
                state["form_values"] = form_values
            except:
                pass
            
            # Add extra data
            if extra_data:
                state["extra_data"] = extra_data
            
            # Save state JSON
            state_filename = f"{prefix}forensic_{step_name}_{timestamp}_state.json"
            state_path = os.path.join(self.debug_dir, state_filename)
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            evidence['state'] = state_path
            
            logger.debug(f"[FORENSIC] Saved state: {step_name}")
            
        except Exception as e:
            logger.warning(f"[FORENSIC] Save error: {e}")
        
        return evidence
    
    def save_stats(self, stats: Dict[str, Any], filename: str = "stats.json") -> bool:
        """
        Save statistics to JSON file
        
        Args:
            stats: Statistics dictionary
            filename: Output filename
        
        Returns:
            Success status
        """
        try:
            filepath = os.path.join(self.session_dir, filename)
            
            # Add metadata
            stats_with_meta = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                **stats
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats_with_meta, f, indent=2)
            
            logger.info(f"[DISK] Stats saved: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save stats: {e}")
            return False
    
    def save_incident(self, incident: Dict[str, Any]) -> bool:
        """
        Save incident report
        From KingSniperV12 Incident system
        
        Args:
            incident: Incident dictionary
        
        Returns:
            Success status
        """
        try:
            timestamp = int(time.time())
            incident_type = incident.get('type', 'unknown')
            filename = f"incident_{incident_type}_{timestamp}.json"
            
            filepath = os.path.join(self.logs_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(incident, f, indent=2)
            
            logger.info(f"[INCIDENT] Incident logged: {incident_type}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save incident: {e}")
            return False
    
    def save_incident_with_evidence(
        self,
        page: Page,
        incident: Dict[str, Any],
        worker_id: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Save incident with full evidence (HTML + Screenshot)
        
        Args:
            page: Playwright page object
            incident: Incident dictionary
            worker_id: Optional worker ID
        
        Returns:
            Dictionary with paths to saved evidence
        """
        evidence_paths = {}
        incident_type = incident.get('type', 'unknown')
        
        # Save HTML
        html_path = self.save_debug_html(page, f"incident_{incident_type}", worker_id)
        if html_path:
            evidence_paths['html'] = html_path
        
        # Save screenshot
        screenshot_path = self.save_screenshot(
            page, 
            f"incident_{incident_type}", 
            worker_id,
            send_telegram=incident.get('severity') in ['ERROR', 'CRITICAL'],
            telegram_caption=f"[INCIDENT] Incident: {incident_type}"
        )
        if screenshot_path:
            evidence_paths['screenshot'] = screenshot_path
        
        # Add evidence paths to incident
        incident['evidence_paths'] = evidence_paths
        
        # Save incident JSON
        self.save_incident(incident)
        
        return evidence_paths
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get session summary information
        
        Returns:
            Summary dictionary
        """
        try:
            return {
                "session_id": self.session_id,
                "directory": self.session_dir,
                "debug_files": len(os.listdir(self.debug_dir)) if os.path.exists(self.debug_dir) else 0,
                "screenshots": len(os.listdir(self.screenshots_dir)) if os.path.exists(self.screenshots_dir) else 0,
                "log_files": len(os.listdir(self.logs_dir)) if os.path.exists(self.logs_dir) else 0
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get summary: {e}")
            return {}
    
    def cleanup_old_files(self, max_age_hours: int = 48):
        """
        Clean up old debug files
        
        Args:
            max_age_hours: Maximum file age in hours
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            deleted_count = 0
            
            for directory in [self.debug_dir, self.screenshots_dir]:
                if not os.path.exists(directory):
                    continue
                
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        
                        if file_age > max_age_seconds:
                            os.remove(filepath)
                            deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"[CLEANUP] Cleaned up {deleted_count} old files")
            
        except Exception as e:
            logger.error(f"[ERROR] Cleanup failed: {e}")
    
    def create_session_report(self, stats: Dict[str, Any] = None) -> str:
        """
        Create a comprehensive session report
        
        Args:
            stats: Optional statistics dictionary
        
        Returns:
            Path to report file
        """
        try:
            report = {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "summary": self.get_session_summary(),
                "stats": stats or {}
            }
            
            filename = f"session_report_{int(time.time())}.json"
            filepath = os.path.join(self.session_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"[REPORT] Session report created: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create session report: {e}")
            return ""
