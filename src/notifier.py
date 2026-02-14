"""
Elite Sniper v2.0 - Telegram Notifier
Enhanced with screenshot support and rate limiting
"""

import time
import logging
import requests
from typing import Optional
from .config import Config

logger = logging.getLogger("EliteSniperV2.Notifier")

# Rate limiting
_last_message_time = 0
_message_interval = 1.0  # Minimum seconds between messages


def _check_rate_limit() -> bool:
    """Check if we can send a message (rate limiting)"""
    global _last_message_time
    now = time.time()
    if now - _last_message_time < _message_interval:
        return False
    _last_message_time = now
    return True


def send_alert(message: str, parse_mode: str = "HTML") -> bool:
    """
    Send text message to Telegram
    
    Args:
        message: Message text
        parse_mode: "HTML" or "Markdown"
    
    Returns:
        Success status
    """
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram not configured")
        return False
    
    if not _check_rate_limit():
        logger.debug("Rate limited, skipping message")
        return False
    
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": Config.TELEGRAM_CHAT_ID, 
        "text": message,
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.debug("📤 Message sent to Telegram")
            return True
        else:
            logger.warning(f"⚠️ Telegram error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Telegram send error: {e}")
        return False


def send_photo(photo_path: str, caption: str = "") -> bool:
    """
    Send photo to Telegram
    
    Args:
        photo_path: Path to image file
        caption: Optional caption
    
    Returns:
        Success status
    """
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram not configured")
        return False
    
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendPhoto"
    data = {
        "chat_id": Config.TELEGRAM_CHAT_ID, 
        "caption": caption[:1024]  # Telegram caption limit
    }
    
    try:
        with open(photo_path, "rb") as image_file:
            files = {"photo": image_file}
            response = requests.post(url, data=data, files=files, timeout=30)
            
        if response.status_code == 200:
            logger.debug("📤 Photo sent to Telegram")
            return True
        else:
            logger.warning(f"⚠️ Telegram photo error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Telegram photo error: {e}")
        return False


def send_document(doc_path: str, caption: str = "") -> bool:
    """
    Send document to Telegram
    
    Args:
        doc_path: Path to document file
        caption: Optional caption
    
    Returns:
        Success status
    """
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram not configured")
        return False
    
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendDocument"
    data = {
        "chat_id": Config.TELEGRAM_CHAT_ID, 
        "caption": caption[:1024]
    }
    
    try:
        with open(doc_path, "rb") as doc_file:
            files = {"document": doc_file}
            response = requests.post(url, data=data, files=files, timeout=30)
            
        if response.status_code == 200:
            logger.debug("📤 Document sent to Telegram")
            return True
        else:
            logger.warning(f"⚠️ Telegram document error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Telegram document error: {e}")
        return False


# ==================== MANUAL CAPTCHA SUPPORT ====================

# Track last update ID for polling
_last_update_id = 0


def get_telegram_updates(timeout: int = 30) -> list:
    """
    Get new messages from Telegram (long polling).
    Used for receiving manual captcha solutions.
    
    Args:
        timeout: Long polling timeout in seconds
    
    Returns:
        List of new message updates
    """
    global _last_update_id
    
    if not Config.TELEGRAM_TOKEN:
        return []
    
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/getUpdates"
    params = {
        "offset": _last_update_id + 1,
        "timeout": timeout,
        "allowed_updates": ["message"]
    }
    
    try:
        response = requests.get(url, params=params, timeout=timeout + 5)
        if response.status_code == 200:
            result = response.json()
            if result.get("ok") and result.get("result"):
                updates = result["result"]
                if updates:
                    # Update offset to acknowledge received messages
                    _last_update_id = updates[-1]["update_id"]
                return updates
        return []
    except Exception as e:
        logger.debug(f"Telegram update error: {e}")
        return []


def send_photo_bytes(image_bytes: bytes, caption: str = "") -> dict:
    """
    Send photo from bytes directly (not file path).
    Used for sending captcha images.
    
    Args:
        image_bytes: Image data as bytes
        caption: Optional caption
    
    Returns:
        Response dict with 'success' and 'message_id'
    """
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured")
        return {"success": False, "message_id": None}
    
    url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendPhoto"
    data = {
        "chat_id": Config.TELEGRAM_CHAT_ID,
        "caption": caption[:1024]
    }
    
    try:
        import io
        files = {"photo": ("captcha.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        response = requests.post(url, data=data, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                message_id = result.get("result", {}).get("message_id")
                logger.debug(f"Captcha sent to Telegram (msg_id: {message_id})")
                return {"success": True, "message_id": message_id}
        
        logger.warning(f"Telegram photo error: {response.status_code}")
        return {"success": False, "message_id": None}
        
    except Exception as e:
        logger.error(f"Telegram photo error: {e}")
        return {"success": False, "message_id": None}


def wait_for_captcha_reply(timeout: int = 60) -> str:
    """
    Wait for user to reply with captcha solution.
    
    Args:
        timeout: Maximum wait time in seconds
    
    Returns:
        User's reply text or None if timeout
    """
    import time
    start_time = time.time()
    
    # Clear any pending messages first
    get_telegram_updates(timeout=1)
    
    while time.time() - start_time < timeout:
        remaining = int(timeout - (time.time() - start_time))
        
        # Poll for new messages (5 second intervals)
        updates = get_telegram_updates(timeout=5)
        
        for update in updates:
            message = update.get("message", {})
            text = message.get("text", "").strip()
            
            # Check if it's from the right chat
            chat_id = str(message.get("chat", {}).get("id", ""))
            if chat_id == str(Config.TELEGRAM_CHAT_ID) and text:
                # Validate it looks like a captcha (alphanumeric, reasonable length)
                if text.isalnum() and 4 <= len(text) <= 10:
                    logger.info(f"Received captcha reply: '{text}'")
                    return text.lower()  # Captchas are lowercase
                else:
                    # Inform user of invalid format
                    send_alert(f"⚠️ Invalid format: '{text}'\nPlease send 6 alphanumeric characters only.")
        
        # Brief pause between polls
        time.sleep(0.5)
    
    logger.warning("Captcha reply timeout")
    return None


def send_status_update(
    session_id: str,
    status: str,
    stats: dict = None,
    mode: str = "PATROL"
) -> bool:
    """
    Send formatted status update
    
    Args:
        session_id: Current session ID
        status: Status message
        stats: Optional statistics dict
        mode: Current operational mode
    """
    emoji_map = {
        "PATROL": "🔍",
        "WARMUP": "⏳",
        "PRE_ATTACK": "⚙️",
        "ATTACK": "🔥",
        "SUCCESS": "🏆"
    }
    
    emoji = emoji_map.get(mode, "📊")
    
    message = f"{emoji} <b>Elite Sniper v2.0</b>\n"
    message += f"└ Session: <code>{session_id[:20]}...</code>\n"
    message += f"└ Mode: {mode}\n"
    message += f"└ Status: {status}\n"
    
    if stats:
        message += f"\n📊 Stats:\n"
        message += f"└ Scans: {stats.get('scans', 0)}\n"
        message += f"└ Days Found: {stats.get('days_found', 0)}\n"
        message += f"└ Slots Found: {stats.get('slots_found', 0)}\n"
        message += f"└ Captchas: {stats.get('captchas_solved', 0)}/{stats.get('captchas_failed', 0)}\n"
    
    return send_alert(message)


def send_success_notification(
    session_id: str,
    worker_id: int,
    screenshot_path: Optional[str] = None
) -> bool:
    """
    Send success notification with optional screenshot
    
    Args:
        session_id: Session ID
        worker_id: Worker that achieved success
        screenshot_path: Optional path to success screenshot
    """
    message = (
        f"🎉🏆 <b>VICTORY! APPOINTMENT SECURED!</b> 🏆🎉\n\n"
        f"✅ Elite Sniper v2.0 has successfully booked an appointment!\n"
        f"📍 Worker: #{worker_id}\n"
        f"🆔 Session: <code>{session_id}</code>\n"
        f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"<b>Check your email for confirmation!</b>"
    )
    
    # Send text message first
    send_alert(message)
    
    # Send screenshot if available
    if screenshot_path:
        send_photo(screenshot_path, "🏆 SUCCESS SCREENSHOT")
    
    return True


def send_error_notification(
    session_id: str,
    error: str,
    worker_id: Optional[int] = None
) -> bool:
    """
    Send error notification
    
    Args:
        session_id: Session ID
        error: Error message
        worker_id: Optional worker ID
    """
    worker_str = f"Worker #{worker_id}" if worker_id else "System"
    
    message = (
        f"🚨 <b>ERROR</b>\n"
        f"└ {worker_str}\n"
        f"└ {error[:200]}\n"
        f"└ Session: <code>{session_id[:20]}...</code>"
    )
    
    return send_alert(message)