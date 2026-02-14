"""
Elite Sniper v2.0 - Enhanced Captcha System
Integrates KingSniperV12 safe captcha checking with pre-solving capability
"""

import time
import logging
from typing import Optional, List, Tuple
from playwright.sync_api import Page
import numpy as np
import requests
import json
import base64
import concurrent.futures # For parallel solving
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logger = logging.getLogger("EliteSniperV2.Captcha")

# Try to import ddddocr
try:
    import ddddocr
    DDDDOCR_AVAILABLE = True
except ImportError:
    DDDDOCR_AVAILABLE = False
    logger.warning("ddddocr not available - captcha solving disabled")

# Import config and notifier for manual captcha
from .config import Config
try:
    from . import notifier
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False


class TelegramCaptchaHandler:
    """
    Handle manual captcha solving via Telegram.
    Sends captcha image to user and waits for reply.
    """
    
    def __init__(self):
        self.enabled = Config.MANUAL_CAPTCHA_ENABLED and NOTIFIER_AVAILABLE
        self.timeout = Config.MANUAL_CAPTCHA_TIMEOUT
        self._attempt_count = 0
        
        if self.enabled:
            logger.info("[MANUAL] Telegram captcha handler enabled")
        else:
            logger.info("[MANUAL] Telegram captcha handler disabled")
    
    def request_manual_solution(
        self, 
        image_bytes: bytes, 
        location: str = "CAPTCHA",
        session_age: int = 0,
        attempt: int = 1,
        max_attempts: int = 5
    ) -> Optional[str]:
        """
        Send captcha to Telegram and wait for user solution.
        Uses C2 Queue if available to avoid polling conflicts.
        """
        if not self.enabled:
            logger.warning("[MANUAL] Telegram captcha disabled")
            return None
        
        self._attempt_count += 1
        
        # Build caption for Telegram message
        caption = (
            f"🔐 CAPTCHA REQUIRED\n\n"
            f"📍 Location: {location}\n"
            f"⏱️ Session Age: {session_age}s\n"
            f"🔄 Attempt: {attempt}/{max_attempts}\n\n"
            f"Reply with the 6 characters you see.\n"
            f"Timeout: {self.timeout} seconds"
        )
        
        # Send captcha image
        logger.info(f"[MANUAL] Sending captcha to Telegram for manual solving...")
        
        # Use C2 to send photo if available (preferred)
        success = False
        if hasattr(self, 'c2') and self.c2:
            try:
                # Save bytes to temp file or handle bytes directly?
                # C2.send_photo takes path... 
                # Let's fallback to notifier for sending, but use C2 for receiving.
                # Or improve C2 to handle bytes? Not critical now.
                # Notifier uses requests directly, which is fine.
                result = notifier.send_photo_bytes(image_bytes, caption)
                success = result.get("success")
            except:
                pass
        else:
             result = notifier.send_photo_bytes(image_bytes, caption)
             success = result.get("success")
        
        if not success:
            logger.error("[MANUAL] Failed to send captcha to Telegram")
            return None
        
        # Wait for user reply
        logger.info(f"[MANUAL] Waiting for reply (timeout: {self.timeout}s)...")
        
        # 1. Use C2 Queue (Thread-Safe)
        if hasattr(self, 'c2') and self.c2:
            return self.c2.wait_for_captcha(timeout=self.timeout)
            
        # 2. Fallback to conflicting polling (Last Resort)
        logger.warning("⚠️ C2 not active - falling back to direct polling (may conflict)")
        return notifier.wait_for_captcha_reply(timeout=self.timeout)

    def notify_result(self, success: bool, location: str = ""):
        """Notify user of captcha result"""
        if not self.enabled:
            return
        
        if success:
            notifier.send_alert(f"🎯 CAPTCHA SUCCESS! Moving to {location}...")
        else:
            notifier.send_alert(f"❌ CAPTCHA WRONG - sending new image...")



class CircuitBreaker:
    """
    Circuit Breaker pattern to handle API failures gracefully.
    If failure threshold is reached, open the circuit for a timeout period.
    """
    def __init__(self, threshold: int = 2, timeout: int = 300):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failures += 1
        self.last_failure_time = time.time()
        logger.warning(f"⚡ [CircuitBreaker] Failure recorded ({self.failures}/{self.threshold})")
        
        if self.failures >= self.threshold:
            self.state = "OPEN"
            logger.error(f"⚡ [CircuitBreaker] Threshold reached! Circuit OPEN for {self.timeout}s")
            
    def record_success(self):
        """Record success and reset logic"""
        if self.failures > 0:
            logger.info("⚡ [CircuitBreaker] Success recorded - Resetting failures")
            self.failures = 0
            self.state = "CLOSED"
            
    def is_open(self) -> bool:
        """Check if circuit is open (requests should be blocked)"""
        if self.state == "CLOSED":
            return False
            
        # Check timeout
        elapsed = time.time() - self.last_failure_time
        if elapsed > self.timeout:
            if self.state == "OPEN":
                logger.info("⚡ [CircuitBreaker] Timeout expired - Switch to HALF-OPEN")
                self.state = "HALF-OPEN" # Allow one trial request
                return False # Allow passage for trial
            return False # HALF-OPEN or resets
            
        return True # Still open and within timeout


class CapSolverHandler:
    """
    Handler for CapSolver API
    Docs: https://docs.capsolver.com/en/guide/recognition/ImageToTextTask/
    """
    
    def __init__(self):
        self.api_key = Config.CAPSOLVER_API_KEY
        self.enabled = Config.CAPSOLVER_ENABLED and bool(self.api_key)
        self.api_url = "https://api.capsolver.com/createTask"
        
        if self.enabled:
            logger.info("[CapSolver] Initialized and ENABLED")
        else:
            if Config.CAPSOLVER_ENABLED and not self.api_key:
                logger.warning("[CapSolver] Enabled in config but NO API KEY found!")
            else:
                logger.info("[CapSolver] Disabled")
        
        # Initialize Circuit Breaker
        self.circuit_breaker = CircuitBreaker(
            threshold=Config.CIRCUIT_BREAKER_THRESHOLD,
            timeout=Config.CIRCUIT_BREAKER_TIMEOUT
        )
    
    def solve_image_to_text(self, image_bytes: bytes, location: str = "CAPSOLVER") -> Tuple[Optional[str], str]:
        """
        Solve captcha using CapSolver ImageToTextTask
        
        Returns:
            (code, status)
        """
        if not self.enabled:
            return None, "DISABLED"
            
        # Circuit Breaker Check
        if self.circuit_breaker.is_open():
            logger.warning(f"[{location}] ⚡ CapSolver circuit OPEN (Skipping API call)")
            return None, "CIRCUIT_OPEN"
            
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare payload
            payload = {
                "clientKey": self.api_key,
                "task": {
                    "type": "ImageToTextTask",
                    "module": "common",  # "common" or "number" - common is safer for alphanum
                    "body": image_base64
                }
            }
            
            start_time = time.time()
            logger.info(f"[{location}] Sending request to CapSolver...")
            
            # Send request (createTask for ImageToText returns result immediately usually)
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"[{location}] CapSolver HTTP Error: {response.status_code} - {response.text}")
                self.circuit_breaker.record_failure()
                return None, f"HTTP_{response.status_code}"
                
            data = response.json()
            
            # Check for API errors
            if data.get("errorId", 0) != 0:
                error_code = data.get("errorCode", "UNKNOWN")
                error_desc = data.get("errorDescription", "")
                logger.error(f"[{location}] CapSolver API Error: {error_code} - {error_desc}")
                self.circuit_breaker.record_failure()
                return None, f"API_{error_code}"
                
            # Extract solution
            STATUS = data.get("status")
            if STATUS == "ready":
                solution = data.get("solution", {}).get("text", "")
                elapsed = time.time() - start_time
                logger.info(f"[{location}] CapSolver SOLVED in {elapsed:.2f}s: '{solution}'")
                self.circuit_breaker.record_success()
                return solution, "SUCCESS"
            else:
                logger.warning(f"[{location}] CapSolver status not ready: {STATUS}")
                self.circuit_breaker.record_failure()
                return None, f"STATUS_{STATUS}"
                
        except Exception as e:
            logger.error(f"[{location}] CapSolver Exception: {e}")
            self.circuit_breaker.record_failure()
            return None, "EXCEPTION"




class EnhancedCaptchaSolver:
    """
    Enhanced captcha solver with:
    - Multiple selector attempts (from KingSniperV12)
    - Safe checking without failures
    - Black captcha detection
    - Pre-solving capability
    - Session-aware solving
    """
    
    def __init__(self, mode: str = "HYBRID", c2_instance=None):
        """Initialize OCR engine and manual handler based on mode"""
        self.mode = mode.upper()
        self.manual_only = (self.mode == "MANUAL")
        self.auto_only = (self.mode == "AUTO")
        self.c2 = c2_instance # Store C2 instance
        
        self.ocr = None
        self._pre_solved_code: Optional[str] = None
        self._pre_solved_time: float = 0.0
        self._pre_solve_timeout: float = 30.0  # Pre-solved code expires after 30s
        
        # Initialize CapSolver
        self.capsolver = CapSolverHandler()
        
        # Initialize manual captcha handler (Telegram fallback)
        
        # Initialize manual captcha handler (Telegram fallback)
        # Check if enabled in config AND in compatbile mode
        self.manual_handler = TelegramCaptchaHandler()
        if self.c2:
            self.manual_handler.c2 = self.c2 # Inject C2 into handler
        
        if self.mode == "MANUAL":
             logger.info("[CAPTCHA] Initialized in MANUAL MODE (OCR Disabled)")
        elif self.mode == "AUTO":
             logger.info("[CAPTCHA] Initialized in AUTO MODE (Manual Fallback Disabled)")
        else:
             logger.info("[CAPTCHA] Initialized in HYBRID MODE (Balanced)")
        
        if DDDDOCR_AVAILABLE and not self.manual_only:
            try:
                # !!! تراجع هام: العودة لاستخدام Beta=True لأنها أثبتت كفاءة أعلى !!!
                self.ocr = ddddocr.DdddOcr(beta=True)
                logger.info("Captcha solver initialized (BETA Mode - High Accuracy)")
            except Exception as e:
                logger.error(f"Captcha solver init failed: {e}")
                self.ocr = None
        elif not DDDDOCR_AVAILABLE and not self.manual_only:
            logger.warning("ddddocr not available - captcha solving disabled")
    
    def safe_captcha_check(self, page: Page, location: str = "GENERAL") -> Tuple[bool, bool]:
        """
        Safe captcha presence check (from KingSniperV12)
        
        Returns:
            (has_captcha: bool, check_successful: bool)
        """
        try:
            # Step 1: Check page content for captcha keywords
            page_content = page.content().lower()
            
            captcha_keywords = [
                "captcha", 
                "security code", 
                "verification", 
                "human check",
                "verkaptxt"  # German sites
            ]
            
            has_captcha_text = any(keyword in page_content for keyword in captcha_keywords)
            
            if not has_captcha_text:
                logger.debug(f"[{location}] No captcha keywords found")
                return False, True
            
            # Step 2: Search for captcha input (multiple selectors)
            # Increased timeout to 3000ms for better reliability
            captcha_selectors = self._get_captcha_selectors()
            
            for selector in captcha_selectors:
                try:
                    if page.locator(selector).first.is_visible(timeout=3000):
                        logger.info(f"[{location}] Captcha found: {selector}")
                        return True, True
                except:
                    continue
            
            # Found keywords but no input field
            logger.warning(f"[{location}] Captcha text found but NO INPUT VISIBLE")
            return False, True
            
        except Exception as e:
            logger.error(f"[{location}] Captcha check error: {e}")
            return False, False
    
    def verify_captcha_solved(self, page: Page, location: str = "VERIFY") -> Tuple[bool, str]:
        """
        Verify if captcha was successfully solved and we're on the next page
        
        Returns:
            (success: bool, page_type: str)
            page_type: CAPTCHA_PAGE, CALENDAR_PAGE, TIME_SLOTS_PAGE, FORM_PAGE, SUCCESS_PAGE, UNKNOWN
        """
        import time as time_module
        
        # Wait for page to stabilize (max 3 retries)
        for attempt in range(3):
            try:
                # Wait for page to be ready
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=3000)
                except:
                    pass
                
                content = page.content().lower()
                
                # Check if still on captcha page
                has_captcha_input = page.locator("input[name='captchaText']").count() > 0
                
                if has_captcha_input:
                    return False, "CAPTCHA_PAGE"
                
                # Check for calendar page indicators
                calendar_indicators = [
                    "please select a date",
                    "appointments are available",
                    "appointment_showday",
                    "no appointments",
                    "keine termine"
                ]
                if any(ind in content for ind in calendar_indicators):
                    return True, "CALENDAR_PAGE"
                
                # Check for time slots page
                time_indicators = [
                    "please select an appointment",
                    "book this appointment",
                    "appointment_showform"
                ]
                if any(ind in content for ind in time_indicators):
                    return True, "TIME_SLOTS_PAGE"
                
                # Check for booking form page
                form_indicators = [
                    "new appointment",
                    "appointment_newappointmentform",
                    "appointment_addappointment"
                ]
                if any(ind in content for ind in form_indicators):
                    return True, "FORM_PAGE"
                
                # Check for success page
                success_indicators = [
                    "appointment number",
                    "confirmation",
                    "successfully"
                ]
                if any(ind in content for ind in success_indicators):
                    return True, "SUCCESS_PAGE"
                
                return False, "UNKNOWN"
                
            except Exception as e:
                if attempt < 2:
                    time_module.sleep(0.5)
                    continue
                logger.error(f"[{location}] Verification error: {e}")
                return False, "ERROR"
        
        return False, "TIMEOUT"
    
    def _get_captcha_selectors(self) -> List[str]:
        """
        Get list of possible captcha selectors
        From KingSniperV12 with additions
        """
        return [
            "input[name='captchaText']",
            "input[name='captcha']",
            "input#captchaText",
            "input#captcha",
            "input[type='text'][placeholder*='code']",
            "input[type='text'][placeholder*='Code']",
            "#appointment_captcha_month input[type='text']",
            "input.verkaptxt",
            "input.captcha-input",
            "input[id*='captcha']",
            "input[name*='captcha']",
            "form[id*='captcha'] input[type='text']"
        ]
    
    def _get_captcha_image_selectors(self) -> List[str]:
        """Get list of possible captcha image selectors"""
        return [
            "captcha > div",
            "div.captcha-image",
            "div#captcha",
            "img[alt*='captcha']",
            "img[alt*='CAPTCHA']",
            "canvas.captcha"
        ]
    
    def _extract_base64_captcha(self, page: Page, location: str = "EXTRACT") -> Optional[bytes]:
        """
        Extract captcha image from CSS background-image base64 data URL
        This is how the German embassy website embeds captcha images
        
        Returns:
            Image bytes or None if not found
        """
        import base64
        import re
        
        try:
            # Try to find captcha div with base64 background
            captcha_div = page.locator("captcha > div").first
            
            if not captcha_div.is_visible(timeout=2000):
                logger.debug(f"[{location}] Captcha div not visible")
                return None
            
            # Get the style attribute
            style = captcha_div.get_attribute("style")
            
            if not style:
                logger.debug(f"[{location}] No style attribute on captcha div")
                return None
            
            # Extract base64 from: background:white url('data:image/jpg;base64,XXXXX') 
            # Pattern matches the base64 data
            pattern = r"url\(['\"]?data:image/[^;]+;base64,([A-Za-z0-9+/=]+)['\"]?\)"
            match = re.search(pattern, style)
            
            if not match:
                logger.debug(f"[{location}] No base64 pattern found in style")
                return None
            
            base64_data = match.group(1)
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            
            logger.info(f"[{location}] Extracted captcha from base64 ({len(image_bytes)} bytes)")
            return image_bytes
            
        except Exception as e:
            logger.warning(f"[{location}] Base64 extraction failed: {e}")
            return None
    
    def _get_captcha_image(self, page: Page, location: str = "GET_IMG") -> Optional[bytes]:
        """
        Get captcha image using multiple methods:
        1. First try CSS background base64 extraction (most reliable for this website)
        2. Fallback to screenshot method
        
        Returns:
            Image bytes or None
        """
        # Method 1: Try base64 extraction first (most reliable for this website)
        image_bytes = self._extract_base64_captcha(page, location)
        if image_bytes:
            return image_bytes
        
        # Method 2: Fallback to screenshot
        for img_selector in self._get_captcha_image_selectors():
            try:
                element = page.locator(img_selector).first
                if element.is_visible(timeout=1000):
                    image_bytes = element.screenshot(timeout=5000)
                    logger.info(f"[{location}] Got captcha via screenshot: {img_selector}")
                    return image_bytes
            except:
                continue
        
        logger.warning(f"[{location}] Could not get captcha image by any method")
        return None
    
    def detect_black_captcha(self, image_bytes: bytes) -> bool:
        """
        Detect poisoned/black captcha
        Black captcha = session is POISONED and needs to be recreated
        
        Black captcha indicators:
        - Very small file size (< 2000 bytes) - includes 931 bytes black images
        - Normal captcha is typically 5000+ bytes
        
        CRITICAL: If detected, DO NOT RETRY! Abort session immediately.
        """
        if len(image_bytes) < 2000:
            logger.critical(f"⛔ [BLACK CAPTCHA] Detected! Size: {len(image_bytes)} bytes - Session POISONED!")
            return True
        
        return False
    
    def validate_captcha_result(self, code: str, location: str = "VALIDATE") -> Tuple[bool, str]:
        """
        Validate captcha OCR result
        
        Rules based on German embassy website behavior:
        - 6 characters = VALID (normal captcha)
        - 7-8 characters = WARNING (session aging, too many refreshes)
        - < 4 or > 8 characters = INVALID (likely OCR error)
        - "4333" or similar repeated = BLACK CAPTCHA garbage
        
        Returns:
            (is_valid: bool, status: str)
            status: VALID, AGING, INVALID, BLACK_DETECTED, TOO_SHORT, TOO_LONG
        """
        if not code:
            logger.warning(f"[{location}] Empty captcha code")
            return False, "EMPTY"
        
        # Clean the code
        code = code.strip().replace(" ", "")
        code_len = len(code)
        
        # Detect black captcha garbage patterns
        # Only truly repeated patterns like "4444", "333", "0000" are garbage
        black_patterns = ["4333", "333", "444", "1111", "0000", "4444", "3333"]
        is_all_same = len(set(code)) == 1  # All characters are the same
        if code in black_patterns or is_all_same:
            logger.critical(f"[{location}] BLACK CAPTCHA pattern detected: '{code}'")
            return False, "BLACK_DETECTED"
        
        # Check length
        if code_len < 4:
            logger.warning(f"[{location}] Captcha too short: '{code}' ({code_len} chars)")
            return False, "TOO_SHORT"
        
        if code_len == 6:
            # Perfect! Normal captcha
            logger.info(f"[{location}] Valid 6-char captcha: '{code}'")
            return True, "VALID"
        
        if code_len == 7:
            # Warning - session aging, but still usable
            logger.warning(f"[{location}] 7-char captcha (session aging): '{code}'")
            return True, "AGING_7"
        
        if code_len == 8:
            # Critical warning - session near death
            logger.warning(f"[{location}] 8-char captcha (session near death): '{code}'")
            return True, "AGING_8"
        
        if code_len > 8:
            logger.error(f"[{location}] Captcha too long: '{code}' ({code_len} chars)")
            return False, "TOO_LONG"
        
        # 4-5 chars - REJECT! Embassy requires exactly 6 chars
        # OCR probably missed a character
        if code_len in [4, 5]:
            logger.warning(f"[{location}] OCR incomplete: '{code}' ({code_len} chars) -需要6个字符!")
            return False, "TOO_SHORT"

    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Restored V1 Strong Preprocessing:
        1. Grayscale
        2. Upscale (2.5x) - Critical for ddddocr accuracy
        3. Contrast Adjustment (CLAHE) - Critical for faint text
        4. Thresholding + Denoising
        """
        if not OPENCV_AVAILABLE:
            return image_bytes

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 1. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Strong Upscale (2.5x) - From V1
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            
            # 3. Strong Contrast (CLAHE) - From V1
            # This makes faint text visible
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 4. Thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 5. Denoising - From V1
            kernel = np.ones((2,2), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            _, encoded_img = cv2.imencode('.png', opening)
            return encoded_img.tobytes()
        except Exception as e:
            logger.debug(f"Image preprocessing failed: {e}")
            return image_bytes
    def solve(self, image_bytes: bytes, location: str = "SOLVE") -> Tuple[str, str]:
        """
        Solve captcha from image bytes with validation
        
        STRATEGY UPDATE: Always Enhance First (Accuracy Over Speed)
        Since the raw image often results in "TOO_SHORT" or failures,
        we skip the raw attempt and go straight to the enhanced version.
        
        Returns:
            (captcha_code: str, status: str)
            status: VALID, AGING_7, AGING_8, BLACK_DETECTED, TOO_SHORT, etc.
        """
        if self.manual_only:
             logger.info(f"[{location}] Manual Mode active - Skipping OCR")
             return "", "MANUAL_REQUIRED"

        if not self.ocr:
            logger.error("[OCR] Engine not initialized")
            return "", "NO_OCR"
        
        try:
            # Detect black captcha first (by image size)
            if self.detect_black_captcha(image_bytes):
                return "", "BLACK_IMAGE"
            
            # ALWAYS PREPROCESS FIRST
            # Upscale + Contrast Enhancement for maximum accuracy
            enhanced_bytes = self._preprocess_image(image_bytes)
            
            # ═══════════════════════════════════════════════════════════════
            # STRATEGY: PARALLEL SOLVING (RACE) vs SEQUENTIAL
            # ═══════════════════════════════════════════════════════════════
            
            if Config.PARALLEL_SOLVING_ENABLED and self.capsolver.enabled and self.ocr:
                 logger.info(f"[{location}] 🚀 STARTING PARALLEL RACE: CapSolver vs Local OCR")
                 
                 with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit tasks
                    future_capsolver = executor.submit(self.capsolver.solve_image_to_text, enhanced_bytes, location)
                    # For local OCR, we wrap it in a lambda or simple call
                    future_local = executor.submit(self._solve_local_ocr, enhanced_bytes, location)
                    
                    futures = [future_capsolver, future_local]
                    
                    # Wait for FIRST COMPLETED
                    for future in as_completed(futures):
                        try:
                            result_code, result_status = future.result()
                            
                            # If successful result, return immediately!
                            if result_code and result_status in ["SUCCESS", "VALID", "AGING_7", "AGING_8"]:
                                # Clean and Validate again to be sure (double check)
                                final_code = self._clean_ocr_result(result_code)
                                is_valid, val_status = self.validate_captcha_result(final_code, f"{location}_PARALLEL")
                                
                                if is_valid:
                                    logger.info(f"[{location}] 🏆 WINNER: {result_status} -> '{final_code}'")
                                    return final_code, result_status
                        except Exception as e:
                            logger.error(f"Parallel task failed: {e}")
                            
                    # If we fall through here, it means the first finished task failed.
                    # We should wait for the second one? 
                    # Simpler logic: If the first one failed, check if the other is done or wait for it.
                    # Ideally wait for ALL if first fails.
                    
                    # Let's try collecting all results if the first one wasn't a winner
                    logger.warning(f"[{location}] First parallel result wasn't a winner - checking others...")
                    
                    for future in futures:
                        if future.done(): continue # Already checked
                        try:
                            result_code, result_status = future.result() # Wait for the slower one
                            if result_code and result_status in ["SUCCESS", "VALID", "AGING_7", "AGING_8"]:
                                final_code = self._clean_ocr_result(result_code)
                                is_valid, val_status = self.validate_captcha_result(final_code, f"{location}_PARALLEL_SLOW")
                                if is_valid:
                                    logger.info(f"[{location}] 🥈 RUNNER-UP WON: {result_status} -> '{final_code}'")
                                    return final_code, result_status
                        except: pass
                        
                 logger.warning(f"[{location}] 🏁 Parallel race ended with NO WINNER")
                 return "", "ALL_FAILED"

            # ═══════════════════════════════════════════════════════════════
            # STRATEGY: SEQUENTIAL (FALLBACK)
            # ═══════════════════════════════════════════════════════════════

            # PRIORITY 1: CAPSOLVER (PAID/PREMIUM)
            if self.capsolver.enabled:
                # Send Enhanced Image directly
                code, status = self.capsolver.solve_image_to_text(enhanced_bytes, location)
                
                if code and status == "SUCCESS":
                    code = self._clean_ocr_result(code)
                    is_valid, val_status = self.validate_captcha_result(code, f"{location}_CAPSOLVER")
                    
                    if is_valid:
                        logger.info(f"[{location}] ✅ CapSolver (Enhanced) result: '{code}'")
                        return code, "CAPSOLVER"
                    else:
                        logger.warning(f"[{location}] CapSolver result invalid: '{code}' ({val_status})")
                else:
                    logger.warning(f"[{location}] CapSolver failed ({status})")
                
                logger.warning(f"[{location}] CapSolver failed - Falling back to local OCR...")

            # PRIORITY 2: LOCAL DDDDOCR (FREE/FALLBACK)
            if self.ocr:
                 result, status = self._solve_local_ocr(enhanced_bytes, location)
                 if status in ["VALID", "AGING_7", "AGING_8"]:
                     return result, status
                        
            return "", "ALL_FAILED"

        except Exception as e:
            logger.error(f"[{location}] Captcha solve error: {e}")
            return "", "ERROR"

    def _solve_local_ocr(self, image_bytes: bytes, location: str) -> Tuple[str, str]:
        """Helper for local OCR solving (thread-safe wrapper)"""
        try:
             logger.info(f"[{location}] Trying local ddddocr (Enhanced)...")
             result = self.ocr.predict(image_bytes)

             result = result.replace(" ", "").strip().lower()
             result = self._clean_ocr_result(result)
             is_valid, status = self.validate_captcha_result(result, location)
             
             if is_valid:
                 logger.info(f"[{location}] Local OCR solved: '{result}' - Status: {status}")
                 return result, status
             else:
                 logger.warning(f"[{location}] Local OCR failed: '{result}' - Status: {status}")
                 return result, status
        except Exception as e:
            logger.error(f"Local OCR Error: {e}")
            return "", "ERROR"
    
    def _clean_ocr_result(self, text: str) -> str:
        """
        Clean common OCR mistakes for the German embassy captcha.
        The captcha uses lowercase letters and digits only.
        
        [CRITICAL FIX] Removed hardcoded replacements (o->0, g->9, etc.)
        because they were corrupting valid captchas (e.g. ego2fy -> e902fy).
        We now trust the raw ddddocr output after enhanced preprocessing.
        """
        if not text:
            return ""
            
        # 1. Basic cleanup
        text = text.strip().replace(" ", "")
        
        # 2. Filter allowed characters only (alphanumeric)
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        cleaned = ''.join(c for c in text if c in allowed_chars)
        
        return cleaned
    
    def pre_solve(self, page: Page, location: str = "PRE_SOLVE") -> Tuple[bool, Optional[str], str]:
        """
        Pre-solve captcha for instant submission later
        
        Returns:
            (success: bool, captcha_code: Optional[str], status: str)
        """
        try:
            # Check if captcha exists
            has_captcha, check_ok = self.safe_captcha_check(page, location)
            
            if not check_ok:
                logger.error(f"[{location}] Pre-solve captcha check failed")
                return False, None, "CHECK_FAILED"
            
            if not has_captcha:
                logger.debug(f"[{location}] No captcha to pre-solve")
                return True, None, "NO_CAPTCHA"
            
            # Find captcha image using unified method
            image_bytes = self._get_captcha_image(page, location)
            
            if not image_bytes:
                logger.warning(f"[{location}] Captcha image not found for pre-solve")
                return False, None, "NO_IMAGE"
            
            # Solve captcha with validation
            code, status = self.solve(image_bytes, location)
            
            if not code:
                logger.warning(f"[{location}] Pre-solve failed: {status}")
                return False, None, status
            
            # Cache the solution
            self._pre_solved_code = code
            self._pre_solved_time = time.time()
            self._pre_solved_status = status
            
            logger.info(f"[{location}] Pre-solved captcha: '{code}' - Status: {status}")
            return True, code, status
            
        except Exception as e:
            logger.error(f"[{location}] Pre-solve error: {e}")
            return False, None, "ERROR"
    
    def get_pre_solved(self) -> Optional[str]:
        """
        Get pre-solved captcha code if still valid
        
        Returns:
            Captcha code or None if expired/unavailable
        """
        if not self._pre_solved_code:
            return None
        
        # Check if expired
        age = time.time() - self._pre_solved_time
        if age > self._pre_solve_timeout:
            logger.warning("Pre-solved captcha expired")
            self._pre_solved_code = None
            return None
        
        return self._pre_solved_code
    
    def clear_pre_solved(self):
        """Clear pre-solved captcha"""
        self._pre_solved_code = None
        self._pre_solved_time = 0.0
    
    def solve_from_page(
        self, 
        page: Page, 
        location: str = "GENERAL",
        timeout: int = 10000,
        session_age: int = 0,
        attempt: int = 1,
        max_attempts: int = 5
    ) -> Tuple[bool, Optional[str]]:
        """
        Complete captcha solving workflow
        Uses pre-solved code if available, then OCR, then manual Telegram fallback
        
        Returns:
            (success: bool, captcha_code: Optional[str], status: str)
        """
        try:
            # Check if captcha exists
            has_captcha, check_ok = self.safe_captcha_check(page, location)
            
            if not check_ok:
                logger.error(f"[{location}] Captcha check failed")
                return False, None, "CHECK_FAILED"
            
            if not has_captcha:
                logger.debug(f"[{location}] No captcha present")
                return True, None, "NO_CAPTCHA"
            
            # Find captcha input field
            input_selector = None
            for selector in self._get_captcha_selectors():
                try:
                    if page.locator(selector).first.is_visible(timeout=1000):
                        input_selector = selector
                        break
                except:
                    continue
            
            if not input_selector:
                logger.warning(f"[{location}] Captcha input not found")
                return False, None, "NO_INPUT"
            
            # Check for pre-solved code first
            code = self.get_pre_solved()
            status = getattr(self, '_pre_solved_status', 'VALID')
            
            if code:
                logger.info(f"[{location}] Using pre-solved captcha: '{code}'")
                self.clear_pre_solved()
            else:
                # [UPDATED] Internal Retry Loop for AUTO mode accuracy
                internal_max_retries = 3
                for internal_attempt in range(internal_max_retries):
                    
                    # Find captcha image using unified method
                    image_bytes = self._get_captcha_image(page, location)
                    
                    if not image_bytes:
                        logger.warning(f"[{location}] Captcha image not found")
                        return False, None, "NO_IMAGE"
                    
                    # Solve captcha with OCR validation
                    code, status = self.solve(image_bytes, location)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # EXECUTION MODE LOGIC
                    # ═══════════════════════════════════════════════════════════════
                    
                    # 1. AUTO MODE: Smart Retry for TOO_SHORT
                    if self.auto_only:
                        if status == "TOO_SHORT":
                            logger.warning(f"[{location}] Result TOO_SHORT in AUTO mode - RELOADING ({internal_attempt+1}/{internal_max_retries})...")
                            if internal_attempt < internal_max_retries - 1:
                                self.reload_captcha(page, f"{location}_RELOAD_{internal_attempt}")
                                continue # NEXT TRY via loop
                            else:
                                logger.warning(f"[{location}] Max internal retries reached for TOO_SHORT")
                                # Fall through to skip logic
                        
                        if not code or status in ["TOO_SHORT", "TOO_LONG", "NO_OCR", "MANUAL_REQUIRED"]:
                            logger.warning(f"[{location}] OCR failed ({status}) and Mode is AUTO - SKIPPING MANUAL")
                            return False, None, f"AUTO_SKIP_{status}"
                        
                        # If we have a code (VALID, AGING, etc.), break loop and submit
                        break
                    
                    # 2. HYBRID/MANUAL MODE: If OCR fails (or skipped in MANUAL), try Telegram
                    if not code or status in ["TOO_SHORT", "TOO_LONG", "NO_OCR", "MANUAL_REQUIRED"]:
                        logger.info(f"[{location}] OCR failed ({status}), trying manual Telegram...")
                    
                    # Request manual solution via Telegram
                    manual_code = self.manual_handler.request_manual_solution(
                        image_bytes=image_bytes,
                        location=location,
                        session_age=session_age,
                        attempt=attempt,
                        max_attempts=max_attempts
                    )
                    
                    if manual_code:
                        code = manual_code
                        status = "MANUAL"
                        logger.info(f"[{location}] Using manual solution: '{code}'")
                    else:
                        logger.warning(f"[{location}] Manual solve also failed/timeout")
                        return False, None, "MANUAL_TIMEOUT"
            
            # Fill captcha (Force write for reliability)
            try:
                page.fill(input_selector, code, timeout=3000, force=True)
                logger.info(f"[{location}] Captcha filled: '{code}' - Status: {status}")
                return True, code, status
            except Exception as e:
                logger.error(f"[{location}] Failed to fill captcha: {e}")
                return False, None, "FILL_ERROR"
            
        except Exception as e:
            logger.error(f"[{location}] Captcha solving workflow error: {e}")
            return False, None, "ERROR"
    
    def submit_captcha(self, page: Page, method: str = "auto") -> bool:
        """
        Submit captcha with enhanced reliability
        """
        try:
            logger.info(f"[CAPTCHA] Submitting answer (Method: {method})...")
            
            # 1. Try generic submit buttons first if method is auto or click
            if method in ["auto", "click"]:
                # Specific buttons for this appointment system
                buttons = [
                    "input[name='submit']",
                    "input[value='Weiter']",
                    "input[value='Continue']",
                    "button:has-text('Weiter')",
                    "button:has-text('Continue')",
                    "input[type='submit']"
                ]
                
                for selector in buttons:
                    try:
                        btn = page.locator(selector).first
                        if btn.is_visible(timeout=500):
                            btn.click(timeout=2000)
                            logger.info(f"Clicked submit button: {selector}")
                            return True
                    except:
                        continue
            
            # 2. Fallback to Enter key (or if method='enter')
            page.keyboard.press("Enter")
            logger.info("Sent Enter key")
            return True
            
        except Exception as e:
            logger.error(f"[CAPTCHA] Submit error: {e}")
            return False
    
    def verify_captcha_solved(self, page: Page, location: str = "VERIFY") -> Tuple[bool, str]:
        """
        Verify if captcha was solved successfully by checking if we moved to next page
        or if captcha is still present.
        """
        logger.info(f"[{location}] Verifying captcha solution...")
        
        # Give it time to load - use extended timeout for manual mode
        start_time = time.time()
        timeout = 10.0 if getattr(self, 'manual_only', False) else 5.0
        
        while time.time() - start_time < timeout:
            try:
                current_url = page.url
                # Safe access to content - if navigating, this might fail, which is fine
                try:
                    content = page.content().lower()
                except Exception:
                    # Page is likely navigating/loading - this is actually a good sign!
                    time.sleep(0.5)
                    continue

                # 1. Check if we moved to Day view (Success)
                if "appointment_showday" in current_url.lower() or page.locator("a.arrow").count() > 0:
                     return True, "DAY_PAGE"
                
                # 2. Check for form page (Success)
                if "appointment_showform" in current_url.lower():
                    return True, "FORM_PAGE"

                # 3. Check for explicitly wrong captcha error
                if "security code" in content and ("valid" in content or "match" in content or "nicht korrekt" in content):
                     logger.warning(f"[{location}] Server reported WRONG captcha")
                     return False, "WRONG_CAPTCHA"

            except Exception as e:
                logger.debug(f"[{location}] Verification check transient error: {e}")
            
            time.sleep(0.5)
            
        # If we are still here, check if captcha is still visible
        has_captcha, _ = self.safe_captcha_check(page, location)
        if has_captcha:
             return False, "CAPTCHA_STILL_PRESENT"
             
        # If captcha is gone but we aren't on success page, assume success for now (maybe loading)
        return True, "UNKNOWN_PAGE"

    def reload_captcha(self, page: Page, location: str = "RELOAD") -> bool:
        """
        Reload captcha image by clicking "Load another picture" button.
        This is used when captcha solving fails - instead of going back to start,
        we just reload and try again.
        
        Returns:
            True if reload was successful
        """
        try:
            # FACT-BASED SELECTORS from RK-Termin form.html
            reload_selectors = [
                # 1. The exact ID from the booking form (RK-Termin form.html)
                "#appointment_newAppointmentForm_form_newappointment_refreshcaptcha",
                # 2. The name attribute from the booking form
                "input[name='action:appointment_refreshCaptcha']",
                # 3. The exact ID from the category form (RK-Termin - Kategorie.html)
                "#appointment_captcha_month_refreshcaptcha",
                "input[name='action:appointment_refreshCaptchamonth']",
                # 4. Fallbacks based on value (confirmed "Load another picture")
                "input[value='Load another picture']",
                "input[value='Bild laden']"
            ]
            
            for selector in reload_selectors:
                try:
                    button = page.locator(selector).first
                    if button.is_visible(timeout=1000):
                        # Try regular click first
                        try:
                            button.click(timeout=2000)
                        except:
                            # JavaScript fallback click
                            page.evaluate(f'document.querySelector("{selector}")?.click()')
                        
                        logger.info(f"[{location}] Clicked reload button - waiting for new captcha...")
                        page.wait_for_timeout(1500)
                        return True
                except:
                    continue
            
            # Final fallback: Try JavaScript to find any reload-related button
            try:
                result = page.evaluate("""
                    const buttons = Array.from(document.querySelectorAll('input[type="submit"], button'));
                    for(const btn of buttons) {
                        const val = (btn.value || btn.textContent || '').toLowerCase();
                        if(val.includes('another') || val.includes('refresh') || val.includes('reload') || val.includes('anderes')) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                """)
                if result:
                    logger.info(f"[{location}] Clicked reload via JS fallback")
                    page.wait_for_timeout(1500)
                    return True
            except:
                pass
            
            logger.warning(f"[{location}] Could not find reload captcha button")
            return False
            
        except Exception as e:
            logger.error(f"[{location}] Reload captcha error: {e}")
            return False
    
    def solve_form_captcha_with_retry(
        self, 
        page: Page, 
        location: str = "FORM_RETRY",
        max_attempts: int = 5,
        session_age: int = 0
    ) -> Tuple[bool, Optional[str], str]:
        """
        Solve form captcha with retry logic.
        
        IMPORTANT: This method is used specifically for FORM page captcha.
        When captcha solving fails, instead of returning to start,
        it clicks "Load another picture" and tries again.
        
        This is the SMART logic: we don't lose our valuable slot by going back,
        we just reload the captcha and try again until we succeed.
        
        Args:
            page: Playwright page
            location: Log location identifier
            max_attempts: Maximum number of attempts
            
        Returns:
            (success: bool, captcha_code: Optional[str], status: str)
        """
        if self.manual_only:
            logger.info("🛠️ MANUAL MODE: Enabling INFINITE RETRY loop on form page!")
            max_attempts = 1000  # Virtually infinite for manual mode
            
        for attempt in range(max_attempts):
            attempt_num = attempt + 1
            
            logger.info(f"[{location}] Captcha attempt {attempt_num}/{max_attempts}")
            
            # Try to solve
            success, code, status = self.solve_from_page(
                page, 
                f"{location}_A{attempt_num}",
                session_age=session_age,
                attempt=attempt_num,
                max_attempts=1 # Inside the loop we scan once per cycle
            )
            
            if success and code:
                # Got a valid solution!
                logger.info(f"[{location}] SUCCESS on attempt {attempt_num}: '{code}'")
                return True, code, status
            
            # Failed - try to reload captcha
            if attempt < max_attempts - 1:  # Don't reload on last attempt
                logger.warning(f"[{location}] Attempt {attempt_num} failed ({status}), reloading captcha...")
                
                # Check session age to prevent zombie loops
                if session_age > 1800: # 30 minutes
                     logger.critical(f"[{location}] Session too old during infinite loop - aborting")
                     return False, None, "SESSION_TOO_OLD"

                if not self.reload_captcha(page, f"{location}_RELOAD"):
                    logger.error(f"[{location}] Could not reload captcha - aborting")
                    # If reload click fails (button gone?), we might have lost the page. Return False.
                    return False, None, "RELOAD_FAILED"
                
                # Small delay after reload
                time.sleep(1.0)
        
        # All attempts failed
        logger.error(f"[{location}] All {max_attempts} attempts failed")
        return False, None, "MAX_ATTEMPTS_REACHED"


# Backward compatibility
class CaptchaSolver:
    """Original captcha solver for backward compatibility"""
    
    def __init__(self):
        if DDDDOCR_AVAILABLE:
            self.ocr = ddddocr.DdddOcr(beta=True)
        else:
            self.ocr = None
    
    def solve(self, image_bytes: bytes) -> str:
        if not self.ocr:
            return ""
        try:
            res = self.ocr.classification(image_bytes)
            res = res.replace(" ", "").strip()
            print(f"[AI] Captcha Solved: {res}")
            return res
        except Exception as e:
            print(f"[AI] Error solving captcha: {e}")
            return ""