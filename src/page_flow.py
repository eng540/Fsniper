"""
Elite Sniper v2.0 - Page Flow Detection
Based on reverse engineering of German Embassy website HTML

Page Types:
1. MONTH_PAGE (appointment_showMonth.do) - Shows available days, NO captcha
2. DAY_PAGE (appointment_showDay.do) - Shows time slots, NO captcha  
3. FORM_PAGE (appointment_showForm.do) - Booking form WITH captcha
4. SUCCESS_PAGE - Confirmation page
5. ERROR_PAGE - Error or session expired
"""

import logging
from typing import Tuple, List, Optional
from playwright.sync_api import Page

logger = logging.getLogger("EliteSniperV2.PageFlow")


class PageFlowDetector:
    """
    Detects page type and available actions based on HTML analysis
    """
    
    # Page type constants
    MONTH_PAGE = "MONTH_PAGE"
    DAY_PAGE = "DAY_PAGE"
    FORM_PAGE = "FORM_PAGE"
    SUCCESS_PAGE = "SUCCESS_PAGE"
    ERROR_PAGE = "ERROR_PAGE"
    UNKNOWN_PAGE = "UNKNOWN_PAGE"
    
    # Selectors from HTML analysis
    SELECTOR_AVAILABLE_DAY = "a.arrow[href*='appointment_showDay']"
    SELECTOR_AVAILABLE_SLOT = "a.arrow[href*='appointment_showForm']"
    SELECTOR_BOOK_THIS = "a.arrow:has-text('Book this appointment')"
    SELECTOR_APPOINTMENTS_AVAILABLE = "a.arrow:has-text('Appointments are available')"
    
    # Form fields (from HTML analysis)
    FORM_FIELDS = {
        "lastname": "input[name='lastname']",
        "firstname": "input[name='firstname']",
        "email": "input[name='email']",
        "emailrepeat": "input[name='emailrepeat']",
        "passport": "input[name='fields[0].content']",
        "phone": "input[name='fields[1].content']",
        "purpose": "select[name='fields[2].content']",
        "captcha": "input[name='captchaText']"
    }
    
    def __init__(self):
        pass
    
    def detect_page_type(self, page: Page) -> str:
        """
        Detect the current page type based on URL and content
        
        Returns:
            Page type constant (MONTH_PAGE, DAY_PAGE, FORM_PAGE, etc.)
        """
        try:
            url = page.url.lower()
            content = page.content().lower()
            
            # Check by URL first (most reliable)
            if "appointment_showmonth" in url:
                return self.MONTH_PAGE
            elif "appointment_showday" in url:
                return self.DAY_PAGE
            elif "appointment_showform" in url or "appointment_newappointmentform" in url:
                return self.FORM_PAGE
            elif "appointment_addappointment" in url:
                # Could be success or error after submission
                if "appointment number" in content or "confirmation" in content:
                    return self.SUCCESS_PAGE
                return self.FORM_PAGE  # Could be form with errors
            
            # Check by content
            if "please select a date" in content or "appointments are available" in content:
                return self.MONTH_PAGE
            elif "please select an appointment" in content or "book this appointment" in content:
                return self.DAY_PAGE
            elif "new appointment" in content and "captchatext" in content:
                return self.FORM_PAGE
            elif "appointment number" in content or "successfully" in content:
                return self.SUCCESS_PAGE
            elif "error" in content or "session expired" in content:
                return self.ERROR_PAGE
            
            return self.UNKNOWN_PAGE
            
        except Exception as e:
            logger.error(f"[DETECT] Error detecting page type: {e}")
            return self.ERROR_PAGE
    
    def get_available_days(self, page: Page) -> List[dict]:
        """
        Get list of available days from MONTH_PAGE
        
        Returns:
            List of {'date': str, 'href': str}
        """
        days = []
        try:
            # Look for both types of day links
            selectors = [
                self.SELECTOR_AVAILABLE_DAY,
                self.SELECTOR_APPOINTMENTS_AVAILABLE
            ]
            
            for selector in selectors:
                links = page.locator(selector).all()
                for link in links:
                    href = link.get_attribute("href")
                    text = link.text_content() or ""
                    if href and "showDay" in href:
                        # Extract date from URL (dateStr=DD.MM.YYYY)
                        import re
                        date_match = re.search(r'dateStr=(\d{2}\.\d{2}\.\d{4})', href)
                        date = date_match.group(1) if date_match else ""
                        days.append({
                            "date": date,
                            "href": href,
                            "text": text.strip()
                        })
            
            # Remove duplicates
            seen = set()
            unique_days = []
            for day in days:
                if day["href"] not in seen:
                    seen.add(day["href"])
                    unique_days.append(day)
            
            logger.info(f"[FLOW] Found {len(unique_days)} available days")
            return unique_days
            
        except Exception as e:
            logger.error(f"[FLOW] Error getting available days: {e}")
            return []
    
    def get_available_slots(self, page: Page) -> List[dict]:
        """
        Get list of available time slots from DAY_PAGE
        
        Returns:
            List of {'time': str, 'href': str, 'openingPeriodId': str}
        """
        slots = []
        try:
            links = page.locator(self.SELECTOR_AVAILABLE_SLOT).all()
            
            for link in links:
                href = link.get_attribute("href")
                if href and "showForm" in href:
                    # Extract openingPeriodId from URL
                    import re
                    period_match = re.search(r'openingPeriodId=(\d+)', href)
                    period_id = period_match.group(1) if period_match else ""
                    
                    # Try to get time from parent elements
                    time_text = ""
                    try:
                        parent_div = link.locator("xpath=..").locator("xpath=..").locator("h4").first
                        time_text = parent_div.text_content() or ""
                    except:
                        pass
                    
                    slots.append({
                        "time": time_text.strip(),
                        "href": href,
                        "openingPeriodId": period_id
                    })
            
            logger.info(f"[FLOW] Found {len(slots)} available slots")
            return slots
            
        except Exception as e:
            logger.error(f"[FLOW] Error getting available slots: {e}")
            return []
    
    def is_captcha_present(self, page: Page) -> bool:
        """
        Check if captcha is present on the FORM_PAGE
        
        Returns:
            True if captcha input field exists and is visible
        """
        try:
            captcha_selector = self.FORM_FIELDS["captcha"]
            captcha_input = page.locator(captcha_selector)
            
            if captcha_input.count() > 0:
                return captcha_input.first.is_visible()
            return False
            
        except Exception as e:
            logger.error(f"[FLOW] Error checking captcha: {e}")
            return False
    
    def has_form_errors(self, page: Page) -> Tuple[bool, List[str]]:
        """
        Check if form has validation errors
        
        Returns:
            (has_errors: bool, error_messages: List[str])
        """
        errors = []
        try:
            # Look for error divs
            error_divs = page.locator(".global-error p").all()
            for div in error_divs:
                text = div.text_content()
                if text and text.strip():
                    errors.append(text.strip())
            
            return len(errors) > 0, errors
            
        except Exception as e:
            logger.error(f"[FLOW] Error checking form errors: {e}")
            return False, []
    
    def is_success_page(self, page: Page) -> bool:
        """
        Check if current page is the success confirmation page
        
        Returns:
            True if booking was successful
        """
        try:
            content = page.content().lower()
            success_indicators = [
                "appointment number",
                "confirmation",
                "your appointment has been booked",
                "successfully booked"
            ]
            return any(ind in content for ind in success_indicators)
            
        except Exception as e:
            logger.error(f"[FLOW] Error checking success: {e}")
            return False
    
    def get_no_appointments_message(self, page: Page) -> bool:
        """
        Check if page shows "no appointments" message
        
        Returns:
            True if no appointments are available
        """
        try:
            content = page.content().lower()
            no_apt_indicators = [
                "no appointments",
                "keine termine",
                "currently no date",
                "no free appointments"
            ]
            return any(ind in content for ind in no_apt_indicators)
            
        except Exception as e:
            return False
