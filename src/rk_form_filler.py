
import os
import sys
import time
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

# Ensure src is in pythonpath
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
    from captcha import EnhancedCaptchaSolver
except ImportError:
    # If run from root
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from src.config import Config
    from src.captcha import EnhancedCaptchaSolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RKFormFiller")

def fill_form(page):
    """
    Fills the appointment form based on Config and analyzed HTML structure.
    """
    logger.info("Filling form fields...")
    
    # 1. Last Name
    page.fill('input[name="lastname"]', Config.LAST_NAME)
    logger.info(f"Filled Last Name: {Config.LAST_NAME}")
    
    # 2. First Name
    page.fill('input[name="firstname"]', Config.FIRST_NAME)
    logger.info(f"Filled First Name: {Config.FIRST_NAME}")
    
    # 3. Email & Repeat
    page.fill('input[name="email"]', Config.EMAIL)
    page.fill('input[name="emailrepeat"]', Config.EMAIL)
    logger.info(f"Filled Email: {Config.EMAIL}")
    
    # 4. Passport (fields[0].content)
    # The selector name is 'fields[0].content'
    page.fill('input[name="fields[0].content"]', Config.PASSPORT)
    logger.info(f"Filled Passport: {Config.PASSPORT}")
    
    # 5. Phone (fields[1].content)
    page.fill('input[name="fields[1].content"]', Config.PHONE)
    logger.info(f"Filled Phone: {Config.PHONE}")
    
    # 6. Purpose (fields[2].content)
    # This is a select box. Config.PURPOSE might need mapping or direct value.
    # Config.PURPOSE is "aupair", but the values are like "Au-Pair".
    # Config.PURPOSE_VALUES maps "aupair" -> "Au-Pair"
    
    purpose_key = Config.PURPOSE.lower()
    purpose_value = Config.PURPOSE_VALUES.get(purpose_key, Config.DEFAULT_PURPOSE)
    
    try:
        page.select_option('select[name="fields[2].content"]', value=purpose_value)
        logger.info(f"Selected Purpose: {purpose_value}")
    except Exception as e:
        logger.error(f"Failed to select purpose '{purpose_value}': {e}")
        # Fallback: try to select by label or index if needed, but value should work
        
    return True

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) # Run headed to see it work
        context = browser.new_context()
        page = context.new_page()
        
        # Use local file for verification as discussed
        local_file_path = r"d:\ai\sniper\rk\RK-Termin form.html"
        if not os.path.exists(local_file_path):
             logger.error(f"Local text file not found: {local_file_path}")
             return

        file_url = Path(local_file_path).as_uri()
        logger.info(f"Navigating to: {file_url}")
        page.goto(file_url)
        
        # Wait for form to be visible
        try:
            page.wait_for_selector('form#appointment_newAppointmentForm', timeout=5000)
        except:
             logger.error("Form not found on the page!")
             # Maybe it's the wrong file or needed loading time
             time.sleep(2)
        
        # Fill Form
        fill_form(page)
        
        # Solve Captcha
        logger.info("Attempting to solve captcha...")
        solver = EnhancedCaptchaSolver(manual_only=False) # Try auto first
        
        # Pass the page to the solver
        # The solver expects to find captcha on the page
        # Location 'FORM'
        success, code, status = solver.solve_from_page(page, location="FORM")
        
        if success and code:
            logger.info(f"Captcha solved: {code} (Status: {status})")
            # In a real scenario, we would click submit here.
            # But since this is a local file, submitting might not work or will POST to a dead endpoint.
            # We will just highlight the submit button or click it and catch the error.
            
            submit_btn_selector = 'input[id="appointment_newAppointmentForm_appointment_addAppointment"]'
            # Scroll to make sure it's viewable
            page.locator(submit_btn_selector).scroll_into_view_if_needed()
            
            logger.info("Simulating click on Submit button (might fail on local file)...")
            # Just hover to show we found it
            page.hover(submit_btn_selector)
            time.sleep(1)
            
            # page.click(submit_btn_selector) 
            logger.info("Form fill and captcha solve complete!")
            
        else:
            logger.error(f"Failed to solve captcha: {status}")
            
        # Keep open for a few seconds to inspect
        time.sleep(10)
        browser.close()

if __name__ == "__main__":
    main()
