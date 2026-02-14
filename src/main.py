"""
Elite Sniper v2.0 - Main Entry Point
Production-grade 24/7 appointment booking system

Usage:
    python -m src.main
    
Features:
    - 3 parallel sessions (1 Scout + 2 Attackers)
    - NTP-synchronized Zero-Hour precision at 2:00 AM Aden time
    - Automatic recovery and session management
    - Telegram notifications with screenshots
"""

import time
import logging
import sys
import os
import signal

# Add the parent directory to sys.path to allow running from src directly or root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Elite Sniper V2
try:
    from src.elite_sniper_v2 import EliteSniperV2
except ImportError:
    # Fallback if run from inside src
    from elite_sniper_v2 import EliteSniperV2

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MainLauncher")


def run_elite_sniper_v2():
    """
    Run Elite Sniper v2.0 with automatic recovery
    Implements supervisor pattern for 24/7 operation
    """
    retry_count = 0
    max_retries = 10  # Maximum restart attempts
    min_runtime = 60  # Minimum runtime before counting as failed start
    
    while retry_count < max_retries:
        start_time = time.time()
        
        try:
            logger.info("="*60)
            logger.info("   ELITE SNIPER v2.0 - STRICT MODE")
            logger.info("   Telegram Bot: DISABLED (Interactive Mode)")
            logger.info("="*60)

            try:
                # Initialize Sniper
                sniper = EliteSniperV2(run_mode="AUTO")
                success = sniper.run()
            except KeyboardInterrupt:
                logger.info("\n[STOP] Shutdown requested by user")
                return False
            
            if success:
                # Mission accomplished!
                logger.info("[SUCCESS] MISSION ACCOMPLISHED! System shutting down gracefully.")
                return True
            else:
                # Completed without success - normal termination
                runtime = time.time() - start_time
                
                if runtime < min_runtime:
                    # Quick failure - something is wrong
                    retry_count += 1
                    logger.warning(f"[WARN] Quick exit after {runtime:.0f}s - possible issue")
                else:
                    # Normal completion - reset retry count
                    retry_count = 0
                    logger.info(f"[INFO] Session completed after {runtime:.0f}s - restarting...")
                
                # Wait before restart
                wait_time = min(30 * (retry_count + 1), 300)  # Max 5 minutes
                logger.info(f"[WAIT] Waiting {wait_time}s before restart...")
                time.sleep(wait_time)

        except KeyboardInterrupt:
            logger.info("\n[STOP] Shutdown requested by user")
            return False
            
        except Exception as e:
            retry_count += 1
            logger.error(f"[ERROR] Critical crash: {e}")
            
            if retry_count < max_retries:
                wait_time = min(30 * retry_count, 300)
                logger.info(f"[RETRY] Restarting in {wait_time}s (attempt {retry_count + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                logger.critical("[FATAL] MAX RETRIES REACHED! Manual intervention required.")
                return False
    
    logger.critical("[FATAL] System stopped after exhausting retry attempts")
    return False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"\n🛑 Received signal {signum} - initiating graceful shutdown")
    sys.exit(0)


def kill_orphaned_chrome_processes():
    """
    Force kill any lingering Chrome/Playwright processes
    to prevent RAM leaks and zombie processes.
    """
    import subprocess
    logger.info("🧹 [CLEANUP] Scanning for orphaned browser processes...")
    try:
        # Windows-specific process kill
        if os.name == 'nt':
            subprocess.run("taskkill /F /IM chrome.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run("taskkill /F /IM msedge.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run("taskkill /F /IM playwright.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            # Linux/Mac
            subprocess.run("pkill -f chrome", shell=True)
            subprocess.run("pkill -f msedge", shell=True)
            
        logger.info("✅ [CLEANUP] Orphaned processes terminated.")
    except Exception as e:
        logger.warning(f"⚠️ [CLEANUP] Warning during cleanup: {e}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("""
    ==============================================================
    ||                                                          ||
    ||     ELITE SNIPER V3 - STRICT BEHAVIORAL MODE             ||
    ||                                                          ||
    ||     Status: LAUNCHING DIRECTLY                           ||
    ||     Note: Telegram Bot disabled for V3 Strict Run        ||
    ||                                                          ||
    ==============================================================
    """)
    
    # 1. Startup Cleanup (Anti-Zombie)
    kill_orphaned_chrome_processes()
    
    run_elite_sniper_v2()
