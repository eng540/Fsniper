
import re
import sys

# Get log path from args or default
log_path = sys.argv[1] if len(sys.argv) > 1 else r"d:\ai\sniper\logs.1770035346930.log.txt"

stats = {
    # CapSolver Stats
    "cs_attempts": 0,
    "cs_success_raw": 0,
    "cs_success_enhanced": 0,
    "cs_failed": 0,
    "cs_retried": 0,
    
    # Local OCR Stats
    "local_attempts": 0, # Implied by "Trying local ddddocr"
    "local_success": 0,
    "local_failed": 0,
    
    # General
    "black_captcha": 0
}

try:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # --- CapSolver Scans ---
            if "Sending request to CapSolver" in line:
                if "_RETRY]" not in line:
                     stats["cs_attempts"] += 1
            
            if "CapSolver result too short" in line:
                stats["cs_retried"] += 1

            if "Using CapSolver result" in line:
                stats["cs_success_raw"] += 1
            
            if "CapSolver (Enhanced) result" in line:
                stats["cs_success_enhanced"] += 1
            
            if "CapSolver chain failed" in line:
                stats["cs_failed"] += 1

            # --- Local OCR Scans ---
            if "Trying local ddddocr" in line:
                stats["local_attempts"] += 1
            
            if "Local OCR solved" in line:
                stats["local_success"] += 1
            
            if "Local OCR failed" in line:
                stats["local_failed"] += 1

            # --- General ---
            if "BLACK CAPTCHA" in line:
                stats["black_captcha"] += 1

    print(f"--- Analysis for {log_path} ---")
    
    if stats['cs_attempts'] > 0:
        print(f"\n[CapSolver Stats]")
        print(f"Total Attempts: {stats['cs_attempts']}")
        print(f"Success Raw: {stats['cs_success_raw']}")
        print(f"Success Enhanced: {stats['cs_success_enhanced']}")
        print(f"Total Success: {stats['cs_success_raw'] + stats['cs_success_enhanced']}")
        print(f"Retries Triggered: {stats['cs_retried']}")
        print(f"Chain Failures: {stats['cs_failed']}")
    
    if stats['local_attempts'] > 0:
        print(f"\n[Local OCR Stats]")
        print(f"Total Attempts: {stats['local_attempts']}")
        print(f"Success: {stats['local_success']}")
        print(f"Failed: {stats['local_failed']}")
        if stats['local_attempts'] > 0:
             print(f"Success Rate: {(stats['local_success'] / stats['local_attempts']) * 100:.2f}%")

    print(f"\n[Errors]")
    print(f"Black Captchas Detected: {stats['black_captcha']}")

except Exception as e:
    print(f"Error: {e}")
