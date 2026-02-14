
import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

try:
    # Treat src as a package
    from src.config import Config
    from src.captcha import EnhancedCaptchaSolver, CapSolverHandler, CircuitBreaker
    print("[SUCCESS] Imports successful")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def verify_config():
    print("\n--- Verifying Configuration ---")
    val_threshold = getattr(Config, 'CIRCUIT_BREAKER_THRESHOLD', 'MISSING')
    print(f"CIRCUIT_BREAKER_THRESHOLD: {val_threshold}")
    val_timeout = getattr(Config, 'CIRCUIT_BREAKER_TIMEOUT', 'MISSING')
    print(f"CIRCUIT_BREAKER_TIMEOUT: {val_timeout}")
    val_parallel = getattr(Config, 'PARALLEL_SOLVING_ENABLED', 'MISSING')
    print(f"PARALLEL_SOLVING_ENABLED: {val_parallel}")
    
    if val_threshold != 'MISSING' and val_parallel != 'MISSING':
        print("[SUCCESS] Config updated correctly")
    else:
        print("[FAIL] Missing config values")

def verify_circuit_breaker():
    print("\n--- Verifying Circuit Breaker ---")
    cb = CircuitBreaker(threshold=2, timeout=1)
    
    print(f"Initial state: {cb.state} (Failures: {cb.failures})")
    
    # Simulate failures
    cb.record_failure()
    print(f"After 1 failure: {cb.state} (Failures: {cb.failures})")
    
    cb.record_failure()
    print(f"After 2 failures: {cb.state} (Failures: {cb.failures})")
    
    # Due to float timing, check state directly
    if cb.state == "OPEN":
        print("[SUCCESS] Circuit is OPEN after threshold reached")
    else:
        print(f"[FAIL] Circuit should be OPEN, but is {cb.state}")
        
    # Simulate timeout
    print("Waiting for timeout (1.1s)...")
    time.sleep(1.1)
    
    is_open = cb.is_open()
    print(f"After timeout check: {cb.state} (Is Open? {is_open})")
    
    if cb.state == "HALF-OPEN" or not is_open:
         print(f"[SUCCESS] Circuit transitioned correctly (State: {cb.state})")
    else:
         print(f"[FAIL] Circuit stuck OPEN (State: {cb.state})")

if __name__ == "__main__":
    verify_config()
    verify_circuit_breaker()
