import sys

modules = ['playwright', 'ddddocr', 'dotenv', 'requests', 'cv2', 'numpy', 'pytz', 'ntplib']
missing = []
for m in modules:
    try:
        if m == 'cv2':
            import cv2
        elif m == 'dotenv':
            import dotenv
        else:
            __import__(m)
        print(f"{m}: OK")
    except ImportError:
        missing.append(m)
        print(f"{m}: MISSING")

if missing:
    print(f"MISSING_MODULES={','.join(missing)}")
    sys.exit(1)
else:
    print("ALL_OK")
