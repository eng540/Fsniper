from src.config import Config
print(f"EMAIL: {Config.EMAIL}")
print(f"PASSPORT: {Config.PASSPORT}")
if Config.EMAIL == "waffaron@gmail.com":
    print("SUCCESS: Config loaded correctly")
else:
    print("FAILURE: Config did not load correctly")
