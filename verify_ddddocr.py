import ddddocr
import sys

try:
    ocr = ddddocr.DdddOcr(beta=True)
    print("OCR initialized with beta=True")
    print(f"Type: {type(ocr)}")
    print(f"Methods: {dir(ocr)}")
    
    # Create dummy image bytes (1x1 black pixel png)
    # import base64
    # dummy_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    
    # try:
    #     res = ocr.classification(dummy_png)
    #     print(f"classification result: {res}")
    # except AttributeError:
    #     print("classification method missing")
    
    # try:
    #     res = ocr.predict(dummy_png)
    #     print(f"predict result: {res}")
    # except AttributeError:
    #     print("predict method missing")
        
except Exception as e:
    print(f"Error: {e}")
