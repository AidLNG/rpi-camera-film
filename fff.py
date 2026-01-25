from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
from datetime import datetime

OUTPUT_DIR = "/home/pi/Pictures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (1024, 768), "format": "RGB888"},
    buffer_count=1
)
picam2.configure(config)

picam2.start()
picam2.set_controls({"FrameRate": 5})
time.sleep(2)

frame = picam2.capture_array()
picam2.stop()

# Convert RGB → BGR
img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Slight fade (lift blacks, reduce contrast)
img = cv2.convertScaleAbs(img, alpha=0.95, beta=12)

# Split channels (uint8 safe)
b, g, r = cv2.split(img)

# Fujifilm-ish color bias
r = cv2.convertScaleAbs(r, alpha=1.05, beta=2)
g = cv2.convertScaleAbs(g, alpha=1.03, beta=1)
b = cv2.convertScaleAbs(b, alpha=0.95, beta=0)

img = cv2.merge([b, g, r])

# Cheap grain (int16 temp, small)
grain = np.random.randint(-5, 5, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

# Lightweight vignette
rows, cols = img.shape[:2]
mask = cv2.getGaussianKernel(cols, cols / 1.8)
mask = mask * mask.T
mask = mask / mask.max()
img = (img * mask[..., None]).astype(np.uint8)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"{OUTPUT_DIR}/fuji_{timestamp}.jpg"
cv2.imwrite(path, img)

print(f"Saved {path}")
