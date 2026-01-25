from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
from datetime import datetime

# ===== CONFIG =====
OUTPUT_DIR = "/home/pi/Pictures"
RESOLUTION = (1024, 768)   # Safe for Pi Zero 2 W
FRAME_RATE = 5
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": RESOLUTION, "format": "RGB888"},
    buffer_count=1
)
picam2.configure(config)

picam2.start()
picam2.set_controls({"FrameRate": FRAME_RATE})
time.sleep(2)

# Capture frame
frame = picam2.capture_array()
picam2.stop()

# Convert RGB → BGR (OpenCV expects BGR)
img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ===== FILM LOOK =====

# Lift blacks + soften contrast (cheap Fuji fade)
img = cv2.convertScaleAbs(img, alpha=0.95, beta=12)

# Color bias (Fuji-ish)
b, g, r = cv2.split(img)
r = cv2.convertScaleAbs(r, alpha=1.05, beta=2)
g = cv2.convertScaleAbs(g, alpha=1.03, beta=1)
b = cv2.convertScaleAbs(b, alpha=0.95, beta=0)
img = cv2.merge([b, g, r])

# Add grain (memory-safe)
grain = np.random.randint(-5, 5, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

# Vignette (FIXED SHAPE BUG)
rows, cols = img.shape[:2]
kernel_x = cv2.getGaussianKernel(cols, cols / 1.8)
kernel_y = cv2.getGaussianKernel(rows, rows / 1.8)
mask = kernel_y @ kernel_x.T
mask = mask / mask.max()

img = (img * mask[..., None]).astype(np.uint8)

# ===== SAVE IMAGE =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"fuji_{timestamp}.jpg"
path = os.path.join(OUTPUT_DIR, filename)

cv2.imwrite(path, img)

print(f"Saved image to {path}")
