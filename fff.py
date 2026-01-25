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

# Prompt user to hold camera steady
input("Camera ready. Press Enter to take the photo...")

# Capture frame (RGB)
img = picam2.capture_array()
print("Photo taken!")  # immediate feedback
picam2.stop()

# ===== FILM LOOK (RGB SPACE) =====

# Lift blacks + soften contrast
img = cv2.convertScaleAbs(img, alpha=0.95, beta=12)

# Fujifilm-style color bias (RGB order!)
r, g, b = cv2.split(img)
r = cv2.convertScaleAbs(r, alpha=1.05, beta=2)  # warm highlights
g = cv2.convertScaleAbs(g, alpha=1.03, beta=1)  # green bias
b = cv2.convertScaleAbs(b, alpha=0.95, beta=0)  # cool shadows
img = cv2.merge([r, g, b])

# Add grain (memory-safe)
grain = np.random.randint(-5, 5, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

# Vignette (correct dimensions)
rows, cols = img.shape[:2]
kernel_x = cv2.getGaussianKernel(cols, cols / 1.8)
kernel_y = cv2.getGaussianKernel(rows, rows / 1.8)
mask = kernel_y @ kernel_x.T
mask = mask / mask.max()
img = (img * mask[..., None]).astype(np.uint8)

# Convert RGB → BGR for saving
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ===== SAVE IMAGE =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"fuji_{timestamp}.jpg"
path = os.path.join(OUTPUT_DIR, filename)
cv2.imwrite(path, img)

print(f"Saved image to {path}")
print("Script finished! You can move the camera now.")
