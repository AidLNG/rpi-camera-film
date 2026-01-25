from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
from datetime import datetime

# Output directory
OUTPUT_DIR = "/home/pi/Pictures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (2592, 1944), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Let sensor settle
time.sleep(2)

# Capture image
frame = picam2.capture_array()
picam2.stop()

# Convert RGB → BGR for OpenCV
img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# --- FILM LOOK PROCESSING ---

# Slight fade (lift blacks)
img = img.astype(np.float32)
img = img * 0.95 + 10

# Split channels
b, g, r = cv2.split(img)

# Fujifilm-ish color bias
r *= 1.05      # warm highlights
g *= 1.02      # subtle green bias
b *= 0.95      # cooler shadows

img = cv2.merge([b, g, r])

# Contrast curve (soft S-curve)
def s_curve(channel):
    x = channel / 255.0
    return np.clip((x - 0.5) * 0.9 + 0.5, 0, 1) * 255

img = np.stack([s_curve(c) for c in cv2.split(img)], axis=-1)

# Add grain
grain = np.random.normal(0, 6, img.shape).astype(np.float32)
img += grain

# Slight vignette
rows, cols = img.shape[:2]
X_kernel = cv2.getGaussianKernel(cols, cols / 2)
Y_kernel = cv2.getGaussianKernel(rows, rows / 2)
kernel = Y_kernel * X_kernel.T
vignette = kernel / kernel.max()
img *= vignette[..., np.newaxis]

# Final clamp
img = np.clip(img, 0, 255).astype(np.uint8)

# Save image
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"fuji_{timestamp}.jpg"
path = os.path.join(OUTPUT_DIR, filename)

cv2.imwrite(path, img)

print(f"Saved film-style image to {path}")
