from picamera2 import Picamera2
import cv2
import numpy as np
from PIL import Image
import time

# ==========================
# ADJUSTABLE PARAMETERS
# ==========================

EXPOSURE = 1.05       # >1 brighter, <1 darker
CONTRAST = 1.1
SATURATION = 1.15

VIGNETTE_STRENGTH = 0.4   # 0 = none, 0.5 = heavy
COLOR_BLEED = 2           # pixels of channel offset
S_CURVE_STRENGTH = 0.8    # 0–1

OUTPUT_FILE = "filmic_output.jpg"

# ==========================
# CAMERA CAPTURE
# ==========================

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(2)

frame = picam2.capture_array()
picam2.stop()

img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ==========================
# BASIC LEVELS
# ==========================

img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=0)
img = np.clip(img * EXPOSURE, 0, 255).astype(np.uint8)

# ==========================
# SATURATION
# ==========================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] *= SATURATION
hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ==========================
# FILMIC S-CURVE
# ==========================

def s_curve(channel, strength=0.8):
    x = channel / 255.0
    curve = (x - 0.5) * (1 + strength) + 0.5
    curve = np.clip(curve, 0, 1)
    return (curve * 255).astype(np.uint8)

for c in range(3):
    img[:, :, c] = s_curve(img[:, :, c], S_CURVE_STRENGTH)

# ==========================
# COLOR BLEED
# ==========================

b, g, r = cv2.split(img)

r_shift = np.roll(r, COLOR_BLEED, axis=1)
b_shift = np.roll(b, -COLOR_BLEED, axis=0)

r_shift = cv2.GaussianBlur(r_shift, (5, 5), 0)
b_shift = cv2.GaussianBlur(b_shift, (5, 5), 0)

img = cv2.merge([b_shift, g, r_shift])

# ==========================
# VIGNETTE
# ==========================

rows, cols = img.shape[:2]
X = cv2.getGaussianKernel(cols, cols * VIGNETTE_STRENGTH)
Y = cv2.getGaussianKernel(rows, rows * VIGNETTE_STRENGTH)
vignette = Y @ X.T
vignette = vignette / vignette.max()

for i in range(3):
    img[:, :, i] = img[:, :, i] * vignette

# ==========================
# SAVE
# ==========================

cv2.imwrite(OUTPUT_FILE, img)
print(f"Saved {OUTPUT_FILE}")
