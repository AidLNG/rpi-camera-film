from picamera2 import Picamera2
import cv2
import numpy as np
import time

# ==========================
# ADJUSTABLE CONTROLS
# ==========================

EXPOSURE = 1.0     # brightness
CONTRAST = 0.8       # 1.0 = neutral
SATURATION = 1.10   # 1.0 = neutral

S_CURVE = 0.9        # <1 = softer highlights
VIGNETTE = 0.2       # 0–0.6 safe
COLOR_BLEED = 0.5      # pixels

WIDTH, HEIGHT = 1920, 1080
OUTPUT = "filmic_output.jpg"

# ==========================
# CAMERA
# ==========================

picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (WIDTH, HEIGHT)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

frame = picam2.capture_array()
picam2.stop()

# RGBA safety
if frame.shape[2] == 4:
    img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
else:
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

img = img.astype(np.float32) / 255.0

# ==========================
# EXPOSURE + CONTRAST
# ==========================

img = img * EXPOSURE
img = (img - 0.5) * CONTRAST + 0.5

# ==========================
# SATURATION (LUMA-BASED)
# ==========================

luma = (
    0.2126 * img[:, :, 2] +
    0.7152 * img[:, :, 1] +
    0.0722 * img[:, :, 0]
)

img = luma[:, :, None] + SATURATION * (img - luma[:, :, None])

# ==========================
# FILMIC S-CURVE
# ==========================

img = np.clip(img, 0, 1)
img = img ** S_CURVE

# ==========================
# COLOR BLEED (SAFE)
# ==========================

img[:, :, 2] = np.roll(img[:, :, 2], COLOR_BLEED, axis=1)
img[:, :, 0] = np.roll(img[:, :, 0], -COLOR_BLEED, axis=0)

# ==========================
# VIGNETTE
# ==========================

rows, cols = img.shape[:2]
X = np.linspace(-1, 1, cols)
Y = np.linspace(-1, 1, rows)
xv, yv = np.meshgrid(X, Y)

v = 1 - VIGNETTE * (xv**2 + yv**2)
v = np.clip(v, 0.6, 1)

img *= v[:, :, None]

# ==========================
# SAVE
# ==========================

img = np.clip(img * 255, 0, 255).astype(np.uint8)
cv2.imwrite(OUTPUT, img)

print("Saved", OUTPUT)
