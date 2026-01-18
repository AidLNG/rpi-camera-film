from picamera2 import Picamera2
import cv2
import numpy as np
import time

# ---------- SETTINGS ----------
WIDTH, HEIGHT = 1280, 720
OUTPUT = "filmic_safe.jpg"

# ---------- CAMERA ----------
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (WIDTH, HEIGHT)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

frame = picam2.capture_array()
picam2.stop()

# Handle RGBA safely
if frame.shape[2] == 4:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
else:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

img = frame.astype(np.float32) / 255.0

# ---------- FILMIC S-CURVE ----------
img = img ** 0.9  # gentle highlight rolloff

# ---------- COLOR BLEED (SAFE) ----------
img[:, :, 2] = np.roll(img[:, :, 2], 1, axis=1)
img[:, :, 0] = np.roll(img[:, :, 0], -1, axis=0)

# ---------- VIGNETTE ----------
rows, cols = img.shape[:2]
X = np.linspace(-1, 1, cols)
Y = np.linspace(-1, 1, rows)
xv, yv = np.meshgrid(X, Y)
vignette = 1 - 0.4 * (xv**2 + yv**2)
vignette = np.clip(vignette, 0.6, 1)

img *= vignette[:, :, None]

# ---------- SAVE ----------
img = np.clip(img * 255, 0, 255).astype(np.uint8)
cv2.imwrite(OUTPUT, img)

print("Saved", OUTPUT)
