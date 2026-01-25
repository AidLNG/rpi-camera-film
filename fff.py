from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
from datetime import datetime

# ===== CONFIG =====
OUTPUT_DIR = "/home/pi/Pictures"
RESOLUTION = (1024, 768)
FRAME_RATE = 5
COUNTDOWN = 3  # seconds before snap
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

print("Camera ready! Press Enter to take a photo. Type 'q' + Enter to quit.")

while True:
    user_input = input("Take a photo? ")
    if user_input.lower() == 'q':
        break

    # Countdown
    for i in range(COUNTDOWN, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Capture frame (RGB)
    img = picam2.capture_array()
    print("Photo taken!")

    # ===== FILM LOOK (RGB SPACE!) =====
    # Lift blacks + soften contrast
    img = cv2.convertScaleAbs(img, alpha=0.95, beta=10)

    # Fujifilm tone mapping (RGB order)
    r, g, b = cv2.split(img)
    r = cv2.convertScaleAbs(r, alpha=1.08, beta=3)  # warm highlights
    g = cv2.convertScaleAbs(g, alpha=1.05, beta=2)  # green midtones
    b = cv2.convertScaleAbs(b, alpha=0.92, beta=0)  # cool shadows
    img = cv2.merge([r, g, b])

    # Subtle grain
    grain = np.random.randint(-6, 6, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Vignette
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 1.8)
    kernel_y = cv2.getGaussianKernel(rows, rows / 1.8)
    mask = kernel_y @ kernel_x.T
    mask = mask / mask.max()
    img = (img * mask[..., None]).astype(np.uint8)

    # ===== SAVE IMAGE =====
    # Convert RGB -> BGR *only once* for OpenCV saving
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fuji_{timestamp}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)

    if cv2.imwrite(path, img_bgr):
        print(f"Saved image to {path}")
    else:
        print("Failed to save image!")

print("Script finished. You can safely move the camera.")
picam2.stop()
