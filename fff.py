from picamera2 import Picamera2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time
import os
from datetime import datetime

# ===== CONFIG =====
OUTPUT_DIR = "/home/pi/Pictures"
RESOLUTION = (1024, 768)
COUNTDOWN = 3  # seconds
GRAIN_LEVEL = 10  # intensity of grain
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
time.sleep(2)

print("Camera ready! Press Enter to take a photo. Type 'q' + Enter to quit.")

def apply_fuji_style(img: Image.Image) -> Image.Image:
    """Apply vintage Fuji-style effect using PIL."""
    # Slight contrast reduction (fade blacks)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.95)

    # Slight brightness boost
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)

    # Split channels
    r, g, b = img.split()

    # Fuji tone adjustments
    r = r.point(lambda i: min(255, i * 1.08 + 3))  # warm highlights
    g = g.point(lambda i: min(255, i * 1.05 + 2))  # green midtones
    b = b.point(lambda i: i * 0.92)                # cool shadows

    img = Image.merge("RGB", (r, g, b))

    # Add grain
    arr = np.array(img).astype(np.int16)
    grain = np.random.randint(-GRAIN_LEVEL, GRAIN_LEVEL, arr.shape, dtype=np.int16)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Vignette
    rows, cols = arr.shape[:2]
    y, x = np.ogrid[:rows, :cols]
    center_y, center_x = rows / 2, cols / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    mask = 1 - 0.5 * (distance / max_distance)  # strength of vignette
    mask = np.clip(mask, 0.5, 1)
    arr = np.array(img)
    arr = (arr * mask[..., None]).astype(np.uint8)
    img = Image.fromarray(arr)

    return img

while True:
    user_input = input("Take a photo? ")
    if user_input.lower() == 'q':
        break

    # Countdown
    for i in range(COUNTDOWN, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Capture frame
    img_array = picam2.capture_array()
    img = Image.fromarray(img_array)
    print("Photo taken!")

    # Apply Fuji effect
    img = apply_fuji_style(img)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fuji_{timestamp}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path)
    print(f"Saved image to {path}")

print("Script finished. You can safely move the camera.")
picam2.stop()
