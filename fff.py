from picamera2 import Picamera2
from PIL import Image, ImageEnhance
import numpy as np
import time
import os
from datetime import datetime

# ================= CONFIG =================
OUTPUT_DIR = "/home/pi/Pictures"
RESOLUTION = (1024, 768)
COUNTDOWN = 3

# White balance tuning (CRITICAL)
RED_GAIN = 1.6
BLUE_GAIN = 1.2

GRAIN_STRENGTH = 8
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": RESOLUTION, "format": "RGB888"},
    buffer_count=1
)
picam2.configure(config)
picam2.start()

# Lock white balance (THIS fixes the blue problem)
picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (RED_GAIN, BLUE_GAIN)
})

time.sleep(2)
print("Camera ready.")
print("Press ENTER to take a photo. Type 'q' then ENTER to quit.")

def apply_fuji_style(img: Image.Image) -> Image.Image:
    # Slight contrast fade
    img = ImageEnhance.Contrast(img).enhance(0.95)

    # Slight brightness lift
    img = ImageEnhance.Brightness(img).enhance(1.05)

    # Gentle saturation
    img = ImageEnhance.Color(img).enhance(1.1)

    # Split channels
    r, g, b = img.split()

    # Fuji-style color bias
    r = r.point(lambda i: min(255, i * 1.05 + 2))
    g = g.point(lambda i: min(255, i * 1.03 + 1))
    b = b.point(lambda i: i * 0.95)

    img = Image.merge("RGB", (r, g, b))

    # Film grain
    arr = np.array(img).astype(np.int16)
    grain = np.random.randint(-GRAIN_STRENGTH, GRAIN_STRENGTH, arr.shape)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Vignette
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    mask = 1 - 0.4 * (dist / max_dist)
    mask = np.clip(mask, 0.6, 1.0)

    arr = (arr * mask[..., None]).astype(np.uint8)
    return Image.fromarray(arr)

while True:
    cmd = input("Ready: ")
    if cmd.lower() == "q":
        break

    print("Hold still...")
    for i in range(COUNTDOWN, 0, -1):
        print(i)
        time.sleep(1)

    frame = picam2.capture_array()
    print("Photo captured.")

    img = Image.fromarray(frame)
    img = apply_fuji_style(img)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fuji_{timestamp}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)

    img.save(path, quality=92, subsampling=0)

    # Force disk write
    os.sync()

    print(f"Saved: {path}")
    print("You can move the camera now.\n")

print("Done.")
picam2.stop()
