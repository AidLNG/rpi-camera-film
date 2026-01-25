#!/usr/bin/env python3
"""
Arducam Fujifilm Film Look - Optimized for Raspberry Pi Zero 2W
Extreme vintage film aesthetic with interactive capture mode
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from datetime import datetime
import time
import gc
import sys
import tty
import termios

class FujiFilmProcessor:
    def __init__(self, resolution=(1920, 1080)):
        """
        Initialize with lower resolution for Pi Zero 2W
        resolution: tuple (width, height) - lower values use less memory
        """
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.photo_count = 0
        
    def setup_camera(self):
        """Configure camera with memory-efficient settings"""
        config = self.picam2.create_still_configuration(
            main={"size": self.resolution},
            buffer_count=2  # Minimize buffer usage
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        
    def apply_extreme_vintage_colors(self, img):
        """Apply unrealistically strong vintage color grading"""
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        
        # Extreme color shifts for that "too vintage" look
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Strong green shift (classic Fuji characteristic, exaggerated)
        lab[:, :, 1] -= 12
        
        # Heavy warm/yellow cast
        lab[:, :, 2] += 15
        
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        del lab
        gc.collect()
        
        # Exaggerated cyan shadows
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        shadow_mask = hsv[:, :, 2] < 0.4
        hsv[shadow_mask, 0] = np.clip(hsv[shadow_mask, 0] + 0.15, 0, 1)
        
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        del hsv, shadow_mask
        gc.collect()
        
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8, copy=False)
    
    def apply_extreme_film_curve(self, img):
        """Apply aggressive S-curve with heavily lifted blacks for faded look"""
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        
        # Much stronger black lift for that washed-out vintage look
        black_lift = 0.15  # Was 0.08, now much more faded
        contrast = 0.75    # Reduced contrast for softer, vintage feel
        
        np.power(img, contrast, out=img)
        img *= (1 - black_lift)
        img += black_lift
        
        # Compress highlights slightly
        highlight_mask = img > 0.7
        img[highlight_mask] = 0.7 + (img[highlight_mask] - 0.7) * 0.6
        
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8, copy=False)
    
    def add_heavy_grain(self, img, intensity=0.025):
        """Add prominent film grain for authentic vintage texture"""
        h, w = img.shape[:2]
        
        # Generate coarser, more visible grain
        chunk_size = 256
        img_float = img.astype(np.float32, copy=False)
        
        for i in range(0, h, chunk_size):
            for j in range(0, w, chunk_size):
                end_i = min(i + chunk_size, h)
                end_j = min(j + chunk_size, w)
                
                # Add both fine and coarse grain
                fine_grain = np.random.normal(0, intensity * 255, 
                                            (end_i - i, end_j - j, 3)).astype(np.float32)
                coarse_grain = np.random.normal(0, intensity * 150,
                                              (end_i - i, end_j - j, 3)).astype(np.float32)
                
                img_float[i:end_i, j:end_j] += fine_grain + coarse_grain
                
        np.clip(img_float, 0, 255, out=img_float)
        return img_float.astype(np.uint8, copy=False)
    
    def add_extreme_expired_effects(self, img, strength=0.8):
        """Heavy expired film effects - light leaks, color shifts, vignette"""
        h, w = img.shape[:2]
        
        # Random strong color shifts
        shift_type = np.random.random()
        if shift_type > 0.66:
            # Heavy magenta shift
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + int(strength * 20), 0, 255).astype(np.uint8)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + int(strength * 15), 0, 255).astype(np.uint8)
        elif shift_type > 0.33:
            # Heavy yellow/orange shift
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + int(strength * 25), 0, 255).astype(np.uint8)
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + int(strength * 18), 0, 255).astype(np.uint8)
        else:
            # Green/cyan shift
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + int(strength * 20), 0, 255).astype(np.uint8)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + int(strength * 12), 0, 255).astype(np.uint8)
        
        # Mild vignette
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        
        r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        # Gentle, subtle vignette
        vignette = 1 - (strength * 0.15 * np.clip(r - 0.6, 0, 1.2))
        
        img_float = img.astype(np.float32, copy=False)
        img_float *= vignette[:, :, np.newaxis]
        
        # Add random light leak in corner
        if np.random.random() > 0.5:
            corner = np.random.choice(['tl', 'tr', 'bl', 'br'])
            leak_strength = strength * 80
            
            if corner == 'tl':
                img_float[0:h//3, 0:w//3] += leak_strength
            elif corner == 'tr':
                img_float[0:h//3, 2*w//3:w] += leak_strength
            elif corner == 'bl':
                img_float[2*h//3:h, 0:w//3] += leak_strength
            else:
                img_float[2*h//3:h, 2*w//3:w] += leak_strength
        
        del r, vignette
        gc.collect()
        
        np.clip(img_float, 0, 255, out=img_float)
        return img_float.astype(np.uint8, copy=False)
    
    def reduce_saturation_extreme(self, img, factor=0.65):
        """Heavy desaturation for authentic vintage look"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * factor).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        del hsv
        gc.collect()
        return img
    
    def process_image(self, img, expired_strength=0.7):
        """
        Process with extreme vintage film aesthetic
        """
        print("Processing: Vintage colors...", end='', flush=True)
        img = self.apply_extreme_vintage_colors(img)
        
        print(" faded curve...", end='', flush=True)
        img = self.apply_extreme_film_curve(img)
        
        print(" desaturation...", end='', flush=True)
        img = self.reduce_saturation_extreme(img, 0.65)
        
        print(" heavy grain...", end='', flush=True)
        img = self.add_heavy_grain(img, intensity=0.025)
        
        print(" expired effects...", end='', flush=True)
        img = self.add_extreme_expired_effects(img, strength=expired_strength)
        
        # Final warm glow
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=5)
        
        print(" done!")
        return img
    
    def capture_photo(self, expired_strength=0.7, output_path=None):
        """Capture and process a photo"""
        self.photo_count += 1
        print(f"\n📷 Capturing photo #{self.photo_count}...")
        
        # Capture directly to array
        img = self.picam2.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Process
        processed = self.process_image(img, expired_strength)
        
        # Free original image memory
        del img
        gc.collect()
        
        # Save
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"fujifilm_{timestamp}.jpg"
        
        print(f"Saving to {output_path}...")
        cv2.imwrite(output_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Free processed image
        del processed
        gc.collect()
        
        print(f"✓ Photo #{self.photo_count} saved successfully: {output_path}\n")
        return output_path
    
    def cleanup(self):
        """Clean up resources"""
        self.picam2.stop()
        gc.collect()

def get_key():
    """Get a single keypress without waiting for Enter"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def interactive_mode():
    """Interactive photo capture mode"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   Fujifilm Vintage Film Camera (Pi Zero 2W Optimized)    ║")
    print("║              EXTREME VINTAGE AESTHETIC MODE               ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Resolution selection
    print("Select resolution:")
    print("  1 - 1280x720  (Low memory, faster)")
    print("  2 - 1920x1080 (Balanced, recommended)")
    print("  3 - 2592x1944 (High quality, needs more memory)")
    print("\nChoice (1-3): ", end='', flush=True)
    
    choice = get_key()
    print(choice)
    
    resolutions = {
        '1': (1280, 720),
        '2': (1920, 1080),
        '3': (2592, 1944)
    }
    resolution = resolutions.get(choice, (1920, 1080))
    
    print(f"\nUsing resolution: {resolution[0]}x{resolution[1]}")
    
    # Expired strength selection
    print("\nSelect vintage intensity:")
    print("  1 - Mild vintage (0.4)")
    print("  2 - Strong vintage (0.7) [RECOMMENDED]")
    print("  3 - EXTREME vintage (1.0)")
    print("\nChoice (1-3): ", end='', flush=True)
    
    choice = get_key()
    print(choice)
    
    strengths = {'1': 0.4, '2': 0.7, '3': 1.0}
    expired_strength = strengths.get(choice, 0.7)
    
    processor = FujiFilmProcessor(resolution=resolution)
    
    print("\nSetting up camera...")
    processor.setup_camera()
    
    print("\n" + "="*60)
    print("READY TO SHOOT!")
    print("="*60)
    print("\nControls:")
    print("  • Press ENTER to take a photo")
    print("  • Press 'q' to quit\n")
    
    try:
        while True:
            print("Waiting for input... (ENTER = capture, q = quit)")
            
            key = get_key()
            
            if key == 'q':
                print("\nQuitting...")
                break
            elif key == '\r' or key == '\n':  # Enter key
                processor.capture_photo(expired_strength=expired_strength)
            else:
                print(f"Unknown key. Press ENTER to capture or 'q' to quit.")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nCleaning up...")
        processor.cleanup()
        print(f"Session complete. {processor.photo_count} photo(s) captured.")
        print("Done!")

def main():
    interactive_mode()

if __name__ == "__main__":
    main()
