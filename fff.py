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
        """Apply subtle film-like color grading"""
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        
        # Gentle color shifts for authentic film look
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Subtle green shift (classic Fuji characteristic)
        lab[:, :, 1] -= 6
        
        # Mild warm/yellow cast
        lab[:, :, 2] += 8
        
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        del lab
        gc.collect()
        
        # Gentle cyan tint in shadows
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        shadow_mask = hsv[:, :, 2] < 0.3
        hsv[shadow_mask, 0] = np.clip(hsv[shadow_mask, 0] + 0.06, 0, 1)
        
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        del hsv, shadow_mask
        gc.collect()
        
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8, copy=False)
    
    def apply_extreme_film_curve(self, img):
        """Apply moderate S-curve with lifted blacks for film look"""
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        
        # Moderate black lift for subtle faded look
        black_lift = 0.10
        contrast = 0.82
        
        np.power(img, contrast, out=img)
        img *= (1 - black_lift)
        img += black_lift
        
        # Gentle highlight compression
        highlight_mask = img > 0.75
        img[highlight_mask] = 0.75 + (img[highlight_mask] - 0.75) * 0.7
        
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8, copy=False)
    
    def add_heavy_grain(self, img, intensity=0.015):
        """Add subtle film grain for authentic texture"""
        h, w = img.shape[:2]
        
        # Generate subtle, realistic grain
        chunk_size = 256
        img_float = img.astype(np.float32, copy=False)
        
        for i in range(0, h, chunk_size):
            for j in range(0, w, chunk_size):
                end_i = min(i + chunk_size, h)
                end_j = min(j + chunk_size, w)
                
                # Just fine grain, no coarse layer
                fine_grain = np.random.normal(0, intensity * 255, 
                                            (end_i - i, end_j - j, 3)).astype(np.float32)
                
                img_float[i:end_i, j:end_j] += fine_grain
                
        np.clip(img_float, 0, 255, out=img_float)
        return img_float.astype(np.uint8, copy=False)
    
    def add_extreme_expired_effects(self, img, strength=0.5):
        """Subtle expired film effects - gentle color shifts and light vignette"""
        h, w = img.shape[:2]
        
        # Gentle random color shifts
        shift_type = np.random.random()
        if shift_type > 0.66:
            # Mild magenta shift
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + int(strength * 8), 0, 255).astype(np.uint8)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + int(strength * 6), 0, 255).astype(np.uint8)
        elif shift_type > 0.33:
            # Mild yellow/warm shift
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + int(strength * 10), 0, 255).astype(np.uint8)
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + int(strength * 7), 0, 255).astype(np.uint8)
        else:
            # Mild green/cyan shift
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + int(strength * 8), 0, 255).astype(np.uint8)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + int(strength * 5), 0, 255).astype(np.uint8)
        
        # Very mild vignette - barely noticeable
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        
        r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        # Extremely subtle vignette
        vignette = 1 - (strength * 0.08 * np.clip(r - 0.7, 0, 1.0))
        
        img_float = img.astype(np.float32, copy=False)
        img_float *= vignette[:, :, np.newaxis]
        
        # Very rare, subtle light leak
        if np.random.random() > 0.8:  # Only 20% chance
            corner = np.random.choice(['tl', 'tr', 'bl', 'br'])
            leak_strength = strength * 30  # Much subtler
            
            if corner == 'tl':
                img_float[0:h//4, 0:w//4] += leak_strength
            elif corner == 'tr':
                img_float[0:h//4, 3*w//4:w] += leak_strength
            elif corner == 'bl':
                img_float[3*h//4:h, 0:w//4] += leak_strength
            else:
                img_float[3*h//4:h, 3*w//4:w] += leak_strength
        
        del r, vignette
        gc.collect()
        
        np.clip(img_float, 0, 255, out=img_float)
        return img_float.astype(np.uint8, copy=False)
    
    def reduce_saturation_extreme(self, img, factor=0.78):
        """Moderate desaturation for subtle vintage look"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * factor).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        del hsv
        gc.collect()
        return img
    
    def process_image(self, img, expired_strength=0.5):
        """
        Process with subtle vintage film aesthetic
        """
        print("Processing: Film colors...", end='', flush=True)
        img = self.apply_extreme_vintage_colors(img)
        
        print(" film curve...", end='', flush=True)
        img = self.apply_extreme_film_curve(img)
        
        print(" saturation...", end='', flush=True)
        img = self.reduce_saturation_extreme(img, 0.78)
        
        print(" grain...", end='', flush=True)
        img = self.add_heavy_grain(img, intensity=0.015)
        
        print(" film effects...", end='', flush=True)
        img = self.add_extreme_expired_effects(img, strength=expired_strength)
        
        # Final subtle warm glow
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=3)
        
        print(" done!")
        return img
    
    def capture_photo(self, expired_strength=0.5, output_path=None):
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
        
        # Save with unique filename using counter
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"fujifilm_{timestamp}_{self.photo_count:03d}.jpg"
        
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
    print("║      Fujifilm Film Camera (Pi Zero 2W Optimized)         ║")
    print("║            SUBTLE VINTAGE FILM AESTHETIC                  ║")
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
    print("\nSelect film look intensity:")
    print("  1 - Subtle film (0.3)")
    print("  2 - Classic film (0.5) [RECOMMENDED]")
    print("  3 - Aged film (0.7)")
    print("\nChoice (1-3): ", end='', flush=True)
    
    choice = get_key()
    print(choice)
    
    strengths = {'1': 0.3, '2': 0.5, '3': 0.7}
    expired_strength = strengths.get(choice, 0.5)
    
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
