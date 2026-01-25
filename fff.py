#!/usr/bin/env python3
"""
Arducam Fujifilm Film Look - Optimized for Raspberry Pi Zero 2W
Memory-efficient version with streaming processing
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from datetime import datetime
import time
import gc

class FujiFilmProcessor:
    def __init__(self, resolution=(1920, 1080)):
        """
        Initialize with lower resolution for Pi Zero 2W
        resolution: tuple (width, height) - lower values use less memory
        """
        self.picam2 = Picamera2()
        self.resolution = resolution
        
    def setup_camera(self):
        """Configure camera with memory-efficient settings"""
        config = self.picam2.create_still_configuration(
            main={"size": self.resolution},
            buffer_count=2  # Minimize buffer usage
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        
    def apply_fuji_colors(self, img):
        """Apply Fujifilm colors in-place to save memory"""
        # Work directly on float32 to avoid extra copies
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        
        # LAB color space manipulation
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Adjust colors in-place
        lab[:, :, 1] -= 5  # Shift toward green
        lab[:, :, 2] += 8  # Warm shift
        
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        del lab
        gc.collect()
        
        # HSV for shadow tint
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Cyan tint in shadows (in-place)
        shadow_mask = hsv[:, :, 2] < 0.3
        hsv[shadow_mask, 0] += 0.05
        
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        del hsv, shadow_mask
        gc.collect()
        
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8, copy=False)
    
    def apply_film_curve(self, img):
        """Apply S-curve with lifted blacks - in-place operations"""
        img = img.astype(np.float32, copy=False)
        img *= (1.0 / 255.0)
        
        # Lift blacks and adjust contrast
        black_lift = 0.08
        contrast = 0.87  # 1/1.15
        
        np.power(img, contrast, out=img)
        img *= (1 - black_lift)
        img += black_lift
        
        np.clip(img, 0, 1, out=img)
        img *= 255
        return img.astype(np.uint8, copy=False)
    
    def add_grain(self, img, intensity=0.012):
        """Add grain using smaller random arrays"""
        h, w = img.shape[:2]
        
        # Generate grain in chunks to save memory
        chunk_size = 256
        img_float = img.astype(np.float32, copy=False)
        
        for i in range(0, h, chunk_size):
            for j in range(0, w, chunk_size):
                end_i = min(i + chunk_size, h)
                end_j = min(j + chunk_size, w)
                
                grain = np.random.normal(0, intensity * 255, 
                                        (end_i - i, end_j - j, 3)).astype(np.float32)
                img_float[i:end_i, j:end_j] += grain
                
        np.clip(img_float, 0, 255, out=img_float)
        return img_float.astype(np.uint8, copy=False)
    
    def add_expired_effects(self, img, strength=0.5):
        """Memory-efficient expired film effects"""
        h, w = img.shape[:2]
        
        # Color shifts in-place
        if np.random.random() > 0.5:
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + int(strength * 8), 0, 255).astype(np.uint8)
            img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + int(strength * 6), 0, 255).astype(np.uint8)
        else:
            img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + int(strength * 10), 0, 255).astype(np.uint8)
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + int(strength * 8), 0, 255).astype(np.uint8)
        
        # Simplified vignette
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        
        # Calculate distance from center (normalized)
        r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        vignette = 1 - (strength * 0.3 * np.clip(r - 0.5, 0, 1))
        
        img_float = img.astype(np.float32, copy=False)
        img_float *= vignette[:, :, np.newaxis]
        
        del r, vignette
        gc.collect()
        
        np.clip(img_float, 0, 255, out=img_float)
        return img_float.astype(np.uint8, copy=False)
    
    def reduce_saturation(self, img, factor=0.85):
        """Desaturate in-place"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * factor).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        del hsv
        gc.collect()
        return img
    
    def process_image(self, img, expired_strength=0.4):
        """
        Process image with minimal memory footprint
        All operations modify arrays in-place where possible
        """
        print("Processing: Fuji colors...", end='', flush=True)
        img = self.apply_fuji_colors(img)
        
        print(" film curve...", end='', flush=True)
        img = self.apply_film_curve(img)
        
        print(" saturation...", end='', flush=True)
        img = self.reduce_saturation(img, 0.88)
        
        print(" grain...", end='', flush=True)
        img = self.add_grain(img, intensity=0.012)
        
        print(" expired effects...", end='', flush=True)
        img = self.add_expired_effects(img, strength=expired_strength)
        
        # Final warm shift
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=3)
        
        print(" done!")
        return img
    
    def capture_photo(self, expired_strength=0.4, output_path=None):
        """Capture and process a photo"""
        print("Capturing image...")
        
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
        cv2.imwrite(output_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        # Free processed image
        del processed
        gc.collect()
        
        print(f"✓ Photo saved successfully")
        return output_path
    
    def cleanup(self):
        """Clean up resources"""
        self.picam2.stop()
        gc.collect()

def main():
    print("=== Fujifilm Film Look Camera (Pi Zero 2W Optimized) ===\n")
    
    # Use moderate resolution for Pi Zero 2W
    # Options: (1920, 1080), (1280, 720), or (2592, 1944) if you have enough memory
    processor = FujiFilmProcessor(resolution=(1920, 1080))
    
    print("Setting up camera...")
    processor.setup_camera()
    
    try:
        # Take photo with slight expired look
        # expired_strength: 0.0 = fresh, 0.3-0.5 = subtle expired, 1.0 = heavy
        processor.capture_photo(expired_strength=0.4)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nCleaning up...")
        processor.cleanup()
        print("Done.")

if __name__ == "__main__":
    main()
