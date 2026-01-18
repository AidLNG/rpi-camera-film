#!/usr/bin/env python3
"""
Raspberry Pi Camera Script with Filmic Filters
Captures photos via terminal command and applies cinematic effects
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from datetime import datetime
import time

# ==================== CONFIGURATION ====================
SAVE_PATH = "/home/James/Pictures/"  # Path to save photos

# FILTER SETTINGS - Modify these to change the look
FILTER_ENABLED = True

# Color Grading
LIFT = [0.95, 0.98, 1.02]  # Shadows (RGB) - slight blue in shadows
GAMMA = [1.0, 1.05, 1.1]   # Midtones (RGB) - warm midtones
GAIN = [1.1, 1.05, 0.95]   # Highlights (RGB) - warm highlights

# Film characteristics
CONTRAST = 1.15            # Overall contrast (1.0 = normal)
SATURATION = 0.85          # Color saturation (1.0 = normal)
GRAIN_AMOUNT = 0.015       # Film grain intensity
VIGNETTE_STRENGTH = 0.3    # Edge darkening (0 = none)

# Color bleed/halation effect
COLOR_BLEED = True
BLEED_STRENGTH = 0.25      # How much colors bleed (0-1)
BLEED_RADIUS = 15          # Blur radius for bleed effect

# Tone curve for filmic S-curve response
USE_TONE_CURVE = True
SHADOWS_CRUSH = 0.05       # Crush blacks slightly
HIGHLIGHTS_ROLLOFF = 0.95  # Roll off highlights
# =======================================================

class FilmicCamera:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (1920, 1080)},
            buffer_count=2
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warm-up
        print("Camera ready!")
        
    def apply_color_grade(self, img):
        """Apply lift/gamma/gain color grading"""
        img_float = img.copy()
        
        # Convert RGB to BGR for processing
        lift = np.array([LIFT[2], LIFT[1], LIFT[0]])  # BGR order
        gamma = np.array([GAMMA[2], GAMMA[1], GAMMA[0]])  # BGR order
        gain = np.array([GAIN[2], GAIN[1], GAIN[0]])  # BGR order
        
        # Apply lift (shadows)
        img_float = img_float + lift * (1 - img_float)
        img_float = np.clip(img_float, 0, 1)
        
        # Apply gamma (midtones)
        gamma_safe = np.where(gamma > 0, gamma, 0.001)
        img_float = np.power(img_float, 1.0 / gamma_safe)
        img_float = np.clip(img_float, 0, 1)
        
        # Apply gain (highlights)
        img_float = img_float * gain
        
        return np.clip(img_float, 0, 1)
    
    def apply_tone_curve(self, img):
        """Apply filmic S-curve to compress highlights and shadows"""
        # Crush shadows slightly
        img = np.where(img < SHADOWS_CRUSH, 
                      img * (0.5 / SHADOWS_CRUSH), 
                      0.5 + (img - SHADOWS_CRUSH) * (0.5 / (1 - SHADOWS_CRUSH)))
        
        # Roll off highlights
        img = np.where(img > HIGHLIGHTS_ROLLOFF,
                      HIGHLIGHTS_ROLLOFF + (img - HIGHLIGHTS_ROLLOFF) * 0.3,
                      img)
        
        return img
    
    def add_color_bleed(self, img):
        """Simulate film halation/color bleeding effect"""
        # Extract highlights
        brightness = np.max(img, axis=2)
        mask = (brightness > 0.7).astype(np.float32)
        
        # Blur the bright areas to create bleed
        blurred = cv2.GaussianBlur(img, (BLEED_RADIUS, BLEED_RADIUS), 0)
        
        # Blend based on brightness
        mask = cv2.GaussianBlur(mask, (BLEED_RADIUS, BLEED_RADIUS), 0)
        mask = np.stack([mask] * 3, axis=2)
        
        result = img * (1 - mask * BLEED_STRENGTH) + blurred * mask * BLEED_STRENGTH
        return result
    
    def adjust_saturation(self, img, sat):
        """Adjust color saturation"""
        # Convert float image to uint8 for OpenCV
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result.astype(np.float32) / 255.0
    
    def add_vignette(self, img):
        """Add edge darkening"""
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        radius = np.sqrt(x**2 + y**2)
        vignette = 1 - VIGNETTE_STRENGTH * np.clip(radius - 0.5, 0, 1)
        vignette = np.stack([vignette] * 3, axis=2)
        return img * vignette
    
    def add_grain(self, img):
        """Add film grain"""
        noise = np.random.normal(0, GRAIN_AMOUNT, img.shape)
        return np.clip(img + noise, 0, 1)
    
    def apply_contrast(self, img, contrast):
        """Apply contrast adjustment"""
        return np.clip((img - 0.5) * contrast + 0.5, 0, 1)
    
    def apply_filters(self, image):
        """Apply all filmic filters to the image"""
        if not FILTER_ENABLED:
            return image
        
        # Ensure we have a valid image
        if image is None or image.size == 0:
            return image
        
        # Convert to float (OpenCV uses BGR by default)
        img = image.astype(np.float32) / 255.0
        
        # Ensure values are in valid range
        img = np.clip(img, 0, 1)
        
        # Apply color grading (lift/gamma/gain)
        img = self.apply_color_grade(img)
        img = np.clip(img, 0, 1)
        
        # Apply tone curve for filmic response
        if USE_TONE_CURVE:
            img = self.apply_tone_curve(img)
            img = np.clip(img, 0, 1)
        
        # Adjust contrast
        img = self.apply_contrast(img, CONTRAST)
        img = np.clip(img, 0, 1)
        
        # Add color bleed effect
        if COLOR_BLEED:
            img = self.add_color_bleed(img)
            img = np.clip(img, 0, 1)
        
        # Adjust saturation
        img = self.adjust_saturation(img, SATURATION)
        img = np.clip(img, 0, 1)
        
        # Add vignette
        if VIGNETTE_STRENGTH > 0:
            img = self.add_vignette(img)
            img = np.clip(img, 0, 1)
        
        # Add film grain
        if GRAIN_AMOUNT > 0:
            img = self.add_grain(img)
            img = np.clip(img, 0, 1)
        
        # Convert back to uint8
        return (img * 255).astype(np.uint8)
    
    def capture(self):
        """Capture and process a photo"""
        print("Capturing photo...")
        
        # Capture image
        image = self.picam2.capture_array()
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply filters
        processed = self.apply_filters(image)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_PATH}photo_{timestamp}.jpg"
        cv2.imwrite(filename, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"✓ Photo saved: {filename}")
        return filename
    
    def cleanup(self):
        """Clean up resources"""
        self.picam2.stop()

def print_menu():
    """Display menu options"""
    print("\n" + "="*50)
    print("FILMIC CAMERA - TERMINAL CONTROL")
    print("="*50)
    print("Commands:")
    print("  c / capture  - Take a photo")
    print("  s / status   - Show current settings")
    print("  q / quit     - Exit program")
    print("="*50)

if __name__ == "__main__":
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    try:
        print("Starting Filmic Camera (Terminal Mode)...")
        print(f"Save path: {SAVE_PATH}")
        
        # Initialize camera
        camera = FilmicCamera()
        
        print_menu()
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter command: ").strip().lower()
                
                if user_input in ['c', 'capture']:
                    camera.capture()
                    
                elif user_input in ['s', 'status']:
                    print("\n--- Current Settings ---")
                    print(f"Filters: {'Enabled' if FILTER_ENABLED else 'Disabled'}")
                    print(f"Contrast: {CONTRAST}")
                    print(f"Saturation: {SATURATION}")
                    print(f"Grain: {GRAIN_AMOUNT}")
                    print(f"Vignette: {VIGNETTE_STRENGTH}")
                    print(f"Color Bleed: {'On' if COLOR_BLEED else 'Off'}")
                    print(f"Tone Curve: {'On' if USE_TONE_CURVE else 'Off'}")
                    
                elif user_input in ['q', 'quit', 'exit']:
                    print("\nShutting down...")
                    break
                    
                elif user_input in ['h', 'help', '?']:
                    print_menu()
                    
                else:
                    print("Unknown command. Type 'h' for help.")
                    
            except EOFError:
                # Handle Ctrl+D
                print("\nShutting down...")
                break
                
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        camera.cleanup()
        print("Goodbye!")
