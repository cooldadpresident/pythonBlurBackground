import cv2
import numpy as np
import sys
import os

def create_person_mask(image):
    """
    Create a mask that identifies people in the image using a combination of
    face detection and color-based segmentation.
    """
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Face detection to identify primary subjects
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Create initial mask
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # If faces are detected, create regions around them
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Create a significantly expanded region around each face
            expanded_width = int(w * 2.5)
            expanded_height = int(h * 4.5)  # More expansion downward for body
            
            # Calculate the center of the face
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate expanded rectangle coordinates
            expanded_x = max(0, center_x - expanded_width // 2)
            expanded_y = max(0, y - h // 2)  # Extend upward for hair
            expanded_right = min(width, expanded_x + expanded_width)
            expanded_bottom = min(height, expanded_y + expanded_height)
            
            # Draw the expanded rectangle on the mask
            cv2.rectangle(mask, 
                         (expanded_x, expanded_y), 
                         (expanded_right, expanded_bottom), 
                         255, 
                         -1)
    else:
        # If no faces detected, use the center area as fallback
        center_width = width // 2
        center_height = height // 2
        x_start = (width - center_width) // 2
        y_start = (height - center_height) // 2
        cv2.rectangle(mask, 
                     (x_start, y_start), 
                     (x_start + center_width, y_start + center_height), 
                     255, 
                     -1)
    
    # Dilate the initial mask
    kernel = np.ones((30, 30), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Refine with GrabCut
    try:
        # Create mask for GrabCut
        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        grabcut_mask[dilated_mask > 0] = cv2.GC_PR_FGD  # Probable foreground
        grabcut_mask[dilated_mask == 0] = cv2.GC_PR_BGD  # Probable background
        
        # Run GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        
        # Create final mask
        final_mask = np.where(
            (grabcut_mask == cv2.GC_PR_FGD) | (grabcut_mask == cv2.GC_FGD), 
            255, 
            0
        ).astype(np.uint8)
        
        # Ensure the mask is properly smoothed
        final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
        
        return final_mask
    except:
        # If GrabCut fails, return the dilated mask as fallback
        return dilated_mask

def apply_edge_preserving_blur(image, blur_strength=35):
    """Apply edge-preserving blur that looks more natural than Gaussian blur"""
    return cv2.edgePreservingFilter(image, flags=1, sigma_s=blur_strength, sigma_r=0.4)

def apply_high_quality_blur(image_path, output_path, blur_strength=35):
    """
    Apply high-quality portrait background blur that maintains sharpness of foreground
    while creating a natural-looking blurred background
    
    Args:
        image_path: Path to input image
        output_path: Path to save result
        blur_strength: Strength of blur effect (higher = more blur)
    """
    try:
        # Check if file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load image
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Failed to load image: {image_path}")
            
        # Get dimensions
        height, width = image.shape[:2]
        
        print("Identifying people in the image...")
        # Generate a mask for the people in the image
        person_mask = create_person_mask(image)
        
        # Normalize mask values between 0 and 1 for blending
        person_mask_norm = person_mask.astype(float) / 255.0
        
        # Create 3-channel mask for processing
        person_mask_3ch = np.stack([person_mask_norm] * 3, axis=2)
        
        # Apply high-quality blur using edge-preserving filter
        print(f"Applying artistic background blur (strength: {blur_strength})...")
        blurred = apply_edge_preserving_blur(image, blur_strength)
        
        # Create an extra-blurred version for distant backgrounds
        extra_blurred = cv2.GaussianBlur(blurred, (25, 25), 0)
        
        # Blend the original image with both blur levels
        # This creates a more natural depth-of-field effect
        result = (person_mask_3ch * image + 
                 (1 - person_mask_3ch) * 
                 (0.7 * blurred + 0.3 * extra_blurred))
        
        # Convert to proper format for saving
        result = result.astype(np.uint8)
        
        # Save the final image
        print(f"Saving result to: {output_path}")
        cv2.imwrite(output_path, result)
        print("Processing complete!")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Main execution logic
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py input_image.jpg [output_image.jpg] [blur_strength]")
        print("Example: python script.py portrait.jpg blurred_portrait.jpg 35")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Use default output path if not provided
    output_path = sys.argv[2] if len(sys.argv) > 2 else "blurred_" + os.path.basename(input_path)
    
    # Use default blur strength if not provided
    try:
        blur_strength = int(sys.argv[3]) if len(sys.argv) > 3 else 35
    except ValueError:
        print("Blur strength must be a number. Using default value 35.")
        blur_strength = 35
    
    success = apply_high_quality_blur(input_path, output_path, blur_strength)
    
    if not success:
        print("\nTroubleshooting Tips:")
        print("1. Try a different image with clearer subjects")
        print("2. Try adjusting the blur strength (higher values = stronger blur)")
        print("3. If all else fails, consider using a dedicated photo editing application")
        sys.exit(1)