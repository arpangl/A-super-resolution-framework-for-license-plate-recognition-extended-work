import os
import cv2
import numpy as np
import torch
import glob
import sys
from tqdm import tqdm

# Add the necessary paths
sys.path.append('./A-super-resolution-framework-for-license-plate-recognition/crnn_plate_recognition')

# Import the necessary modules
from plateNet import myNet_ocr
from alphabets import plate_chr

# Define constants
SOURCE_DIR = 'license-plate-generator/plate_images/single_blue'
OUTPUT_DIR = 'noisy_plates'
OUTPUT_HR_DIR = 'noisy_plates_HR'
MODEL_PATH = './A-super-resolution-framework-for-license-plate-recognition/crnn_plate_recognition/saved_model/best.pth'
TARGET_SIZE = (168, 48)  # Width, Height for the model input
INPUT_SIZE = (55, 20)    # Width, Height for the initial resize, as requested
HR_SIZE = (110, 40)      # Width, Height for the HR images, as requested
EXPECTED_CHARS = 5  # Number of characters we want to recognize correctly

# Ensure the output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_HR_DIR, exist_ok=True)

# Image processing function from demo.py
mean_value, std_value = (0.588, 0.193)

def cv_imread(path):
    """Read image with OpenCV, handling non-ASCII paths"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

def cv_imwrite(path, img):
    """Write image with OpenCV, handling non-ASCII paths"""
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    if is_success:
        im_buf_arr.tofile(path)
    return is_success

def image_processing(img, device, img_size):
    """Preprocess image for OCR model"""
    img_h, img_w = img_size
    img = cv2.resize(img, (img_w, img_h))
    
    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def decodePlate(preds):
    """Decode the predictions to characters"""
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    return newPreds

def get_plate_result(img, device, model, img_size):
    """Get OCR recognition result"""
    input_tensor = image_processing(img, device, img_size)
    preds = model(input_tensor)
    preds = preds.argmax(dim=2)
    preds = preds.view(-1).detach().cpu().numpy()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate

def init_model(device, model_path):
    """Initialize the OCR model"""
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=len(plate_chr), export=True, cfg=cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def count_exact_matches(ground_truth, ocr_result):
    """
    Count the number of matching characters between ground truth and OCR result
    without considering position
    """
    # Count frequency of each character in ground truth
    gt_chars = {}
    for char in ground_truth:
        gt_chars[char] = gt_chars.get(char, 0) + 1
    
    # Count matches
    matches = 0
    
    # Check each character in OCR result
    for char in ocr_result:
        if char in gt_chars and gt_chars[char] > 0:
            matches += 1
            gt_chars[char] -= 1
    
    return matches

def apply_gaussian_blur(image, kernel_size):
    """
    Apply Gaussian blur to an image
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        
    Returns:
        Blurred image
    """
    # Make sure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

def find_optimal_blur(img_path, device, model, img_size):
    """
    Find the optimal blur level where the OCR recognizes exactly EXPECTED_CHARS characters
    
    Args:
        img_path: Path to the license plate image
        device: Torch device
        model: OCR model
        img_size: Input size for the model
        
    Returns:
        Tuple of (optimal_kernel_size, blurred_image, original_image)
    """
    # Extract ground truth from filename
    base_name = os.path.basename(img_path)
    id_parts = base_name.split('_')
    if len(id_parts) >= 2:
        ground_truth = id_parts[1].split('.')[0]
    else:
        ground_truth = base_name.split('.')[0]
    
    # Read the image
    img = cv_imread(img_path)
    if img is None:
        print(f"Could not read image {img_path}")
        return None, None, None
    
    if img.shape[-1] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # First resize to the target input size (55x20) as requested
    img = cv2.resize(img, (INPUT_SIZE[0], INPUT_SIZE[1]))
    original_img = img.copy()  # Save a copy of the clean, resized image
    
    # Initial blur parameters - try different kernel sizes
    # Limited to a maximum of 6 as requested
    kernel_sizes = [1, 3, 5]
    
    best_kernel_size = None
    best_img = None
    best_match_diff = float('inf')
    
    # Try different blur levels
    for kernel_size in kernel_sizes:
        # Skip kernel size 1 (no blur)
        if kernel_size == 1:
            continue
            
        # Apply blur to the image
        blurred_img = apply_gaussian_blur(img.copy(), kernel_size)
        
        # Resize for OCR model
        resized_blurred = cv2.resize(blurred_img, (TARGET_SIZE[0], TARGET_SIZE[1]))
        
        # Get OCR result
        plate_text = get_plate_result(resized_blurred, device, model, img_size)
        
        # Count matching characters
        matching_chars = count_exact_matches(ground_truth, plate_text)
        match_diff = abs(matching_chars - EXPECTED_CHARS)
        
        print(f"  Kernel size {kernel_size}: matches={matching_chars}, target={EXPECTED_CHARS}")
        
        # Check if this is the best match so far
        if match_diff < best_match_diff:
            best_match_diff = match_diff
            best_kernel_size = kernel_size
            best_img = blurred_img
        
        # If exact match found
        if matching_chars == EXPECTED_CHARS:
            break
    
    # If no kernel size gave exactly 5 matches, try kernel size 6 as the maximum
    if best_match_diff > 0 and best_kernel_size is not None:
        kernel_size = 6  # Maximum allowed kernel size
        blurred_img = apply_gaussian_blur(img.copy(), kernel_size)
        
        # Resize for OCR model
        resized_blurred = cv2.resize(blurred_img, (TARGET_SIZE[0], TARGET_SIZE[1]))
        
        # Get OCR result
        plate_text = get_plate_result(resized_blurred, device, model, img_size)
        
        # Count matching characters
        matching_chars = count_exact_matches(ground_truth, plate_text)
        match_diff = abs(matching_chars - EXPECTED_CHARS)
        
        print(f"  Kernel size {kernel_size}: matches={matching_chars}, target={EXPECTED_CHARS}")
        
        # Check if this is the best match so far
        if match_diff < best_match_diff:
            best_match_diff = match_diff
            best_kernel_size = kernel_size
            best_img = blurred_img
            
    # If we didn't find an exact match, use the best approximation
    if best_kernel_size is None:
        best_kernel_size = 5  # Default to kernel size 5 if nothing worked
        best_img = apply_gaussian_blur(img.copy(), best_kernel_size)
        
    return best_kernel_size, best_img, original_img

def main():
    # Use CPU for inference
    device = torch.device("cpu")
    
    # Load OCR model
    print("Loading OCR model...")
    model = init_model(device, MODEL_PATH)
    img_size = (48, 168)  # Model input size (height, width)
    
    # Find all license plate images
    plate_images = glob.glob(os.path.join(SOURCE_DIR, "*.jpg"))
    print(f"Found {len(plate_images)} license plate images")
    
    results = []
    
    # Process each image
    for img_path in tqdm(plate_images):
        try:
            base_name = os.path.basename(img_path)
            print(f"\nProcessing {base_name}...")
            
            # Find optimal blur level
            optimal_kernel_size, blurred_img, original_img = find_optimal_blur(img_path, device, model, img_size)
            
            if blurred_img is not None and original_img is not None:
                # Extract ground truth from filename
                id_parts = base_name.split('_')
                if len(id_parts) >= 2:
                    ground_truth = id_parts[1].split('.')[0]
                else:
                    ground_truth = base_name.split('.')[0]
                base_name = ground_truth + '.jpg'
                # Save the original HR image (110x40)
                hr_original_img = cv2.resize(original_img, (HR_SIZE[0], HR_SIZE[1]))
                hr_original_path = os.path.join(OUTPUT_HR_DIR, f"{base_name}")
                cv_imwrite(hr_original_path, hr_original_img)
                
                # Save the blurred image (55x20)
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}")
                cv_imwrite(output_path, blurred_img)
                
                # Resize for OCR and check result
                resized = cv2.resize(blurred_img, (TARGET_SIZE[0], TARGET_SIZE[1]))
                ocr_result = get_plate_result(resized, device, model, img_size)
                
                matching_chars = count_exact_matches(ground_truth, ocr_result)
                
                print(f"Image: {base_name}, Blur kernel size: {optimal_kernel_size}")
                print(f"Ground truth: {ground_truth}, OCR: {ocr_result}")
                print(f"Matching chars: {matching_chars}/{len(ground_truth)}")
                
                results.append({
                    'filename': base_name,
                    'kernel_size': optimal_kernel_size,
                    'ground_truth': ground_truth,
                    'ocr_result': ocr_result,
                    'matching_chars': matching_chars
                })
            else:
                print(f"Failed to process {base_name}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Results saved to {OUTPUT_DIR} and {OUTPUT_HR_DIR}")
    
    # Calculate average kernel size
    if results:
        avg_kernel_size = sum(r['kernel_size'] for r in results) / len(results)
        print(f"Average optimal blur kernel size: {avg_kernel_size:.2f}")

if __name__ == "__main__":
    main() 