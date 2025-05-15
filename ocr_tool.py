import os
import cv2
import numpy as np
import torch
import argparse
import sys

# Add the necessary paths
sys.path.append('./A-super-resolution-framework-for-license-plate-recognition/crnn_plate_recognition')

# Import CRNN OCR modules
from plateNet import myNet_ocr
from alphabets import plate_chr

# Constants
CRNN_MODEL_PATH = './A-super-resolution-framework-for-license-plate-recognition/crnn_plate_recognition/saved_model/best.pth'

# Image processing utilities
mean_value, std_value = (0.588, 0.193)

def cv_imread(path):
    """Read image with OpenCV, handling non-ASCII paths"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

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

def count_matching_chars(truth, prediction):
    """Count how many characters in the prediction match the ground truth"""
    # Count frequency of each character in ground truth
    gt_chars = {}
    for char in truth:
        gt_chars[char] = gt_chars.get(char, 0) + 1
    
    # Count matches
    matches = 0
    
    # Check each character in prediction
    for char in prediction:
        if char in gt_chars and gt_chars[char] > 0:
            matches += 1
            gt_chars[char] -= 1
    
    return matches

def init_ocr_model(device, model_path):
    """Initialize the OCR model"""
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=len(plate_chr), export=True, cfg=cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='License Plate OCR Tool')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--model_path', type=str, default=CRNN_MODEL_PATH, help='Path to OCR model')
    parser.add_argument('--ground_truth', type=str, help='Ground truth text (for single image) or filename pattern for batch (e.g., use filename without extension as truth)')
    parser.add_argument('--use_filename', action='store_true', help='Use filename (without extension) as ground truth')
    args = parser.parse_args()
    
    # Use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize OCR model
    print("Loading OCR model...")
    ocr_model = init_ocr_model(device, args.model_path)
    ocr_img_size = (48, 168)  # Default OCR model input size (height, width)
    
    # Process input (file or directory)
    if os.path.isdir(args.input):
        # Process all images in directory
        total_matches = 0
        total_chars = 0
        processed_files = 0
        
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(args.input, filename)
                try:
                    # Read the image
                    img = cv_imread(img_path)
                    if img is None:
                        print(f"Failed to read: {img_path}")
                        continue
                    
                    # Convert to BGR if needed
                    if img.shape[-1] != 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    # Get OCR result
                    text = get_plate_result(img, device, ocr_model, ocr_img_size)
                    
                    # Check accuracy if ground truth is available
                    if args.use_filename:
                        # Use filename (without extension) as ground truth
                        ground_truth = os.path.splitext(filename)[0]
                        matches = count_matching_chars(ground_truth, text)
                        total_matches += matches
                        total_chars += len(ground_truth)
                        accuracy = matches / len(ground_truth) if ground_truth else 0
                        
                        # Print result with matching stats
                        print(f"Image: {filename}, OCR Result: {text}")
                        print(f"  Ground Truth: {ground_truth}, Matches: {matches}/{len(ground_truth)} ({accuracy:.2%})")
                    else:
                        # Just print the OCR result
                        print(f"Image: {filename}, OCR Result: {text}")
                    
                    processed_files += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Print summary statistics if we were checking accuracy
        if args.use_filename and processed_files > 0:
            overall_accuracy = total_matches / total_chars if total_chars > 0 else 0
            print(f"\n--- Summary Statistics ---")
            print(f"Processed {processed_files} images")
            print(f"Overall accuracy: {total_matches}/{total_chars} characters ({overall_accuracy:.2%})")
    else:
        # Process single image
        try:
            # Read the image
            img = cv_imread(args.input)
            if img is None:
                print(f"Failed to read: {args.input}")
                return
            
            # Convert to BGR if needed
            if img.shape[-1] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Get OCR result
            text = get_plate_result(img, device, ocr_model, ocr_img_size)
            
            # Check accuracy if ground truth is provided
            if args.ground_truth:
                matches = count_matching_chars(args.ground_truth, text)
                accuracy = matches / len(args.ground_truth) if args.ground_truth else 0
                
                # Print result with matching stats
                print(f"OCR Result: {text}")
                print(f"Ground Truth: {args.ground_truth}, Matches: {matches}/{len(args.ground_truth)} ({accuracy:.2%})")
            else:
                # Just print the OCR result
                print(f"OCR Result: {text}")
            
        except Exception as e:
            print(f"Error processing {args.input}: {e}")

if __name__ == "__main__":
    main() 