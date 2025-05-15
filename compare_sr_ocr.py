import os
import cv2
import numpy as np
import torch
import glob
import sys
import argparse
from tqdm import tqdm
import pandas as pd
import yaml

# Add the necessary paths
sys.path.append('./A-super-resolution-framework-for-license-plate-recognition/crnn_plate_recognition')

# Import CRNN OCR modules
from plateNet import myNet_ocr
from alphabets import plate_chr
from swinfir.archs.swinfir_arch import SwinFIR

# Constants
PROCESSED_DIR = 'noisy_plates'  # LR images
PROCESSED_HR_DIR = 'noisy_plates_HR'  # HR ground truth images
OUTPUT_DIR = 'sr_results'
CRNN_MODEL_PATH = './A-super-resolution-framework-for-license-plate-recognition/crnn_plate_recognition/saved_model/best.pth'
VGG_CONFIG_PATH = './A-super-resolution-framework-for-license-plate-recognition/options/test/SwinFIR/test_VGG_only.yml'
SWINT_DISTS_CONFIG_PATH = './A-super-resolution-framework-for-license-plate-recognition/options/test/SwinFIR/test_SwinT_DISTS.yml'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'VGG'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'SwinT_DISTS'), exist_ok=True)

# Image processing utilities
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

def init_sr_model_from_config(device, config_path):
    """Initialize a super-resolution model from config file"""
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    
    # Get model parameters from config
    model_params = opt['network_g']
    model_path = opt['path']['pretrain_network_g']
    param_key = opt['path'].get('param_key_g', 'params')
    
    # Create model
    model = SwinFIR(
        upscale=model_params['upscale'],
        in_chans=model_params['in_chans'],
        img_size=model_params['img_size'],
        window_size=model_params['window_size'],
        img_range=model_params['img_range'],
        depths=model_params['depths'],
        embed_dim=model_params['embed_dim'],
        num_heads=model_params['num_heads'],
        mlp_ratio=model_params['mlp_ratio'],
        upsampler=model_params['upsampler'],
        resi_connection=model_params['resi_connection']
    )
    
    # Load model weights
    loadnet = torch.load(model_path, map_location=device)
    if param_key in loadnet:
        model.load_state_dict(loadnet[param_key], strict=True)
    else:
        model.load_state_dict(loadnet['params_ema'], strict=True)
    
    model.eval()
    model = model.to(device)
    return model

def run_sr_model(model, img, device, window_size=5, scale=2):
    """Run super-resolution model on an image"""
    # Prepare input
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float() / 255.
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Pad input image to be a multiple of window_size
        _, _, h, w = img.size()
        mod_pad_h = (h // window_size + 1) * window_size - h
        mod_pad_w = (w // window_size + 1) * window_size - w
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]
        
        output = model(img)
        output = output[..., :h * scale, :w * scale]
    
    # Convert output to image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    return output

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

def main():
    parser = argparse.ArgumentParser(description='Compare SR models with OCR performance')
    parser.add_argument('--vgg_config', type=str, default=VGG_CONFIG_PATH, help='Path to VGG config')
    parser.add_argument('--swint_dists_config', type=str, default=SWINT_DISTS_CONFIG_PATH, help='Path to SwinT+DISTS config')
    args = parser.parse_args()
    
    # Use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize OCR model
    print("Loading OCR model...")
    ocr_model = init_ocr_model(device, CRNN_MODEL_PATH)
    ocr_img_size = (48, 168)  # Default OCR model input size (height, width)
    
    # Initialize SR models from config files
    print("Loading VGG SR model from config...")
    vgg_model = init_sr_model_from_config(device, args.vgg_config)
    
    print("Loading SwinT+DISTS SR model from config...")
    swint_dists_model = init_sr_model_from_config(device, args.swint_dists_config)
    
    # Get all LR images
    lr_images = glob.glob(os.path.join(PROCESSED_DIR, "*.jpg"))[:10000]
    print(f"Found {len(lr_images)} LR images to process")
    
    # Results storage
    results = []
    deleted_count = 0
    
    # Process each image
    for img_path in tqdm(lr_images):
        try:
            # Extract ID (ground truth) from filename
            base_name = os.path.basename(img_path)
            image_id = os.path.splitext(base_name)[0]
            
            # Read the LR image
            lr_img = cv_imread(img_path)
            if lr_img.shape[-1] != 3:
                lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGRA2BGR)
            
            # Get corresponding HR image path and read it
            hr_img_path = os.path.join(PROCESSED_HR_DIR, base_name)
            hr_img = cv_imread(hr_img_path)
            if hr_img.shape[-1] != 3:
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGRA2BGR)
            
            # Get OCR result on HR image
            hr_text = get_plate_result(hr_img, device, ocr_model, ocr_img_size)
            hr_matches = count_matching_chars(image_id, hr_text)
            
            # Get OCR result on LR image
            lr_text = get_plate_result(lr_img, device, ocr_model, ocr_img_size)
            lr_matches = count_matching_chars(image_id, lr_text)
            
            # Process with VGG model
            vgg_sr_img = run_sr_model(vgg_model, lr_img, device, window_size=5, scale=2)
            vgg_sr_text = get_plate_result(vgg_sr_img, device, ocr_model, ocr_img_size)
            vgg_matches = count_matching_chars(image_id, vgg_sr_text)
            
            # Save VGG SR image
            vgg_output_path = os.path.join(OUTPUT_DIR, 'VGG', f"{image_id}_SR.jpg")
            
            # Process with SwinT+DISTS model
            swint_sr_img = run_sr_model(swint_dists_model, lr_img, device, window_size=5, scale=2)
            swint_sr_text = get_plate_result(swint_sr_img, device, ocr_model, ocr_img_size)
            swint_matches = count_matching_chars(image_id, swint_sr_text)
            
            # Save SwinT+DISTS SR image
            swint_output_path = os.path.join(OUTPUT_DIR, 'SwinT_DISTS', f"{image_id}_SR.jpg")
            
            # Check if we should keep this result
            should_keep = (lr_matches == 5) and (vgg_matches > lr_matches and swint_matches > lr_matches)
            
            if not should_keep:
                # Don't save images and don't add to results
                deleted_count += 1
                # Delete SR images if they already exist
                if os.path.exists(vgg_output_path):
                    os.remove(vgg_output_path)
                if os.path.exists(swint_output_path):
                    os.remove(swint_output_path)
                    
                # delete lr image and hr image
                os.remove(img_path)
                os.remove(hr_img_path)
            else:
                # Save SR images
                cv_imwrite(vgg_output_path, vgg_sr_img)
                cv_imwrite(swint_output_path, swint_sr_img)
                
                # Store results
                results.append({
                    'ID': image_id,
                    'HR_OCR': hr_text,
                    'HR_Matches': hr_matches,
                    'LR_OCR': lr_text,
                    'LR_Matches': lr_matches,
                    'VGG_OCR': vgg_sr_text,
                    'VGG_Matches': vgg_matches,
                    'SwinT_DISTS_OCR': swint_sr_text,
                    'SwinT_DISTS_Matches': swint_matches
                })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    if not results_df.empty:
        hr_avg_matches = results_df['HR_Matches'].mean()
        lr_avg_matches = results_df['LR_Matches'].mean()
        vgg_avg_matches = results_df['VGG_Matches'].mean()
        swint_avg_matches = results_df['SwinT_DISTS_Matches'].mean()
    else:
        hr_avg_matches = lr_avg_matches = vgg_avg_matches = swint_avg_matches = 0
    
    # Print summary
    print("\n--- OCR Recognition Results ---")
    print(f"Total images processed: {len(results_df)}")
    print(f"Records deleted: {deleted_count}")
    print(f"Average matching characters in HR images: {hr_avg_matches:.2f}")
    print(f"Average matching characters in LR images: {lr_avg_matches:.2f}")
    print(f"Average matching characters in VGG SR images: {vgg_avg_matches:.2f}")
    print(f"Average matching characters in SwinT+DISTS SR images: {swint_avg_matches:.2f}")
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'sr_ocr_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    main() 