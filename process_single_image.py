import os
import cv2
import numpy as np
import torch
import argparse
import yaml
import sys

from swinfir.archs.swinfir_arch import SwinFIR

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

def main():
    parser = argparse.ArgumentParser(description='Process image with super-resolution model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--config', type=str, default='./A-super-resolution-framework-for-license-plate-recognition/options/test/SwinFIR/test_SwinT_DISTS.yml', 
                        help='Path to SR model config')
    parser.add_argument('--output', type=str, default='sr_output.jpg', help='Output image filename')
    args = parser.parse_args()
    
    # Use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize SR model from config file
    print("Loading SR model from config...")
    sr_model = init_sr_model_from_config(device, args.config)
    
    try:
        # Read the input image
        original_img = cv_imread(args.input)
        if original_img is None:
            print(f"Error: Could not read image at {args.input}")
            return
            
        if original_img.shape[-1] == 4:  # Handle RGBA images
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
        
        # Resize to 55x20
        resized_img = cv2.resize(original_img, (55, 20))
        print(f"Resized image to 55x20")
        
        # Process with SR model
        sr_img = run_sr_model(sr_model, resized_img, device, window_size=5, scale=2)
        print(f"Processed image through SR model")
        
        # Save the SR image
        output_path = args.output
        cv_imwrite(output_path, sr_img)
        print(f"Saved SR image to: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main() 