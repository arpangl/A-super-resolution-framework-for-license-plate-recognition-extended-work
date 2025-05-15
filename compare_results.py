import os
import cv2
import numpy as np
    # Use PIL for Chinese text rendering
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv

# Read IDs from the comparison CSV
ids = []
with open('sr_results/sr_ocr_comparison.csv', 'r') as f:
    reader = csv.DictReader(f)
    print(reader)
    for row in reader:
        print(row)
        ids.append(row['ID'])

# Create output directory
os.makedirs('comparison_results', exist_ok=True)

def load_and_resize(img_path, target_size=(128, 48)):
    """Load an image and resize it to target size"""
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} does not exist")
        return np.zeros((*target_size, 3), dtype=np.uint8)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return np.zeros((*target_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def add_label(img, label, label_height=24):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    label_color = (0, 0, 0)
    labeled = np.full((label_height + img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
    labeled[label_height:, :, :] = img
    cv2.putText(labeled, label, (5, int(label_height*0.8)), font, font_scale, label_color, font_thickness, cv2.LINE_AA)
    return labeled

comparison_rows = []

# Process each ID
for img_id in ids:
    # Load images
    lr_path = f'noisy_plates/{img_id}.jpg'
    hr_path = f'noisy_plates_HR/{img_id}.jpg'
    swint_path = f'sr_results/SwinT_DISTS/{img_id}_SR.jpg'
    vgg_path = f'sr_results/VGG/{img_id}_SR.jpg'
    
    # Load all images
    lr_img = load_and_resize(lr_path)
    hr_img = load_and_resize(hr_path)
    swint_img = load_and_resize(swint_path)
    vgg_img = load_and_resize(vgg_path)
    
    lr_img = add_label(lr_img, 'LR')
    hr_img = add_label(hr_img, 'HR')
    swint_img = add_label(swint_img, 'SwinT-DISTS')
    vgg_img = add_label(vgg_img, 'VGG')

    # Add ID label to the left of the row
    id_label_width = 120
    id_label_height = lr_img.shape[0]
    id_label_img = np.full((id_label_height, id_label_width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    
    # Convert OpenCV image to PIL
    pil_img = Image.fromarray(id_label_img)
    draw = ImageDraw.Draw(pil_img)
    
    # Use a font that supports Chinese characters
    # You may need to specify the full path to your font file
    try:
        font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # Common path for Chinese fonts on Linux
        font_pil = ImageFont.truetype(font_path, size=int(font_scale * 24))
    except IOError:
        # Fallback to default font if Chinese font not available
        font_pil = ImageFont.load_default()
    
    # Draw text with PIL
    draw.text((5, int(id_label_height/2) - 15), img_id, fill=(0, 0, 0), font=font_pil)
    
    # Convert back to OpenCV format
    id_label_img = np.array(pil_img)

    row = np.concatenate([id_label_img, lr_img, hr_img, swint_img, vgg_img], axis=1)
    comparison_rows.append(row)
    # Optionally, save each row as before
    cv2.imwrite(f'comparison_results/{img_id}_comparison.png', cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

# Stack all rows vertically
if comparison_rows:
    all_comparisons = np.concatenate(comparison_rows, axis=0)
    cv2.imwrite('comparison_results/all_comparisons.png', cv2.cvtColor(all_comparisons, cv2.COLOR_RGB2BGR))
    print("All comparison images have been combined into 'comparison_results/all_comparisons.png'")
else:
    print("No comparison rows to combine.")

print("Comparison images have been generated in the 'comparison_results' directory") 