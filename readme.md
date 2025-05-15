# Read Me
This is a repo related to "A-super-resolution-framework-for-license-plate-recognition", extends the original dataset and conducts experiments on new dataset

## How to prepare new dataset?
1. Refer to this [repo](https://github.com/zheng-yuwei/license-plate-generator), we generated blue background license plates.
2. Use 'add_blur_to_plates.py' to add gaussian blur, it will iteratively find the optimal gaussian kernel for each plate, which will let it be able to recognize exactly 5 characters.
3. The 'add_blur_to_plates.py' will resize and put the results under 'noisy_plates' and 'noisy_plates_HR'

## Run a experiment based on the new generated dataset
1. run 'compare_sr_ocr.py', it will super-resolve the images under noisy_plates
2. the results will be put under 'sr_results', it also provides a csv file.

## Run a single image super resolution
1. run 'process_single_image.py' with image file path, it will generated the super-resolved images under same folder

## Some tools
1. 'compare_results.py' will take the images under 'sr_results', 'noisy_plates' and 'noisy_plates_HR' to make a comparison grid.
2. 'ocr_tool.py' can run a ocr recognition on single image.
