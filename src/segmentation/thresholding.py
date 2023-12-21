import numpy as np
import cv2
from PIL.Image import Image
from tqdm import tqdm
import argparse
import random
import os

def thresholding_on_one_image(path):
    img = cv2.imread(path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    threshold = int(threshold) # Convert to int as normally it is float which will generate error on the next line
    thresholds = list(range(threshold, 255, threshold))
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in thresholds]

    final_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Apply color to regions based on thresholds
    for i in range(len(thresholds)):
        final_img[grayscale_img > thresholds[i]] = colors[i]
        
    grayscale_final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)

    return final_img, grayscale_final_img

def thresholding_on_multiple_images(paths, output_path):
    assert os.path.exists(paths), f"The path {paths} does not exist"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "colored"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "grayscale"), exist_ok=True)
    for image_name in tqdm(os.listdir(paths), desc="Processing images", total=len(os.listdir(paths))):
        final_img, grayscale_final_img = thresholding_on_one_image(os.path.join(paths, image_name))
        cv2.imwrite(os.path.join(output_path, "colored", image_name), final_img)
        cv2.imwrite(os.path.join(output_path, "grayscale", image_name), grayscale_final_img)
        
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, default="archive/images", help="Path to the input image")
    argparser.add_argument("--output", type=str, default="archive/thresholding", help="Path to the output image")
    args = argparser.parse_args()
    thresholding_on_multiple_images(args.input, args.output)
    print(f"Done, the images are saved in {args.output}")
    
main()
        