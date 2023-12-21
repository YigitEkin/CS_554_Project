import math
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import sys
import PIL.Image as Image


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="archive/images/3063.png", help="Path to the image you want to restore")
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations you want for ICM")
    parser.add_argument("--beta", type=float, default=2, help="Value of regularisation berween neighbors of a pixel")
    parser.add_argument("--sigma", type=float, default=1, help="Value of sigma")
    parser.add_argument("--output", type=str, default="restored_image.jpg", help="Path to the output image")
    parser.add_argument("--classes", type=int, default=10, help="Number of classes")
    args = parser.parse_args()

    # Call ICM function with the parsed arguments
    sys.stdout.write(str(ICM(args)))
    
def IoU(label_img, ground_truth, class_to_compare):
    intersection = np.sum(np.logical_and(label_img == class_to_compare, ground_truth == class_to_compare))
    union = np.sum(np.logical_or(label_img == class_to_compare, ground_truth == class_to_compare))
    return intersection / union
    
def calculate_probabilities(label_img, classes):
    probabilities = [0] * len(classes)
    for i in range(len(classes)):
        probabilities[i] = len(label_img[label_img == classes[i]]) / (len(label_img) * len(label_img[0]))
    return probabilities

def binary_potential_8_neighbors(label_img, i, j, i_new, j_new):
    if i < 0 or j < 0 or i >= label_img.shape[0] or j >= label_img.shape[1]:
        return 0
    else:
        return 0 if label_img[i][j] == label_img[i_new][j_new] else 1

def potential_of_8_neighbors(label_img, i, j, probabilities, beta):
    unary_potential = -1 * math.log(probabilities[label_img[i][j].astype(int)])
    binary_potential = [
                    binary_potential_8_neighbors(label_img, i, j - 1, i, j),
                    binary_potential_8_neighbors(label_img, i, j + 1, i, j),
                    binary_potential_8_neighbors(label_img, i - 1, j, i, j),
                    binary_potential_8_neighbors(label_img, i + 1, j, i, j),
                    binary_potential_8_neighbors(label_img, i - 1, j - 1, i, j),
                    binary_potential_8_neighbors(label_img, i - 1, j + 1, i, j),
                    binary_potential_8_neighbors(label_img, i + 1, j - 1, i, j),
                    binary_potential_8_neighbors(label_img, i + 1, j + 1, i, j)
                ]
        
    return unary_potential + sum(binary_potential) * beta

def energy_of_possible_updates(label_img, i, j, probabilities, beta, new_label):
    unary_potential = -1 * math.log(probabilities[new_label]) * 2
    label_img_copy = label_img.copy()
    label_img_copy[i][j] = new_label
    binary_potential = [
                    binary_potential_8_neighbors(label_img_copy,i, j - 1, i, j),
                    binary_potential_8_neighbors(label_img_copy, i, j + 1, i, j),
                    binary_potential_8_neighbors(label_img_copy, i - 1, j, i, j),
                    binary_potential_8_neighbors(label_img_copy, i + 1, j, i, j),
                    binary_potential_8_neighbors(label_img_copy, i - 1, j - 1, i, j),
                    binary_potential_8_neighbors(label_img_copy, i - 1, j + 1, i, j),
                    binary_potential_8_neighbors(label_img_copy, i + 1, j - 1, i, j),
                    binary_potential_8_neighbors(label_img_copy, i + 1, j + 1, i, j)
                ]
    
    return unary_potential + sum(binary_potential) * beta 

def ICM(args):
    # Load the noisy image
    label_img = np.array(Image.open(args.image))
    classes = np.arange(args.classes)
    probabilities = calculate_probabilities(label_img, classes)
    height, width = label_img.shape
    beta = args.beta

    print(probabilities)
    # Perform ICM iterations
    for _ in tqdm(range(args.iter), desc="ICM iterations"):
        for i in tqdm(range(height - 1), desc="for each row"):
            for j in range(width - 1):
                current_energy = 255 ** 2
                current_energy = potential_of_8_neighbors(label_img, i, j, probabilities, beta)
                class_to_update = label_img[i][j]
                for x in range(args.classes):
                    energy_of_update = energy_of_possible_updates(label_img, i, j, probabilities, beta, x)
                    if current_energy > energy_of_update:
                        #print(f"energy_of_update: {energy_of_update}, current_energy: {current_energy}")
                        current_energy = energy_of_update
                        class_to_update = x
                label_img[i][j] = class_to_update

        # Save the restored image
        cv2.imwrite(args.output, label_img)
    return f"Segmented image saved to {args.output}"


if __name__ == '__main__':
    main()
