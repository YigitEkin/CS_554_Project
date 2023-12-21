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
    args = parser.parse_args()

    # Call ICM function with the parsed arguments
    sys.stdout.write(str(ICM(args)))


def potential(fi, fj):
    # Calculate the potential between two pixel values
    return float((fi - fj)) ** 2


def norm_image(x):
    # Normalize the image
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def energy_of_pixel(noisy_img, i, j, sigma2, beta):
    # Calculate the energy of a pixel based on its neighbors
    height, width = noisy_img.shape
    if i < 0 or i >= height - 1 or j < 0 or j >= width - 1:
        return 255 ** 2  # temp value for handling border cases
    return float((noisy_img[i][j] * noisy_img[i][j])) / (2.0 * sigma2) + beta * (
            potential(noisy_img[i][j - 1], noisy_img[i][j]) +
            potential(noisy_img[i][j + 1], noisy_img[i][j]) +
            potential(noisy_img[i - 1][j], noisy_img[i][j]) +
            potential(noisy_img[i + 1][j], noisy_img[i][j]) +
            potential(noisy_img[i - 1][j - 1], noisy_img[i][j]) +
            potential(noisy_img[i - 1][j + 1], noisy_img[i][j]) +
            potential(noisy_img[i + 1][j - 1], noisy_img[i][j]) +
            potential(noisy_img[i + 1][j + 1], noisy_img[i][j]))


def energy_of_possible_update(noisy_img, i, j, sigma2, beta, new_value):
    # Calculate the energy if a pixel value is updated
    height, width = noisy_img.shape
    if i < 0 or i >= height - 1 or j < 0 or j >= width - 1:
        return 255 ** 2  # temp value for handling border cases
    return float((noisy_img[i][j] * noisy_img[i][j])) / (2.0 * sigma2) + potential(noisy_img[i, j], new_value) + beta * (
            potential(noisy_img[i][j - 1], new_value) +
            potential(noisy_img[i][j + 1], new_value) +
            potential(noisy_img[i - 1][j], new_value) +
            potential(noisy_img[i + 1][j], new_value) +
            potential(noisy_img[i - 1][j - 1], new_value) +
            potential(noisy_img[i - 1][j + 1], new_value) +
            potential(noisy_img[i + 1][j - 1], new_value) +
            potential(noisy_img[i + 1][j + 1], new_value))

def average_pixel_distance(original_img, noisy_img):
    assert original_img.shape == noisy_img.shape, "The two images must have the same shape"
    return np.sum(np.abs(original_img - noisy_img)) / (original_img.shape[0] * original_img.shape[1])

def ICM(args):
    # Load the noisy image
    noisy_img = np.array(Image.open(args.image))
    height, width = noisy_img.shape

    sigma2 = args.sigma ** 2
    beta = args.beta

    # Perform ICM iterations
    for _ in tqdm(range(args.iter), desc="ICM iterations"):
        for i in tqdm(range(height - 1), desc="for each row"):
            for j in range(width - 1):
                # Calculate the energies of the pixel's neighbors
                energies_of_neighbors = [
                    energy_of_pixel(noisy_img, i, j - 1, sigma2, beta),
                    energy_of_pixel(noisy_img, i, j + 1, sigma2, beta),
                    energy_of_pixel(noisy_img, i - 1, j, sigma2, beta),
                    energy_of_pixel(noisy_img, i + 1, j, sigma2, beta),
                    energy_of_pixel(noisy_img, i - 1, j - 1, sigma2, beta),
                    energy_of_pixel(noisy_img, i - 1, j + 1, sigma2, beta),
                    energy_of_pixel(noisy_img, i + 1, j - 1, sigma2, beta),
                    energy_of_pixel(noisy_img, i + 1, j + 1, sigma2, beta)
                ]
                # Find the minimum energy and its index
                current_min = min(energies_of_neighbors)
                index = energies_of_neighbors.index(current_min)

                # Update the pixel value if a lower energy is found
                for x in range(256):
                    probability_of_update = energy_of_possible_update(noisy_img, i, j, sigma2, beta, x)
                    if current_min > probability_of_update:
                        current_min = probability_of_update
                        index = x
                        noisy_img[i][j] = index

        # Save the restored image
        cv2.imwrite(args.output, noisy_img)
    return f"Restored image saved to {args.output}"


if __name__ == '__main__':
    main()
