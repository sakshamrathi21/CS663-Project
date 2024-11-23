import cv2
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

def decoder(mask_path, masked_image_path, orig_image_path, save_path):
    """
    Decodes an image using the provided mask and performs homogenous diffusion
    to estimate the original image.

    Parameters:
    - mask_path: Path to the input mask (must be binary).
    - masked_image_path: Path to the masked image.
    - orig_image_path: Path to the original image (for PSNR calculation).
    - save_path: Path to save the restored image.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Read the input files
    res_im = cv2.imread(masked_image_path, cv2.IMREAD_COLOR)
    mask_im = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    orig_im = cv2.imread(orig_image_path, cv2.IMREAD_COLOR)

    if res_im is None or mask_im is None or orig_im is None:
        raise ValueError("One or more input files could not be loaded.")

    orig_im = cv2.copyMakeBorder(orig_im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Homogenous Diffusion parameters
    delta_t = 0.09
    max_time = 1000

    # Convert mask and image to double precision
    msk = np.expand_dims(mask_im.astype(np.float64) / 255, axis=-1)  # Normalize and expand mask
    msk = np.repeat(msk, 3, axis=2)  # Match 3 channels
    res_im = res_im.astype(np.float64)

    m, n, _ = res_im.shape

    # Perform diffusion
    for t in np.arange(0, max_time, delta_t):
        res_xx = res_im[:, [1, *range(n - 1)], :] - 2 * res_im + res_im[:, [*range(1, n), n - 2], :]
        res_yy = res_im[[1, *range(m - 1)], :, :] - 2 * res_im + res_im[[*range(1, m), m - 2], :, :]
        lap = res_xx + res_yy
        div = delta_t * lap
        res_im += div * msk

    # Convert back to uint8
    res_im = np.clip(res_im, 0, 255).astype(np.uint8)

    # Calculate PSNR
    mse = np.mean((orig_im - res_im) ** 2)
    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    print(f"PSNR: {psnr:.2f} dB")

    # Save the restored image
    cv2.imwrite(save_path, res_im)

    # Display the images
    plt.figure()
    plt.imshow(cv2.cvtColor(orig_im, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.show()

    plt.figure()
    plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
    plt.title("Restored Image")
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Decoder using homogenous diffusion")
    parser.add_argument("mask_path", type=str, help="Path to the input mask (binary .pbm file)")
    parser.add_argument("masked_image_path", type=str, help="Path to the masked image")
    parser.add_argument("orig_image_path", type=str, help="Path to the original image (for PSNR calculation)")
    parser.add_argument("save_path", type=str, help="Path to save the restored image")

    args = parser.parse_args()

    # Call the decoder function with the parsed arguments
    decoder(args.mask_path, args.masked_image_path, args.orig_image_path, args.save_path)
