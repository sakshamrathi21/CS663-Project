import cv2
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
from PIL import Image


def encoder(image_path, mask_path, masked_image_path):
    """
    Generates a mask from an input image using Canny edge detection,
    applies the mask to the image, and saves the results.

    Parameters:
    - image_path: Path to the input image (3-channel RGB image).
    - mask_path: Path to save the generated mask (must have .pbm extension).
    - masked_image_path: Path to save the masked image.
    """

    # Read the input image
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    # Pad the image with a border (same as MATLAB)
    im = cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    m, n, _ = im.shape

    # Convert to grayscale and apply Canny edge detection
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_im, 100, 200)  # Adjust thresholds as needed

    # Display edges
    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.show()

    # Create a white mask (3-channel, 255 for all pixels)
    mask_im = np.ones((m, n, 3), dtype=np.uint8) * 255

    # Define the window size for the neighborhood exclusion (same as MATLAB)
    window = 7
    pd = (window - 1) // 2

    # Pad the mask for edge handling
    padded_mask = cv2.copyMakeBorder(mask_im, pd, pd, pd, pd, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Exclude neighbors around detected edges in the mask
    for i in range(pd, pd + m):
        for j in range(pd, pd + n):
            if edges[i - pd, j - pd]:  # If edge is detected
                padded_mask[i - pd:i + pd + 1, j - pd:j + pd + 1, :] = 0

    # Extract the mask out of the padded array
    mask_im = padded_mask[pd:pd + m, pd:pd + n, :]

    # Generate the masked image (apply the mask to the input image)
    res_im = (1 - (mask_im / 255.0)) * im
    res_im = res_im.astype(np.uint8)

    # Display the mask and the result
    plt.figure()
    plt.imshow(mask_im[:, :, 0], cmap='gray')
    plt.title('Mask')
    plt.show()

    plt.figure()
    plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
    plt.title('Masked Result')
    plt.show()

    # Save the mask and the masked image
    save_pbm_matlab_style(mask_im, mask_path)  # Save the mask in MATLAB-style PBM
    cv2.imwrite(masked_image_path, res_im)  # Save the masked image


def save_pbm_matlab_style(mask_im, mask_path):
    """
    Mimic MATLAB's behavior of saving a 3D logical mask as a PBM file.

    Parameters:
    - mask_im: 3D logical mask (numpy array of type bool or uint8).
    - mask_path: Path to save the PBM file.
    """
    # Ensure all channels are identical
    if not np.all(mask_im[:, :, 0] == mask_im[:, :, 1]):
        raise ValueError("MATLAB assumes identical channels for logical masks. Ensure uniformity across channels.")

    # Use the first channel (MATLAB-style behavior)
    single_channel = mask_im[:, :, 0].astype(bool)  # Convert to binary (logical)

    # Save as a binary PBM file using Pillow
    img = Image.fromarray(single_channel)
    img.save(mask_path, format="PPM")  # PPM supports P4 PBM format internally


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Encoder for image edge masking")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("mask_path", type=str, help="Path to save the generated mask (must end with .pbm)")
    parser.add_argument("masked_image_path", type=str, help="Path to save the masked image")

    args = parser.parse_args()

    # Call the encoder function with the parsed arguments
    encoder(args.image_path, args.mask_path, args.masked_image_path)
