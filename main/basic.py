import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from algorithms.huffman import HuffmanTree
from collections import Counter
from algorithms.helper import dct2d, quantization, save_compressed_image, load_compressed_image, calculate_rmse, calculate_bpp
from config.config import Config

config = Config()
image = plt.imread('../images/msrcorid/aeroplanes/general/165_6531.JPG')
if image.shape[-1] == 4:
    image = image[..., :3]

grayscale_image = rgb2gray(image)
grayscale_image = grayscale_image[:824, :824]  # Crop if needed
grayscale_image = grayscale_image * 255
dct_image = dct2d(grayscale_image.copy())
quality = config.default_quality
quantized_dct_image = quantization(dct_image.copy())
huffman_tree = HuffmanTree()
flat_quantized_data = quantized_dct_image.flatten().tolist()
frequency = Counter(flat_quantized_data)
huffman_tree.build_tree(frequency)
encoded_data = huffman_tree.encode(flat_quantized_data)
image_shape = grayscale_image.shape
save_compressed_image("compressed_image.bin", encoded_data, image_shape, patch_size=(8, 8), huffman_tree=huffman_tree)
reconstructed_image = load_compressed_image("compressed_image.bin")

# # Print first 100 elements of the original and reconstructed image
# print("Original Image:")
# print(grayscale_image[:10, :10])
# print("Reconstructed Image:")
# print(reconstructed_image[:10, :10])

# print RMSE, BPP
rmse = calculate_rmse(grayscale_image, reconstructed_image)
bpp = calculate_bpp(encoded_data, grayscale_image.shape)
print(f"Quality: {config.default_quality}" + "\n" + f"RMSE: {rmse}" + "\n" + f"BPP: {bpp}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')    
plt.show()