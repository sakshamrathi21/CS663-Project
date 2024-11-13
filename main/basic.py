import math
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from algorithms.huffman import HuffmanTree
from collections import Counter
from algorithms.helper import dct2d, quantization, create_quantization_matrix
from config.config import Config

config = Config()
# Load and preprocess the image
image = plt.imread('../images/jpgb.png')
if image.shape[-1] == 4:
    image = image[..., :3]
grayscale_image = rgb2gray(image)
grayscale_image = grayscale_image[:824, :824]  # Crop if needed
grayscale_image = grayscale_image * 255

# Apply DCT, Quantization, and Reconstruction
dct_image = dct2d(grayscale_image.copy())
quality = config.default_quality
quantized_dct_image = quantization(dct_image.copy(), quality)

# Encode the quantized DCT coefficients using Huffman coding
huffman_tree = HuffmanTree()
flat_quantized_data = quantized_dct_image.flatten().tolist()
frequency = Counter(flat_quantized_data)
huffman_tree.build_tree(frequency)
encoded_data = huffman_tree.encode(flat_quantized_data)

# Decode the Huffman encoded data
decoded_flat_quantized_data = np.array(huffman_tree.decode(encoded_data), dtype=np.int32)
quantized_dct_image = decoded_flat_quantized_data.reshape(*quantized_dct_image.shape)

reconstructed_dct_image = quantization(quantized_dct_image, quality, inverse=True)
reconstructed_image = dct2d(reconstructed_dct_image, inverse=True)

# Rescale to 8-bit integer range for display
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

plt.imshow(reconstructed_image, cmap='gray')
plt.show()