import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from algorithms.huffman import HuffmanTree
from collections import Counter
from algorithms.helper import dct2d, quantization, create_quantization_matrix, save_compressed_image, load_compressed_image
from config.config import Config

config = Config()
image = plt.imread('../images/jpgb.png')
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
plt.imshow(reconstructed_image, cmap='gray')
plt.show()