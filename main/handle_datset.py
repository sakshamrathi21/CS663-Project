import glob
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from algorithms.huffman import HuffmanTree
from collections import Counter
from algorithms.helper import dct2d, quantization, save_compressed_image, load_compressed_image, calculate_rmse, calculate_bpp
from config.config import Config
import random

config = Config()
image_paths = glob.glob('../images/msrcorid/aeroplanes/general/*.JPG')
num_random_images = config.basic_step_5_num_images
image_paths = random.sample(image_paths, num_random_images)
quality_factors = np.linspace(1, 100, 20)
bpp_results = []
rmse_results = []

for image_path in image_paths:
    image = plt.imread(image_path)
    if image.shape[-1] == 4:
        image = image[..., :3]
    grayscale_image = rgb2gray(image)
    grayscale_image = grayscale_image[:824, :824]  # Crop if needed
    grayscale_image = grayscale_image * 255
    grayscale_image_copy = grayscale_image.copy()
    
    bpp_per_image = []
    rmse_per_image = []
    
    print("hello", image_path)

    for quality in quality_factors:
        grayscale_image = grayscale_image_copy.copy()
        # print(np.sum(grayscale_image))
        dct_image = dct2d(grayscale_image.copy())
        # print("hello", quality)
        quality = int(quality)
        quantized_dct_image = quantization(dct_image, quality=quality)
        # Huffman encoding
        huffman_tree = HuffmanTree()
        flat_quantized_data = quantized_dct_image.flatten().tolist()
        frequency = Counter(flat_quantized_data)
        # print(len(frequency))
        if len(frequency) == 1:
            continue
        huffman_tree.build_tree(frequency)
        encoded_data = huffman_tree.encode(flat_quantized_data)
        # print(frequency)
        # Save and load the compressed image

        save_compressed_image("compressed_image.bin", encoded_data, grayscale_image.shape, patch_size=(8, 8), huffman_tree=huffman_tree)
        reconstructed_image = load_compressed_image("compressed_image.bin", quality=quality)
        
        # Calculate RMSE and BPP
        rmse = calculate_rmse(grayscale_image, reconstructed_image)
        bpp = calculate_bpp(encoded_data, grayscale_image.shape)

        print(f"Quality: {quality}, RMSE: {rmse}, BPP: {bpp}")
        
        # bpp_per_image.append(bpp)
        bpp_per_image.append(bpp)
        rmse_per_image.append(rmse)
    
    bpp_results.append(bpp_per_image)
    rmse_results.append(rmse_per_image)

# Plot RMSE vs. BPP for each image
for i in range(len(image_paths)):
    plt.plot(bpp_results[i], rmse_results[i], label=f'Image {i+1}')
plt.xlabel('BPP (Bits Per Pixel)')
plt.ylabel('RMSE (Relative Root Mean Squared Error)')
plt.title('RMSE vs BPP for Different Images and Quality Factors')
plt.legend()
plt.savefig('../results/basic.png')
plt.show()