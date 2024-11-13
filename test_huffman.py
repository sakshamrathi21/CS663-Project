# Test huffman code on an image
from algorithms.huffman import HuffmanTree
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

image = plt.imread('images/jpgb.png')
if image.shape[-1] == 4:  # Check if there's an alpha channel
    image = image[..., :3]  # Keep only RGB channels

grayscale_image = rgb2gray(image)
grayscale_image = (grayscale_image * 255).astype(np.uint8)

# Create a Huffman tree and encode the image
huffman_tree = HuffmanTree()
# build_tree
frequency = {}
for i in range(256):
    frequency[i] = 0
for row in grayscale_image:
    for pixel in row:
        frequency[pixel] += 1
huffman_tree.build_tree(frequency)
# encode
encoded_data = huffman_tree.encode(grayscale_image.flatten().tolist())
print(encoded_data[:100])  # Print the first 100 encoded bits
# decode
decoded_data = huffman_tree.decode(encoded_data)
decoded_image = np.array(decoded_data).reshape(grayscale_image.shape)
# Plot the original and decoded images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(decoded_image, cmap='gray')
plt.title('Decoded Image')
plt.axis('off')
plt.show()
# The original and decoded images should look identical if the Huffman encoding/decoding is correct.