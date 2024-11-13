import numpy as np
import math
import pickle
import json
import sys
sys.path.append('..')
from algorithms.huffman import HuffmanTree


def create_dct_matrix(n):
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a = math.sqrt(1 / n) if i == 0 else math.sqrt(2 / n)
            matrix[i][j] = a * math.cos(((2 * j + 1) * i * math.pi) / (2 * n))
    return matrix

def dct2d(matrix, inverse=False):
    dct_matrix = create_dct_matrix(8)
    dct_transpose = dct_matrix.T
    (height, width) = matrix.shape
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = matrix[y:y+8, x:x+8]
            if inverse:
                matrix[y:y+8, x:x+8] = np.dot(dct_transpose, np.dot(block, dct_matrix))
            else:
                matrix[y:y+8, x:x+8] = np.dot(dct_matrix, np.dot(block, dct_transpose))
    return matrix

def create_quantization_matrix(quality):
    base_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])
    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    quantization_matrix = np.floor((scale * base_matrix + 50) / 100).astype(np.int32)
    quantization_matrix[quantization_matrix == 0] = 1
    return quantization_matrix

def quantization(matrix, quality, inverse=False):
    q_matrix = create_quantization_matrix(quality)
    (height, width) = matrix.shape
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = matrix[y:y+8, x:x+8]
            if inverse:
                matrix[y:y+8, x:x+8] = block * q_matrix
            else:
                matrix[y:y+8, x:x+8] = np.round(block / q_matrix)
    return matrix

def save_compressed_image(filename, quantized_data, image_shape, patch_size, huffman_tree):
    """
    Save the compressed image data to a file.
    
    Parameters:
    - filename: Name of the file to save data.
    - quantized_data: Huffman-encoded data for DCT coefficients.
    - image_shape: Shape of the original image.
    - patch_size: Size of the patches (e.g., (8, 8)).
    - huffman_tree: Huffman tree used for encoding (stores codes and symbols).
    """
    # Prepare data for saving
    metadata = {
        "image_shape": image_shape,
        "patch_size": patch_size,
        "huffman_codes": huffman_tree.codes  # Save the Huffman codes for decoding
    }
    
    with open(filename, 'wb') as file:
        # Save metadata as JSON for easy parsing
        file.write(json.dumps(metadata).encode('utf-8') + b'\n')
        # Save the Huffman encoded data as binary
        pickle.dump(quantized_data, file)


def load_compressed_image(filename, quality):
    """
    Load and decompress the image data from a file.
    
    Parameters:
    - filename: Name of the file to load data.
    
    Returns:
    - Reconstructed image from the compressed data.
    """
    with open(filename, 'rb') as file:
        metadata = json.loads(file.readline().decode('utf-8'))
        encoded_data = pickle.load(file)
    huffman_tree = HuffmanTree()
    huffman_tree.codes = metadata["huffman_codes"]
    
    patch_size = metadata["patch_size"]
    image_shape = metadata["image_shape"]
    huffman_decoded_data = huffman_tree.decode(encoded_data)
    # Convert to floats, then round and cast to int32
    decoded_flat_quantized_data = np.array(huffman_decoded_data, dtype=np.float32)
    decoded_flat_quantized_data = np.round(decoded_flat_quantized_data).astype(np.int32)

    quantized_dct_image = decoded_flat_quantized_data.reshape(image_shape)

    reconstructed_dct_image = quantization(quantized_dct_image, quality, inverse=True)
    reconstructed_image = dct2d(reconstructed_dct_image, inverse=True)
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    return reconstructed_image