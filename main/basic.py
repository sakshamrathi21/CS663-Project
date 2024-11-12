import numpy as np
import json
import pickle
from scipy.fftpack import idct
import matplotlib.pyplot as plt

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

def load_compressed_image(filename):
    """
    Load and decompress the image data from a file.
    
    Parameters:
    - filename: Name of the file to load data.
    
    Returns:
    - Reconstructed image from the compressed data.
    """
    with open(filename, 'rb') as file:
        # Read metadata
        metadata = json.loads(file.readline().decode('utf-8'))
        # Read encoded data
        encoded_data = pickle.load(file)
    
    # Reconstruct Huffman tree from saved codes
    huffman_tree = HuffmanTree()
    huffman_tree.codes = metadata["huffman_codes"]
    
    # Decode the Huffman encoded data
    decoded_quantized_data = np.array(huffman_tree.decode(encoded_data), dtype=np.int32)
    
    # Reshape decoded data into patches and perform inverse DCT
    patch_size = metadata["patch_size"]
    image_shape = metadata["image_shape"]
    patches_shape = (image_shape[0] // patch_size[0], image_shape[1] // patch_size[1])
    quantized_patches = decoded_quantized_data.reshape(patches_shape + patch_size)
    
    # Inverse DCT for each patch to reconstruct the image
    reconstructed_patches = np.array([idct(idct(patch.T, norm='ortho').T, norm='ortho') 
                                      for patch in quantized_patches.reshape(-1, *patch_size)])
    reconstructed_image = reconstructed_patches.reshape(image_shape)
    
    return reconstructed_image

# Usage Example:
# Assuming `encoded_data` is the Huffman encoded output and `image_shape` is the original image's shape
# save_compressed_image("compressed_image.bin", encoded_data, image_shape, patch_size=(8, 8), huffman_tree=huffman_tree)

# Load and display the decompressed image
reconstructed_image = load_compressed_image("compressed_image.bin")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()
