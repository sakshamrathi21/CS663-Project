import sys
sys.path.append('..')
from include.common_imports import *

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
    # print(quality, scale)
    quantization_matrix = np.floor((scale * base_matrix + 50) / 100).astype(np.int32)
    quantization_matrix[quantization_matrix == 0] = 1
    return quantization_matrix

def quantization(matrix, quality=Config.default_quality, inverse=False):
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
    # print(metadata, quantized_data)
    # If the type of values of huffman_codes is int, convert to float:
    metadata["huffman_codes"] = {
        int(k) if isinstance(k, np.integer) else k: 
        v if isinstance(v, str) else str(v)  # Convert value to string if not already
        for k, v in metadata["huffman_codes"].items()
    }
    
    with open(filename, 'wb') as file:
        # Save metadata as JSON for easy parsing
        file.write(json.dumps(metadata).encode('utf-8') + b'\n')
        # Save the Huffman encoded data as binary
        pickle.dump(quantized_data, file)


def load_compressed_image(filename, quality=Config.default_quality):
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

def calculate_bpp(encoded_data, image_shape):
    total_bits = len(encoded_data) * Config.bits_per_symbol  # Assuming bits_per_symbol is defined
    num_pixels = image_shape[0] * image_shape[1]
    return total_bits / num_pixels

def calculate_rmse(original, compressed):
    return np.sqrt(mean_squared_error(original, compressed))

def rmse_vs_bpp_plot(bpp_results, rmse_results, image_paths, plot_path):
    plt.figure(figsize=(12, 8))
    for i in range(len(image_paths)):
        plt.plot(bpp_results[i], rmse_results[i], label=f'Image {i+1}')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Images")
    plt.tight_layout()
    plt.xlabel('BPP')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. BPP for each image')
    plt.legend()
    plt.savefig(plot_path)

def get_image_paths():
    image_paths = glob.glob(Config.dataset_path)
    num_random_images = Config.basic_step_5_num_images
    image_paths = random.sample(image_paths, num_random_images)
    return image_paths

def get_gray_scale_image(image_path):
    image = plt.imread(image_path)
    if image.shape[-1] == 4:
        image = image[..., :3]
    grayscale_image = rgb2gray(image)
    grayscale_image = grayscale_image[:824, :824]  # Crop if needed
    grayscale_image = grayscale_image * 255
    return grayscale_image

# Get an array of index pairs to use for following a zigzag scan of a block
def get_zigzag_order(block_size):
    indices = np.indices((block_size, block_size)).transpose(1, 2, 0)
    flipped = np.fliplr(indices)
    order = np.zeros((block_size ** 2, 2), dtype=np.int64)
    upwards = True

    i = 0
    for offset in range(block_size - 1, -block_size, -1):
        diagonal = np.diagonal(flipped, offset=offset).transpose()

        if offset % 2 == 1:
            diagonal = np.flip(diagonal, axis=1)

        for coordinates in diagonal:
            order[i] = coordinates
            i += 1
    return order

# Perform runlength encoding for a channel
def runlength_encode(channel):
    zigzag_order = get_zigzag_order(8)
    (height, width) = channel.shape
    pairs = []

    for y in range(height // 8):
        for x in range(width // 8):
            block = channel[8*y:8*y+8, 8*x:8*x+8]
            skip = 0

            for [i, j] in zigzag_order:
                value = block[i, j]
                if value == 0:
                    skip += 1
                else:
                    pairs.append([skip, value])
                    skip = 0

            pairs.append([0, 0])
    return np.array(pairs, dtype=np.int16).flatten()

# Decode the runlength pairs and reconstruct the original channel
def runlength_decode(height, width, pairs):
    # print(pairs)
    pairs = pairs.reshape((-1, 2))
    zigzag_order = get_zigzag_order(8)
    matrix = np.zeros((height, width))
    y_max = height // 8
    x_max = width // 8

    # Indices of current block
    y = 0
    x = 0

    # Index of position within zigzag scan
    index = 0

    for [skip, value] in pairs:
        if skip == 0 and value == 0:
            x += 1
            if x == x_max:
                y += 1
                x = 0
            index = 0
        else:
            index += skip
            i, j = zigzag_order[index]
            matrix[y*8+i, x*8+j] = value
            index += 1

    return matrix

def load_compressed_image_runlength(filename, quality=Config.default_quality):
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
    huffman_decoded_data = huffman_tree.decode(encoded_data)
    # print(huffman_decoded_data)
    huffman_decoded_data = np.array(huffman_decoded_data, dtype=np.int16)
    decoded_flat_quantized_data = runlength_decode(metadata["image_shape"][0], metadata["image_shape"][1], huffman_decoded_data)
    # print(decoded_flat_quantized_data)
    # Apply Run-Length Decoding
    # decoded_flat_quantized_data = run_length_decoding(huffman_decoded_data)
    # Convert to array and reshape into 8x8 blocks using inverse zigzag
    image_shape = metadata["image_shape"]
    # quantized_dct_image = np.zeros(image_shape, dtype=int)
    
    # block_index = 0
    # for i in range(image_shape[0] // 8):
    #     for j in range(image_shape[1] // 8):
    #         flat_block = decoded_flat_quantized_data[block_index * 64 : (block_index + 1) * 64]
    #         quantized_dct_image[i*8:(i+1)*8, j*8:(j+1)*8] = inverse_zigzag_scan(flat_block)
    #         block_index += 1

    # Inverse quantization and inverse DCT
    quantized_dct_image = decoded_flat_quantized_data.reshape(image_shape)
    reconstructed_dct_image = quantization(quantized_dct_image, quality, inverse=True)
    reconstructed_image = dct2d(reconstructed_dct_image, inverse=True)
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    
    return reconstructed_image

def show_images_side_by_side(reconstructed_image, grayscale_image, title="Reconstructed vs. Grayscale Image"):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(reconstructed_image, cmap='gray')
    ax[0].set_title('Reconstructed Image')
    ax[1].imshow(grayscale_image, cmap='gray')
    ax[1].set_title('Grayscale Image')
    # plt.show()
    plt.savefig(title)
    plt.close()

def convert_to_grayscale_bmp(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to grayscale
        grayscale_img = img.convert("L")
        # Create an in-memory file
        bmp_buffer = io.BytesIO()
        # Save the grayscale image as BMP in the buffer
        grayscale_img.save(bmp_buffer, format="BMP")
        # Move the cursor to the beginning of the buffer
        bmp_buffer.seek(0)
        # Convert the grayscale image to a NumPy array
        np_array = np.array(grayscale_img, dtype=np.float64)
    return np_array


def apply_jpeg_compression(np_array, save_path, quality=50):
    """
    Apply JPEG compression to a NumPy array and save the compressed image.

    Args:
        np_array (np.ndarray): Grayscale image as a NumPy array.
        save_path (str): Path to save the compressed JPEG image.
        quality (int): Compression quality (1-95, higher is better quality).
    
    Returns:
        np.ndarray: NumPy array of the compressed image.
    """
    if quality < 1 or quality > 100:
        raise ValueError(f"Quality value must be between 1 and 100. Got {quality}.")
    # Convert NumPy array to PIL Image
    image = Image.fromarray(np_array.astype(np.uint8), mode='L')  # 'L' for grayscale
    # print(quality, "hello")
    quality = int(quality)
    # Save the image in JPEG format with specified quality
    image.save(save_path, "JPEG", quality=quality)
    
    # Reload the compressed image as a NumPy array
    compressed_image = Image.open(save_path).convert('L')
    return np.array(compressed_image)

def show_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()