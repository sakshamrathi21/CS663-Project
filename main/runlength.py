import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *

def runlength():
    config = Config()
    image_paths = get_image_paths()
    quality_factors = Config.quality_factors
    bpp_results = []
    rmse_results = []

    for image_path in image_paths:
        # grayscale_image_copy = get_gray_scale_image(image_path)
        grayscale_image_copy = convert_to_grayscale_bmp(image_path)
        bpp_per_image = []
        rmse_per_image = []
        for quality in quality_factors:
            grayscale_image = grayscale_image_copy.copy()
            dct_image = dct2d(grayscale_image.copy())
            quality = int(quality)
            quantized_dct_image = quantization(dct_image, quality=quality)
            huffman_tree = HuffmanTree()
            encoded_data = runlength_encode(quantized_dct_image)
            frequency = Counter(encoded_data)
            if len(frequency) == 1:
                continue
            huffman_tree.build_tree(frequency)
            
            encoded_data = huffman_tree.encode(encoded_data)
            save_compressed_image("compressed_image.bin", encoded_data, grayscale_image.shape, patch_size=Config.patch_size, huffman_tree=huffman_tree)
            reconstructed_image = load_compressed_image_runlength("compressed_image.bin", quality=quality)
            rmse = calculate_rmse(grayscale_image, reconstructed_image)
            bpp = calculate_bpp(encoded_data, grayscale_image.shape)
            print(f"Quality: {quality}, RMSE: {rmse}, BPP: {bpp}")
            bpp_per_image.append(bpp)
            rmse_per_image.append(rmse)
        bpp_results.append(bpp_per_image)
        rmse_results.append(rmse_per_image)
    return bpp_results, rmse_results