import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *

def colour(num_paths=20):
    config = Config()
    image_paths = get_image_paths()[0:num_paths]
    image_paths = ['../images/jpgb.png']
    quality_factors = Config.quality_factors
    quality_factors = [2, 10, 50, 80]
    bpp_results = []
    rmse_results = []

    for image_path in image_paths:
        image_copy = convert_to_rgb_bmp(image_path)
        original_image_shape = image_copy.shape
        # show_image(image_copy/255, title="Original", cmap='brg')
        bpp_per_image = []
        rmse_per_image = []
        for quality in quality_factors:
            image = image_copy.copy()
            yuv_image = rgb_image_to_yuv(image)
            y_channel, u_channel, v_channel = subsampling(yuv_image)

            # Padding
            y_channel = pad_image(y_channel)
            u_channel = pad_image(u_channel)
            v_channel = pad_image(v_channel)

            # Store the Padded Image Shape
            image_shape = (y_channel.shape[0], y_channel.shape[1], image.shape[2])

            # DCT
            dct_y_channel = dct2d(y_channel)
            dct_u_channel = dct2d(u_channel)
            dct_v_channel = dct2d(v_channel)

            quality = int(quality)
            # Quantization
            quantized_y_channel = quantization(dct_y_channel, quality)
            quantized_u_channel = quantization(dct_u_channel, quality)
            quantized_v_channel = quantization(dct_v_channel, quality)

            # Flatten
            flat_quantized_y_channel = quantized_y_channel.flatten().tolist()
            flat_quantized_u_channel = quantized_u_channel.flatten().tolist()
            flat_quantized_v_channel = quantized_v_channel.flatten().tolist()

            # Sizes
            y_size = len(flat_quantized_y_channel)
            u_size = len(flat_quantized_u_channel)
            v_size = len(flat_quantized_v_channel)

            # Huffman Encoding
            flat_quantized_data = flat_quantized_y_channel + flat_quantized_u_channel + flat_quantized_v_channel
            huffman_tree = HuffmanTree()
            frequency = Counter(flat_quantized_data)
            if len(frequency) == 1:
                continue
            huffman_tree.build_tree(frequency)
            encoded_data = huffman_tree.encode(flat_quantized_data)

            # Save and Load
            save_compressed_rgb_image("compressed_image.bin", encoded_data, y_size, u_size, v_size, image_shape, patch_size=Config.patch_size, huffman_tree=huffman_tree)
            reconstructed_image = load_compressed_rgb_image("compressed_image.bin", quality=quality)

            # Trim back to original shape
            reconstructed_image = reconstructed_image[0:original_image_shape[0], 0:original_image_shape[1], :]

            # show_image(reconstructed_image, title=f"Reconstructed for quality = {quality}", cmap='brg')

            rmse = calculate_rmse(image, reconstructed_image)
            bpp = calculate_bpp(encoded_data, image.shape)
            print(f"Quality: {quality}, RMSE: {rmse}, BPP: {bpp}")
            show_images_side_by_side(reconstructed_image, image/255, title=f"../results/Quality: {quality}_comparison_colour.png")
            bpp_per_image.append(bpp)
            rmse_per_image.append(rmse)
        bpp_results.append(bpp_per_image)
        rmse_results.append(rmse_per_image)
    return bpp_results, rmse_results