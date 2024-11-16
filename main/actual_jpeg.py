import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *

def jpeg():
    config = Config()
    image_paths = get_image_paths()
    quality_factors = Config.quality_factors
    bpp_results = []
    rmse_results = []
    # print(image_paths)
    for image_path in image_paths:
        # grayscale_image_copy = get_gray_scale_image(image_path)
        grayscale_image_copy = convert_to_grayscale_bmp(image_path)
        # original_image = Image.open(image_path).convert("L")
        bpp_per_image = []
        rmse_per_image = []
        for quality in quality_factors:
            compressed_image_array = apply_jpeg_compression(grayscale_image_copy, "../results/compressed_image.jpeg", quality=quality)
            grayscale_image_copy = grayscale_image_copy.copy()
            rmse = np.sqrt(np.mean((compressed_image_array - grayscale_image_copy) ** 2))
            compressed_file_size_bytes = os.path.getsize("../results/compressed_image.jpeg")
            compressed_file_size_bits = compressed_file_size_bytes * 8
            total_pixels = grayscale_image_copy.size
            bpp = compressed_file_size_bits / total_pixels
            bpp_per_image.append(bpp)
            rmse_per_image.append(rmse)
        bpp_results.append(bpp_per_image)
        rmse_results.append(rmse_per_image)
    return bpp_results, rmse_results

