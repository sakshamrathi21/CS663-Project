import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *
from algorithms.advanced_helper import *

def basic(num_paths=20):
    config = Config()
    image_paths = get_image_paths()[0:num_paths]
    window_sizes = [3, 5, 7, 9, 11]
    bpp_results = []
    rmse_results = []

    for image_path in image_paths:
        for window_size in window_sizes:
            size_in_bits = encode(image_path, window_size)
            rmse = decode(image_path)
            print("Size in bits: ", size_in_bits)
            print("RMSE: ", rmse)
            bpp_results.append(size_in_bits)
            rmse_results.append(rmse)
    return bpp_results, rmse_results

