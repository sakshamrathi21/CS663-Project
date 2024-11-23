import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *
from algorithms.advanced_helper import *

def basic(num_paths=20):
    config = Config()
    image_paths = get_image_paths()[0:num_paths]
    # image_paths = ['../images/jpgb.png']
    quality_factors = Config.quality_factors
    # quality_factors = [2, 10, 50, 80]
    bpp_results = []
    rmse_results = []

    for image_path in image_paths:
        size_in_bits = encode(image_path)
        rmse = decode(image_path)
        print("Size in bits: ", size_in_bits)
        print("RMSE: ", rmse)
        bpp_results.append(size_in_bits)
        rmse_results.append(rmse)
basic(1)