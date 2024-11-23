import numpy as np
class Config:
    patch_size = (8, 8)
    default_quality = 50
    default_components = 24
    bits_per_symbol = 1
    basic_step_5_num_images = 20
    dataset_path = '../images/msrcorid/miscellaneous/*.JPG'
    cartoon_path = '../images/cartoonset10k/*.png'
    quality_factors = np.linspace(1, 100, 20)
    num_components_list = np.linspace(8,64,8)
    # quality_factors = [1, 2, 10, 50, 80]