import numpy as np
class Config:
    patch_size = (8, 8)
    default_quality = 50
    bits_per_symbol = 1
    basic_step_5_num_images = 20
    dataset_path = '../images/msrcorid/miscellaneous/*.JPG'
    quality_factors = np.linspace(1, 100, 20)