import numpy as np
from scipy.fftpack import dct
from skimage.util import view_as_blocks
from skimage import io, color
import sys
sys.path.append('..')
from config.config import Config

class Dct_f:
    config = Config()
    def compute_dct_on_patches(image, patch_size=config.patch_size):
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        h, w = image.shape
        h -= h % patch_size[0]
        w -= w % patch_size[1]
        image = image[:h, :w]
        patches = view_as_blocks(image, block_shape=patch_size)
        patches_shape = patches.shape
        patches = patches.reshape(-1, patch_size[0], patch_size[1])
        dct_patches = np.array([dct(dct(patch.T, norm='ortho').T, norm='ortho') for patch in patches])
        dct_image = dct_patches.reshape(patches_shape)
        return dct_image

    def quantize_dct_coefficients(dct_patches, quant_matrix = config.quantization_matrix):
        assert dct_patches.shape[-2:] == quant_matrix.shape, "Quantization matrix must match the patch size"
        quantized_patches = np.round(dct_patches / quant_matrix).astype(np.int32)
        return quantized_patches
