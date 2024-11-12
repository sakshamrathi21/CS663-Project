import numpy as np
from scipy.fftpack import dct
from skimage.util import view_as_blocks
from skimage import io, color

class DCT:
    def compute_dct_on_patches(image, patch_size=(8, 8)):
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
