import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *


def edgesMarrHildreth(img, sigma):
    # NEED TO ADD HYSTERESIS THRESHOLDING
    """
            finds the edges using MarrHildreth edge detection method...
            :param im : input image
            :param sigma : sigma is the std-deviation and refers to the spread of gaussian
            :return:
            a binary edge image...
    """
    size = int(2*(np.ceil(3*sigma))+1)

    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # LoG filter

    kern_size = kernel.shape[0]
    log = np.zeros_like(img, dtype=float)

    # applying filter
    for i in range(img.shape[0]-(kern_size-1)):
        for j in range(img.shape[1]-(kern_size-1)):
            window = img[i:i+kern_size, j:j+kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)

    zero_crossing = np.zeros_like(log)

    # computing zero crossing
    for i in range(log.shape[0]-(kern_size-1)):
        for j in range(log.shape[1]-(kern_size-1)):
            if log[i][j] == 0:
                if (log[i][j-1] < 0 and log[i][j+1] > 0) or (log[i][j-1] < 0 and log[i][j+1] < 0) or (log[i-1][j] < 0 and log[i+1][j] > 0) or (log[i-1][j] > 0 and log[i+1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j-1] > 0) or (log[i][j+1] > 0) or (log[i-1][j] > 0) or (log[i+1][j] > 0):
                    zero_crossing[i][j] = 255

    return log, zero_crossing


def decoder(mask_path, masked_image_path, orig_image_path, save_path):
    """
    Decodes an image using the provided mask and performs homogenous diffusion
    to estimate the original image.

    Parameters:
    - mask_path: Path to the input mask (must be binary).
    - masked_image_path: Path to the masked image.
    - orig_image_path: Path to the original image (for PSNR calculation).
    - save_path: Path to save the restored image.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Read the input files
    res_im = cv2.imread(masked_image_path, cv2.IMREAD_COLOR)
    mask_im = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    orig_im = cv2.imread(orig_image_path, cv2.IMREAD_COLOR)
    if res_im is None or mask_im is None or orig_im is None:
        raise ValueError("One or more input files could not be loaded.")

    orig_im = cv2.copyMakeBorder(orig_im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Homogenous Diffusion parameters
    delta_t = 0.09
    max_time = 100

    # Convert mask and image to double precision
    msk = np.expand_dims(mask_im.astype(np.float64) / 255, axis=-1)  # Normalize and expand mask
    msk = np.repeat(msk, 3, axis=2)  # Match 3 channels
    res_im = res_im.astype(np.float64)

    m, n, _ = res_im.shape

    # Perform diffusion
    for t in np.arange(0, max_time, delta_t):
        # print(f"Time: {t:.2f}")
        res_xx = res_im[:, [1, *range(n - 1)], :] - 2 * res_im + res_im[:, [*range(1, n), n - 2], :]
        res_yy = res_im[[1, *range(m - 1)], :, :] - 2 * res_im + res_im[[*range(1, m), m - 2], :, :]
        lap = res_xx + res_yy
        div = delta_t * lap
        res_im += div * msk

    # Convert back to uint8
    res_im = np.clip(res_im, 0, 255).astype(np.uint8)

    # Calculate PSNR
    # mse = np.mean((orig_im - res_im) ** 2)
    # psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    # print(f"PSNR: {psnr:.2f} dB")

    # Save the restored image
    cv2.imwrite(save_path, res_im)

    # I want to display the original and the reconstructed image:
    # plt.figure()
    # plt.imshow(cv2.cvtColor(orig_im, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.show()

    # plt.figure()
    # plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
    # plt.title("Restored Image")
    # plt.show()

    # Display the images
    # plt.figure()
    # plt.imshow(cv2.cvtColor(orig_im, cv2.COLOR_BGR2RGB))
    # plt.title("Original Image")
    # plt.show()

    # plt.figure()
    # plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
    # plt.title("Restored Image")
    # plt.show()
    return calculate_rmse(orig_im, res_im)

def encoder(image_path, mask_path, masked_image_path, window=7):
    """
    Generates a mask from an input image using Canny edge detection,
    applies the mask to the image, and saves the results.

    Parameters:
    - image_path: Path to the input image (3-channel RGB image).
    - mask_path: Path to save the generated mask (must have .pbm extension).
    - masked_image_path: Path to save the masked image.
    """

    # Read the input image
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    # Pad the image with a border (same as MATLAB)
    im = cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    m, n, _ = im.shape

    # Convert to grayscale and apply Canny edge detection
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_im, 100, 200)  # Adjust thresholds as needed

    # Display edges
    # plt.figure()
    # plt.imshow(edges, cmap='gray')
    # plt.title('Edges')
    # plt.show()

    # Create a white mask (3-channel, 255 for all pixels)
    mask_im = np.ones((m, n, 3), dtype=np.uint8) * 255

    # Define the window size for the neighborhood exclusion (same as MATLAB)
    # window = 7
    pd = (window - 1) // 2

    # Pad the mask for edge handling
    padded_mask = cv2.copyMakeBorder(mask_im, pd, pd, pd, pd, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Exclude neighbors around detected edges in the mask
    for i in range(pd, pd + m):
        for j in range(pd, pd + n):
            if edges[i - pd, j - pd]:  # If edge is detected
                padded_mask[i - pd:i + pd + 1, j - pd:j + pd + 1, :] = 0

    # Extract the mask out of the padded array
    mask_im = padded_mask[pd:pd + m, pd:pd + n, :]

    # Generate the masked image (apply the mask to the input image)
    res_im = (1 - (mask_im / 255.0)) * im
    res_im = res_im.astype(np.uint8)

    # Display the mask and the result
    # plt.figure()
    # plt.imshow(mask_im[:, :, 0], cmap='gray')
    # plt.title('Mask')
    # plt.show()

    # plt.figure()
    # plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
    # plt.title('Masked Result')
    # plt.show()

    # Save the mask and the masked image
    save_pbm_matlab_style(mask_im, mask_path)  # Save the mask in MATLAB-style PBM
    cv2.imwrite(masked_image_path, res_im)  # Save the masked image
    return im.shape[0]*im.shape[1]


def save_pbm_matlab_style(mask_im, mask_path):
    """
    Mimic MATLAB's behavior of saving a 3D logical mask as a PBM file.

    Parameters:
    - mask_im: 3D logical mask (numpy array of type bool or uint8).
    - mask_path: Path to save the PBM file.
    """
    # Ensure all channels are identical
    if not np.all(mask_im[:, :, 0] == mask_im[:, :, 1]):
        raise ValueError("MATLAB assumes identical channels for logical masks. Ensure uniformity across channels.")

    # Use the first channel (MATLAB-style behavior)
    single_channel = mask_im[:, :, 0].astype(bool)  # Convert to binary (logical)

    # Save as a binary PBM file using Pillow
    img = Image.fromarray(single_channel)
    img.save(mask_path, format="PPM")  # PPM supports P4 PBM format internally


def get_file_size_in_bits(file_path):
    # Get the file size in bytes using os.stat
    file_size_bytes = os.stat(file_path).st_size
    
    # Convert the size from bytes to bits (1 byte = 8 bits)
    file_size_bits = file_size_bytes * 8
    
    return file_size_bits

def encode(image_path, window=7):
    mask_path = '../temp/mask.pbm'
    masked_image_path = '../temp/masked_image.png'
    im_pixels = encoder(image_path, mask_path, masked_image_path, window)
    os.system("rm -rf ../temp/compressed")
    os.system("mkdir -p ../temp && mkdir -p ../temp/compressed")
    os.system("../jbigkit/pbmtools/pbmtojbg ../temp/mask.pbm ../temp/compressed/masked_image.jbg")
    os.system("zpaq a ../temp/compressed/masked_image.archive ../temp/compressed/masked_image.jbg ../temp/masked_image.png > /dev/null 2>&1")
    # os.system("ls -lR ../temp")
    size_in_bits = get_file_size_in_bits('../temp/compressed/masked_image.archive')
    # print(f"Size in bits: {size_in_bits}")
    return size_in_bits/im_pixels

def decode(image_path):
    os.system("zpaq x ../temp/compressed/masked_image.archive > /dev/null 2>&1")
    os.system("../jbigkit/pbmtools/jbgtopbm ../temp/compressed/masked_image.jbg ../temp/mask.pbm")
    rmse = decoder('../temp/mask.pbm', '../temp/masked_image.png', image_path, '../temp/decoded_image.png')
    os.system("rm -rf ../temp/compressed")
    return rmse
    # PATH=../jbigkit/pbmtools
    # # zpaq x ./compressed/im1.archive
    # # $PATH/pbmtools/jbgtopbm ./compressed/m_im1.jbg m_im1.pbm
    # zpaq x ./compressed/$1
    # $PATH/jbgtopbm ./compressed/$2 $3
    # zpaq x ./compressed/im1.archive
    # ../../../jbigkit/pbmtools/jbgtopbm ./compressed/m_im1.jbg m_im1.pbm