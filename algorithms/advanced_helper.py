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


def cannyEdge(img_path, colour=False):
    if not colour:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(img_path, cv.IMREAD_COLOR)
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(img,100,200)
    return edges

def saveAsPBM(edges, pbm_path):
    # Ensure edges are binary (0 and 255) and save as PBM
    edges_binary = (edges > 0).astype(np.uint8) * 255  # Convert to binary
    with open(pbm_path, "wb") as f:
        f.write(b"P4\n")
        f.write(f"{edges.shape[1]} {edges.shape[0]}\n".encode())
        f.write(edges_binary.tobytes())

def jbigEncode(pbm_path, jbig_path, jbigkit_path="./jbigkit/pbmtools"):
    command = f"{jbigkit_path}/pbmtojbg {pbm_path} {jbig_path}"
    os.system(command)


def jbigDecode(jbig_path, pbm_path, jbigkit_path="./jbigkit/pbmtools"):
    """
    Decodes a JBIG file into a PBM file using JBIG-KIT.

    Parameters:
    - jbig_path: Path to the input JBIG file.
    - pbm_path: Path to save the output PBM file.
    - jbigkit_path: Path to the directory containing the jbgtopbm tool.

    Returns:
    - pbm_path: Path to the decoded PBM file.
    """
    command = f"{jbigkit_path}/jbgtopbm {jbig_path} {pbm_path}"
    os.system(command)
    if os.path.exists(pbm_path):
        print(f"Decoded PBM file saved at {pbm_path}")
        return pbm_path
    else:
        raise FileNotFoundError("Decoding failed. Ensure JBIG-KIT tools are set up correctly.")

def displayPBM(pbm_path):
    """
    Displays a PBM file using OpenCV.

    Parameters:
    - pbm_path: Path to the PBM file.
    """
    import cv2 as cv
    pbm_image = cv.imread(pbm_path, cv.IMREAD_GRAYSCALE)
    assert pbm_image is not None, "Failed to load PBM file. Check file path."
    cv.imshow("Decoded PBM", pbm_image)
    cv.waitKey(0)
    cv.destroyAllWindows()