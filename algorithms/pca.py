# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.util import view_as_windows
# from PIL import Image

# def load_grayscale_image(filepath):
#     """
#     Loads an image and converts it to grayscale.
#     """
#     image = Image.open(filepath).convert('L')  # Convert to grayscale ('L' mode)
#     return np.array(image)
 
def pad_image(image, patch_size=8):
    """
    Pads the image so that its dimensions are divisible by patch_size.
    :param image: The image to be padded
    :param patch_size: The patch size (default is 8)
    :return: Padded image
    """
    h, w = image.shape
    # Calculate padding required for height and width
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # Pad the image
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    return padded_image
# def visualize_patches(patches, patch_size=8):
#     """
#     Visualizes the patches extracted from the image.
#     """
#     num_patches = patches.shape[0]
#     num_columns = int(np.sqrt(num_patches))
#     num_rows = num_patches // num_columns

#     plt.figure(figsize=(num_columns * 2, num_rows * 2))
#     for i in range(num_patches):
#         plt.subplot(num_rows, num_columns, i + 1)
#         plt.imshow(patches[i], cmap='gray')
#         plt.axis('off')
#         plt.title(f"Patch {i + 1}")
#     plt.tight_layout()
#     plt.show()
# # Extract patches
# def extract_patches(image, patch_size=8):

#     patches = view_as_windows(image, (patch_size, patch_size), step=patch_size)
#     patches = patches.reshape(-1, patch_size, patch_size)
#     # print(patches)
#     # visualize_patches(patches)
#     return patches

# # Compute correlation matrices
# def compute_correlation_matrices(patches):
#     num_patches, patch_size, _ = patches.shape
#     col_corr = np.zeros((patch_size,patch_size))
#     for i in range(num_patches):
#         for j in range(patch_size):
#             col_corr = col_corr + patches[i][:,j]@(patches[i][:,j].T)
#     col_corr = col_corr/(num_patches - 1)
#     row_corr = np.zeros((patch_size,patch_size))
#     for i in range(num_patches):
#         for j in range(patch_size):
#             row_corr = row_corr + (patches[i][j,:].T)@(patches[i][j,:])
#     row_corr = row_corr/(num_patches - 1)

#     return row_corr, col_corr

# # Compute eigenvectors
# def compute_eigenvectors(matrix):
#     eigvals, eigvecs = np.linalg.eigh(matrix)
#     return eigvecs

# # Compute Kronecker product
# def compute_kronecker_basis(VR, VC):
#     """
#     Computes the Kronecker product of VR and VC.
#     """
#     print(np.kron(VR,VC).shape)
#     return np.kron(VR, VC)

# # Visualize DCT-like bases
# def visualize_basis(basis_matrix, patch_size=8):
#     """
#     Visualizes the bases formed from the Kronecker product.
#     """
#     num_bases = patch_size ** 2
#     plt.figure(figsize=(10, 10))
#     for i in range(num_bases):
#         basis = basis_matrix[:, i].reshape(patch_size, patch_size)
#         plt.subplot(patch_size, patch_size, i + 1)
#         plt.imshow(basis, cmap='gray')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# # Compress patches
# def compress_patches(patches, basis_matrix, num_components):
#     """
#     Compress patches using the specified number of components from the basis.
#     :param patches: Flattened patches of shape (num_patches, patch_size^2)
#     :param basis_matrix: Basis matrix of shape (patch_size^2, patch_size^2)
#     :param num_components: Number of basis components to retain
#     :return: Compressed representation
#     """
#     # print(basis_matrix.shape)
#     patches_reshaped = patches.reshape(patches.shape[0], -1)  # Flatten patches
#     # print(patches_reshaped.shape)
#     basis_subset = basis_matrix[:, :num_components]  # Select top components
#     compressed_patches = np.dot(patches_reshaped, basis_subset)  # Project onto the basis
#     return compressed_patches, basis_subset

# # Restore patches
# def restore_patches(compressed_patches, basis_subset, patch_size):
#     """
#     Restore patches from compressed representation.
#     :param compressed_patches: Compressed patches (low-dimensional representation)
#     :param basis_subset: Selected basis vectors
#     :param patch_size: Size of each patch
#     :return: Reconstructed patches
#     """
#     restored_patches = np.dot(compressed_patches, basis_subset.T)  # Reconstruct patches
#     restored_patches = restored_patches.reshape(-1, patch_size, patch_size)  # Reshape to original patch size
#     return restored_patches

# # Reconstruct image from patches
# def reconstruct_image(patches, image_shape, patch_size):
#     """
#     Reconstructs the image from patches.
#     :param patches: Flattened patches of shape (num_patches, patch_size, patch_size)
#     :param image_shape: Original shape of the image
#     :param patch_size: Size of the patches
#     :return: Reconstructed image
#     """
#     h, w = image_shape
#     reconstructed = np.zeros((h, w))
#     patch_idx = 0

#     # Ensure that the number of patches matches the expected grid of the image
#     expected_patches = (h // patch_size) * (w // patch_size)
#     assert len(patches) == expected_patches, f"Mismatch in number of patches: expected {expected_patches}, got {len(patches)}"

#     # Fill the reconstructed image with patches
#     for i in range(0, h, patch_size):
#         for j in range(0, w, patch_size):
#             # Ensure that we don't go out of bounds with patch_idx
#             if patch_idx >= len(patches):
#                 break

#             # Determine the size of the patch to place (handle boundaries)
#             if i + patch_size > h:
#                 size_x = h - i
#             else:
#                 size_x = patch_size
                
#             if j + patch_size > w:
#                 size_y = w - j
#             else:
#                 size_y = patch_size

#             # Place the patch in the reconstructed image
#             reconstructed[i:i + size_x, j:j + size_y] = patches[patch_idx][:size_x, :size_y]
#             patch_idx += 1

#             if patch_idx >= len(patches):
#                 break

#     return reconstructed


# # Compute reconstruction error
# def compute_reconstruction_error(original_image, reconstructed_image):
#     """
#     Computes the mean squared error between the original and reconstructed image.
#     """
#     return np.mean((original_image - reconstructed_image) ** 2)

# # Main script (extended)
# if __name__ == "__main__":
#     # Example image (replace with your actual image data)
#     image_path = 'jpgb.png'  # Update with your actual image path
#     image = load_grayscale_image(image_path)
#     print(image.shape )
#     # Display the image (optional)
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.title("Loaded Grayscale Image")
#     plt.show()
#     # Pad the image
#     image = pad_image(image, patch_size=8)
#     # Parameters
#     patch_size = 8
#     num_components_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]  # Test different numbers of components

#     # Step 1: Extract patches
#     patches = extract_patches(image, patch_size=patch_size)

#     # Step 2: Compute correlation matrices
#     row_corr, col_corr = compute_correlation_matrices(patches)

#     # Step 3: Compute eigenvectors (basis)
#     VR = compute_eigenvectors(row_corr)
#     VC = compute_eigenvectors(col_corr)

#     # Step 4: Compute Kronecker product
#     V = compute_kronecker_basis(VR, VC)
#     # Visualize DCT-like bases
#     visualize_basis(V, patch_size=patch_size)
#     # Step 5: Compression, restoration, and evaluation
#     for num_components in num_components_list:
#         # Compress patches
#         compressed_patches, basis_subset = compress_patches(patches, V, num_components)

#         # Restore patches
#         restored_patches = restore_patches(compressed_patches, basis_subset, patch_size)

#         # Reconstruct image
#         reconstructed_image = reconstruct_image(restored_patches, image.shape, patch_size)

#         # Compute reconstruction error
#         error = compute_reconstruction_error(image, reconstructed_image)

#         # Display results
#         print(f"Number of components: {num_components}, Reconstruction Error: {error:.4f}")
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(image, cmap='gray')
#         plt.title("Original Image")
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(reconstructed_image, cmap='gray')
#         plt.title(f"Reconstructed with {num_components} components")
#         plt.axis('off')

#         plt.tight_layout()
#         plt.show()
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from PIL import Image

# Step 1: Load grayscale image
def load_grayscale_image(filepath):
    """
    Loads an image and converts it to grayscale.
    """
    image = Image.open(filepath).convert('L')  # Convert to grayscale ('L' mode)
    return np.array(image)  # Normalize to range [0, 1]

# Step 2: Extract patches from image
def extract_patches(image, patch_size=8):
    patches = view_as_windows(image, (patch_size, patch_size), step=patch_size)
    patches = patches.reshape(-1, patch_size, patch_size)
    return patches

# Step 3: Compute row-row and column-column correlation matrices
def compute_correlation_matrices(patches):
    num_patches, patch_size, _ = patches.shape
    
    # Compute row-row and column-column correlation matrices
    row_corr = np.zeros((patch_size, patch_size))
    col_corr = np.zeros((patch_size, patch_size))
    
    for i in range(num_patches):
        row_corr += np.outer(patches[i, :, :].mean(axis=1), patches[i, :, :].mean(axis=1))
        col_corr += np.outer(patches[i, :, :].mean(axis=0), patches[i, :, :].mean(axis=0))
    
    row_corr /= num_patches
    col_corr /= num_patches
    
    return row_corr, col_corr

# Step 4: Compress the patches by projecting them onto the reduced eigenbasis
def compress_patches(patches, VR, VC, num_components=64):
    """
    Compress the patches by projecting onto the top 'num_components' eigenvectors.
    """
    # Reshape patches into a matrix of shape (num_patches, patch_size*patch_size)
    patches_flat = patches.reshape(patches.shape[0], -1)
    
    # Calculate Kronecker product of row and column eigenvectors
    V = np.kron(VR, VC)
    
    # Select the first 'num_components' eigenvectors
    V_reduced = V[:, :num_components]
    
    # Project patches onto the reduced eigenbasis
    compressed_patches = np.dot(patches_flat, V_reduced)
    
    return compressed_patches, V_reduced

# Step 5: Restore patches from the compressed data
def restore_patches(compressed_patches, V_reduced, patch_size=8):
    """
    Restores patches from the compressed projection.
    """
    # Reconstruct the patches from the compressed coefficients
    restored_patches_flat = np.dot(compressed_patches, V_reduced.T)
    
    # Reshape the patches back into the original patch shape
    restored_patches = restored_patches_flat.reshape(-1, patch_size, patch_size)
    
    return restored_patches

# Step 6: Reconstruct the full image from patches
def reconstruct_image(patches, image_shape, patch_size):
    """
    Reconstruct the full image from the patches.
    """
    h, w = image_shape
    reconstructed = np.zeros((h, w))
    patch_idx = 0
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if patch_idx < patches.shape[0]:
                size_x = patch_size if i + patch_size <= h else h - i
                size_y = patch_size if j + patch_size <= w else w - j
                reconstructed[i:i + size_x, j:j + size_y] = patches[patch_idx][:size_x, :size_y]
                patch_idx += 1
    
    return reconstructed

# Step 7: Compute reconstruction error
def compute_reconstruction_error(original, restored):
    return np.sqrt(np.sum((original - restored) ** 2) / original.size)

# Main script
if __name__ == "__main__":
    # Load image and extract patches
    image_path = 'jpgb.png'  # Replace with your image path
    image = pad_image(load_grayscale_image(image_path))
    patch_size = 8

    # Extract patches from image
    patches = extract_patches(image, patch_size)

    # Compute row and column correlation matrices
    row_corr, col_corr = compute_correlation_matrices(patches)

    # Compute eigenvectors of row and column correlation matrices
    VR = np.linalg.eigh(row_corr)[1]
    VC = np.linalg.eigh(col_corr)[1]

    # Number of components to keep (can be tuned)
    num_components_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]  # Test different numbers of components
    for num_components in num_components_list:

        # Compress patches using the reduced eigenvectors
        compressed_patches, V_reduced = compress_patches(patches, VR, VC, num_components)

        # Restore patches from the compressed data
        restored_patches = restore_patches(compressed_patches, V_reduced)

        # Reconstruct the full image
        reconstructed_image = reconstruct_image(restored_patches, image.shape, patch_size)

        # Compute reconstruction error
        error = compute_reconstruction_error(image, reconstructed_image)

        # Display results
        print(f"Number of components: {num_components}, Reconstruction Error: {error:.4f}")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f"Reconstructed with {num_components} components")
        plt.axis('off')

        plt.tight_layout()
        plt.show()