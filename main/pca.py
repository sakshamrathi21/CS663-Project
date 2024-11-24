from sklearn.decomposition import PCA
from include.common_imports import *
from algorithms.helper import *

# def load_grayscale_images_from_folder(folder_path):
#     """
#     Loads all grayscale images from a folder, normalizes them, and returns as a list.
#     """
#     images = []
#     for filename in os.listdir(folder_path):
#         filepath = os.path.join(folder_path, filename)
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image file extensions
#             image = Image.open(filepath).convert('L')  # Convert to grayscale
#             images.append(np.array(image) / 255.0)  # Normalize to range [0, 1]
#     return images



# def compute_correlation_matrices(patches):
#     """
#     Computes row-row and column-column correlation matrices.
#     """
#     num_patches, patch_size, _ = patches.shape
#     row_corr = np.zeros((patch_size, patch_size))
#     col_corr = np.zeros((patch_size, patch_size))
#     for patch in patches:
#         row_corr += patch.T @ patch
#         col_corr += patch @ patch.T
#     row_corr /= (num_patches - 1)
#     col_corr /= (num_patches - 1)
#     return row_corr, col_corr

# def compress_patches(patches, V, num_components=64):
#     """
#     Compress the patches by projecting onto the top 'num_components' eigenvectors.
#     """
#     patches_flat = patches.reshape(patches.shape[0], -1)
#     # V = np.kron(VR, VC)
#     V_reduced = V[:, :num_components]
#     compressed_patches = np.dot(patches_flat, V_reduced)
#     return compressed_patches, V_reduced

# def restore_patches(compressed_patches, V_reduced, patch_size=8):
#     """
#     Restores patches from the compressed projection.
#     """
#     restored_patches_flat = np.dot(compressed_patches, V_reduced.T)
#     restored_patches = restored_patches_flat.reshape(-1, patch_size, patch_size)
#     return restored_patches



# def compute_reconstruction_error(original, restored):
#     """
#     Computes the reconstruction error (RMSE).
#     """
#     return np.sqrt(np.sum((original - restored) ** 2) / original.size)

# # Main script
# if __name__ == "__main__":
#     folder_path = "images/"  # Replace with your folder path
#     patch_size = 8

#     # Load images
#     images = load_grayscale_images_from_folder(folder_path)
#     patches = []
#     for img_idx, image in enumerate(images):
#         # print(f"Processing Image {img_idx + 1}/{len(images)}")
#         padded_image = pad_image(image, patch_size)

#         # Extract patches
#         patches = extract_patches(padded_image, patch_size)
#         flattened_patches = patches.reshape(patches.shape[0],-1)
#         # num_components_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]  # Test different numbers of components
#         num_components_list = [24]      
#         for num_components in num_components_list:
#             pca = PCA(num_components)
#             pca.fit(flattened_patches)
#             print(f"Processing Image {img_idx + 1}/{len(images)}")
#             compressed_patches = pca.transform(flattened_patches)
#             decompressed_patches = (pca.inverse_transform(compressed_patches)).reshape(patches.shape)
#             # Reconstruct the full image
#             reconstructed_image = reconstruct_image(decompressed_patches, image.shape, patch_size)

#             # Compute reconstruction error
#             error = compute_reconstruction_error(image, reconstructed_image)

#             # Display results
#             print(f"Number of components: {num_components}, Reconstruction Error: {error:.4f}")
#             plt.figure(figsize=(10, 5))
#             plt.subplot(1, 2, 1)
#             plt.imshow(image, cmap='gray')
#             plt.title("Original Image")
#             plt.axis('off')

#             plt.subplot(1, 2, 2)
#             plt.imshow(reconstructed_image, cmap='gray')
#             plt.title(f"Reconstructed with {num_components} components")
#             plt.axis('off')

#             plt.tight_layout()
#             plt.show()    

def pca(num_paths = 20):
    image_paths = get_image_paths()[0:num_paths]
    num_components_list = Config.num_components_list
    bpp_results = []
    rmse_results = []
    for image_path in image_paths:
        # grayscale_image_copy = get_gray_scale_image(image_path)
        grayscale_image_copy = convert_to_grayscale_bmp(image_path)
        print(image_path)
        bpp_per_image = []
        rmse_per_image = []
        for k in num_components_list:
            P = PCA(int(k))
            grayscale_image = grayscale_image_copy.copy()
            patches = extract_patches(grayscale_image)
            flattened_patches = patches.reshape(patches.shape[0],-1)
            P.fit(flattened_patches)
            compressedImage = P.transform(flattened_patches)
            save_compressed_patches("compressed_image.bin", compressedImage, grayscale_image.shape, patch_size=Config.patch_size, pca_model = P)
            reconstructed_image = load_compressed_patches("compressed_image.bin", k)
            rmse = calculate_rmse(grayscale_image, reconstructed_image)
            bpp = calculate_bpp_pca(flattened_patches.shape, compressedImage.shape,patches.shape[1],k)
            print(f"Components {k}, RMSE: {rmse}, BPP: {bpp}")
            bpp_per_image.append(bpp)
            rmse_per_image.append(rmse)
        bpp_results.append(bpp_per_image)
        rmse_results.append(rmse_per_image)
    return bpp_results, rmse_results

def dpca():
    folder_path = Config.folder_path  # Replace with your folder path
    grayscale_images = get_grayscale_images_from_folder(folder_path)
    # print(np.array(grayscale_images).shape)
    patches = extract_patches_from_dataset(grayscale_images)

    flattened_patches = patches.reshape(patches.shape[0],-1)
    num_components_list = Config.num_components_list
    folder = "compressed_images"
    bpp_results = np.zeros((len(grayscale_images),len(num_components_list)))
    rmse_results = np.zeros_like(bpp_results)
    i = 0
    for k in num_components_list:
        P = PCA(int(k))
        P.fit(flattened_patches)
        save_compressed_pca_model("compressed_model.bin",P)
        for img_idx,grayscale_image in enumerate(grayscale_images):
            padded_image = pad_image(grayscale_image)
            patches = extract_patches(padded_image)
            flattened = patches.reshape(patches.shape[0],-1)
            compressedImage = P.transform(flattened)
            output_file = os.path.join(folder, f"compressed_image{img_idx}.bin")
            save_compressed_patches_without_model(output_file, compressedImage, grayscale_image.shape, patch_size=Config.patch_size)
        P = load_compressed_pca_model("compressed_model.bin")
        for img_idx,grayscale_image in enumerate(grayscale_images):
            input_file = os.path.join(folder, f"compressed_image{img_idx}.bin")
            reconstructed_image = load_compressed_patches_without_model(input_file,P, Config.num_components)        
            rmse = calculate_rmse(grayscale_image, reconstructed_image)
            bpp = calculate_bpp_dpca(flattened_patches.shape, compressedImage.shape,patches.shape[1],k)
            print(f"Components {k}, RMSE: {rmse}, BPP: {bpp}")
            bpp_results[img_idx][i]= bpp
            rmse_results[img_idx][i] = rmse
        i = i + 1
    return bpp_results, rmse_results

def cpca(num_paths = 20):
    image_paths = get_image_paths()[0:num_paths]
    num_components_list = Config.num_components_list
    bpp_results = []
    rmse_results = []
    for image_path in image_paths:
        colour_image_copy = extract_colour_image(image_path)/255
        blue,green,red = cv2.split(colour_image_copy)
        bpp_per_image = []
        rmse_per_image = []
        for k in num_components_list:
            P_blue = PCA(int(k))
            P_green = PCA(int(k))
            P_red = PCA(int(k))
            blue_patches = extract_patches(blue)
            green_patches = extract_patches(green)
            red_patches = extract_patches(red)
            flattened_blue_patches = blue_patches.reshape(blue_patches.shape[0],-1)
            flattened_green_patches = green_patches.reshape(green_patches.shape[0],-1)
            flattened_red_patches = red_patches.reshape(red_patches.shape[0],-1)
            P_blue.fit(flattened_blue_patches)
            P_green.fit(flattened_green_patches)
            P_red.fit(flattened_red_patches)
            compressedBlueImage = P_blue.transform(flattened_blue_patches)
            compressedGreenImage = P_green.transform(flattened_green_patches)
            compressedRedImage = P_red.transform(flattened_red_patches)
            save_compressed_patches("compressed_blue_image.bin", compressedBlueImage, blue.shape, patch_size=Config.patch_size, pca_model = P_blue)
            save_compressed_patches("compressed_green_image.bin", compressedGreenImage, green.shape, patch_size=Config.patch_size, pca_model = P_green)
            save_compressed_patches("compressed_red_image.bin", compressedRedImage, red.shape, patch_size=Config.patch_size, pca_model = P_red)
            reconstructed_image = load_compressed_coloured_patches("compressed_blue_image.bin","compressed_green_image.bin","compressed_red_image.bin",k)
            bpp = 3*calculate_bpp_pca(flattened_blue_patches.shape, compressedBlueImage.shape,blue_patches.shape[1],k)
            rmse = calculate_rmse(reconstructed_image,colour_image_copy)
            print(f"Components {k}, RMSE: {rmse}, BPP: {bpp}")
            bpp_per_image.append(bpp)
            rmse_per_image.append(rmse)
        bpp_results.append(bpp_per_image)
        rmse_results.append(rmse_per_image)
    return bpp_results, rmse_results