import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *

config = Config()
image_paths = get_image_paths()[0:1]
# image_paths = ['../images/jpgb.png']
quality_factors = Config.quality_factors
# quality_factors = [2, 10, 50, 80]
bpp_results = []
rmse_results = []

for image_path in image_paths:
    grayscale_image_copy = get_gray_scale_image(image_path)
    bpp_per_image = []
    rmse_per_image = []
    for quality in quality_factors:
        grayscale_image = grayscale_image_copy.copy()
        dct_image = dct2d(grayscale_image.copy())
        quality = int(quality)
        quantized_dct_image = quantization(dct_image, quality=quality)
        huffman_tree = HuffmanTree()
        flat_quantized_data = quantized_dct_image.flatten().tolist()
        frequency = Counter(flat_quantized_data)
        if len(frequency) == 1:
            continue
        huffman_tree.build_tree(frequency)
        encoded_data = huffman_tree.encode(flat_quantized_data)
        save_compressed_image("compressed_image.bin", encoded_data, grayscale_image.shape, patch_size=Config.patch_size, huffman_tree=huffman_tree)
        reconstructed_image = load_compressed_image("compressed_image.bin", quality=quality)
        rmse = calculate_rmse(grayscale_image, reconstructed_image)
        # I want to show the reconstructed and the grayscale image side by side
        # I will use the following code to do that
        # show_images_side_by_side(reconstructed_image, grayscale_image, title=f"../results/Quality: {quality}_comparison.png")
        # exit()

        bpp = calculate_bpp(encoded_data, grayscale_image.shape)
        print(f"Quality: {quality}, RMSE: {rmse}, BPP: {bpp}")
        bpp_per_image.append(bpp)
        rmse_per_image.append(rmse)
    bpp_results.append(bpp_per_image)
    rmse_results.append(rmse_per_image)

# rmse_vs_bpp_plot(bpp_results, rmse_results, image_paths, plot_path='../results/basic.png')

bpp_results2 = []
rmse_results2 = []

for image_path in image_paths:
    grayscale_image_copy = get_gray_scale_image(image_path)
    bpp_per_image = []
    rmse_per_image = []
    for quality in quality_factors:
        grayscale_image = grayscale_image_copy.copy()
        dct_image = dct2d(grayscale_image.copy())
        quality = int(quality)
        quantized_dct_image = quantization(dct_image, quality=quality)
        huffman_tree = HuffmanTree()
        encoded_data = runlength_encode(quantized_dct_image)
        frequency = Counter(encoded_data)
        if len(frequency) == 1:
            continue
        huffman_tree.build_tree(frequency)
        
        encoded_data = huffman_tree.encode(encoded_data)
        save_compressed_image("compressed_image.bin", encoded_data, grayscale_image.shape, patch_size=Config.patch_size, huffman_tree=huffman_tree)
        reconstructed_image = load_compressed_image_runlength("compressed_image.bin", quality=quality)
        rmse = calculate_rmse(grayscale_image, reconstructed_image)
        bpp = calculate_bpp(encoded_data, grayscale_image.shape)
        print(f"Quality: {quality}, RMSE: {rmse}, BPP: {bpp}")
        bpp_per_image.append(bpp)
        rmse_per_image.append(rmse)
    bpp_results2.append(bpp_per_image)
    rmse_results2.append(rmse_per_image)


# We can also compare the results of both algorithms on the same plot using matplotlib:

plt.figure(figsize=(12, 8))

plt.figure(figsize=(12, 8))

for i in range(len(image_paths)):
    # Plot Basic algorithm results
    plt.plot(bpp_results[i], rmse_results[i], label=f'Basic')
    plt.scatter(bpp_results[i], rmse_results[i], color='blue', s=40, edgecolor='black')  # Add circles
    for x, y in zip(bpp_results[i], rmse_results[i]):
        plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, ha='left', va='bottom')

    # Plot Runlength algorithm results
    plt.plot(bpp_results2[i], rmse_results2[i], label=f'Runlength')
    plt.scatter(bpp_results2[i], rmse_results2[i], color='orange', s=40, edgecolor='black')  # Add circles
    for x, y in zip(bpp_results2[i], rmse_results2[i]):
        plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, ha='left', va='bottom')

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Images")
plt.tight_layout()
plt.xlabel('BPP')
plt.ylabel('RMSE')
plt.title('RMSE vs. BPP comparison between Basic and Runlength algorithms')
plt.savefig('../results/comparison.png')
plt.close()
