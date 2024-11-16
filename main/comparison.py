import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *

def comparison(bpp_results, rmse_results, bpp_results2, rmse_results2, algo1, algo2):
    plt.figure(figsize=(12, 8))

    plt.figure(figsize=(12, 8))
    len_image_paths = 1
    for i in range(len_image_paths):
        # Plot Basic algorithm results
        plt.plot(bpp_results[i], rmse_results[i], label=f'{algo1}')
        plt.scatter(bpp_results[i], rmse_results[i], color='blue', s=40, edgecolor='black')  # Add circles
        for x, y in zip(bpp_results[i], rmse_results[i]):
            plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, ha='left', va='bottom')

        # Plot Runlength algorithm results
        plt.plot(bpp_results2[i], rmse_results2[i], label=f'{algo2}')
        plt.scatter(bpp_results2[i], rmse_results2[i], color='orange', s=40, edgecolor='black')  # Add circles
        for x, y in zip(bpp_results2[i], rmse_results2[i]):
            plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, ha='left', va='bottom')

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Images")
    plt.tight_layout()
    plt.xlabel('BPP')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs. BPP comparison between {algo1} and {algo2} algorithms')
    plt.savefig(f'../results/{algo1}_{algo2}_comparison.png')
    plt.close()
