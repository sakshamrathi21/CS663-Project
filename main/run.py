import sys
import subprocess
import sys
sys.path.append('..')
import runlength, basic, actual_jpeg
from algorithms.helper import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <file_names>")
    if len(sys.argv) == 2:
        file_number = sys.argv[1]
        if file_number == "basic":
            bpp_results, rmse_results = basic.basic()
            rmse_vs_bpp_plot(bpp_results, rmse_results, get_image_paths(), plot_path='../results/basic.png')
        elif file_number == "runlength":
            bpp_results, rmse_results = runlength.runlength()
            rmse_vs_bpp_plot(bpp_results, rmse_results, get_image_paths(), plot_path='../results/runlength.png')
        elif file_number == "jpeg":
            bpp_results, rmse_results = actual_jpeg.jpeg()
            rmse_vs_bpp_plot(bpp_results, rmse_results, get_image_paths(), plot_path='../results/jpeg.png')
        else:
            print("Invalid file name")
