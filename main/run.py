import sys
import subprocess
import runlength, basic, actual_jpeg, runlength, comparison
import sys
sys.path.append('..')
from algorithms.helper import *

if __name__ == "__main__":
    # print("check")
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
            # print("hello")
            bpp_results, rmse_results = actual_jpeg.jpeg()
            rmse_vs_bpp_plot(bpp_results, rmse_results, get_image_paths(), plot_path='../results/jpeg.png')
        else:
            print("Invalid file name")
    elif len(sys.argv) == 3:
        algo1 = sys.argv[1]
        algo2 = sys.argv[2]
        if algo1 == "basic" and algo2 == "runlength":
            bpp_results1, rmse_results1 = basic.basic(1)
            bpp_results2, rmse_results2 = runlength.runlength(1)
            comparison.comparison(bpp_results1, rmse_results1, bpp_results2, rmse_results2, algo1, algo2)
        elif algo1 == "basic" and algo2 == "jpeg":
            bpp_results1, rmse_results1 = basic.basic(1)
            bpp_results2, rmse_results2 = actual_jpeg.jpeg(1)
            comparison.comparison(bpp_results1, rmse_results1, bpp_results2, rmse_results2, algo1, algo2)
        elif algo1 == "runlength" and algo2 == "jpeg":
            bpp_results1, rmse_results1 = runlength.runlength(1)
            bpp_results2, rmse_results2 = actual_jpeg.jpeg(1)
            comparison.comparison(bpp_results1, rmse_results1, bpp_results2, rmse_results2, algo1, algo2)
        else:
            print("Invalid file names")
