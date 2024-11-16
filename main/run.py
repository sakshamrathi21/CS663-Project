import sys
import subprocess
import sys
sys.path.append('..')
import runlength, comparison, basic
from algorithms.helper import *



def run_file(filename):
    try:
        subprocess.run(["python3", filename], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {filename}: {e}")
        sys.exit(1)

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
            run_file("actual_jpeg.py")
        else:
            print("Invalid file name")
