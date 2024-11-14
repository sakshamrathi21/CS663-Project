import sys
import subprocess

def run_file(filename):
    try:
        subprocess.run(["python3", filename], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 run.py <file_number>")
        sys.exit(1)

    file_number = sys.argv[1]

    if file_number == "basic":
        run_file("basic.py")
    elif file_number == "runlength":
        run_file("runlength.py")
    else:
        print("Invalid argument. Use 1 or 2.")
        sys.exit(1)
