import os
from PIL import Image
def process_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for root, dirs, files in os.walk(input_folder):
        # Maintain folder hierarchy in output
        relative_path = os.path.relpath(root, input_folder)
        target_path = os.path.join(output_folder, relative_path)
        # os.makedirs(target_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(target_path, file)
                
                try:
                    # Open the image and convert to grayscale
                    with Image.open(input_file_path) as img:
                        gray_img = img.convert('L')
                        # Save the grayscale image to the target folder
                        gray_img.save(output_file_path)
                        print(f"Processed: {input_file_path} -> {output_file_path}")
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")

# Example usage
input_folder = "../images/msrcorid/single/"
output_folder = "../images/msrcorid/grayscale"

process_images(input_folder, output_folder)
