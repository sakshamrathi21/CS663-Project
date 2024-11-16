from PIL import Image

# Open an image
image = Image.open("../images/jpgb.png")

# Check the mode of the image
if image.mode == 'RGBA':
    # Convert RGBA to RGB
    image = image.convert('RGB')

# Convert the image to grayscale
image = image.convert('L')  # 'L' mode is for grayscale

# Save the image in JPEG format with adjustable quality
image.save("../results/compressed_image.jpeg", "JPEG", quality=50)  # Quality ranges from 1 (worst) to 95 (best)
