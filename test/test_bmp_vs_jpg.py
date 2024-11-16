import sys
sys.path.append('..')
from include.common_imports import *
from algorithms.helper import *

image_path = '../images/msrcorid/miscellaneous/101_0125.jpg'

# import a .jpg
with Image.open(image_path) as img:
    jpg_img = np.array(img)
    jpg_img = img.convert('L')
    # convert to numpy array
    jpg_img = np.array(jpg_img)

# convert to .bmp
bmp_img = convert_to_grayscale_bmp(image_path)

# print first 100 pixels of each image
print('JPG Image')
print(jpg_img[:10, :10])
print()
print('BMP Image')
print(bmp_img[:10, :10])

# check if same
assert np.array_equal(jpg_img, bmp_img)

# show the two images side by side
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(jpg_img, cmap='gray')
plt.title('JPG Image')

plt.subplot(122)
plt.imshow(bmp_img, cmap='gray')
plt.title('BMP Image')

plt.show()