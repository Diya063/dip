import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

# Read sample image
img = data.astronaut()  
print("Original Image Shape:", img.shape)

# Display original
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.show()

#resizing the image
resized = resize(img, (200, 200))  # stays in RGB format
plt.imshow(resized)
plt.title("Resized Image (200x200)")
plt.axis("off")
plt.show()


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# Convert to black & white (thresholding)
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(bw, cmap="gray")
plt.title("Black & White Image")
plt.axis("off")
plt.show()

# Draw the image profile (intensity values of row 100)
row = gray[100, :]   # take row 100
plt.plot(row)
plt.title("Image Profile (Row 100 Intensity)")
plt.xlabel("Column Index")
plt.ylabel("Intensity")
plt.show()

# Separate into R, G, B planes
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(R, cmap="Reds"); plt.title("Red Plane"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(G, cmap="Greens"); plt.title("Green Plane"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(B, cmap="Blues"); plt.title("Blue Plane"); plt.axis("off")
plt.show()

# Merge R, G, B planes back
merged = cv2.merge([R, G, B])
plt.imshow(merged)
plt.title("Merged Image")
plt.axis("off")
plt.show()

# Write given 2D data to an image file
data_2d = np.array([[0, 128, 255],
                    [64, 192, 128],
                    [255, 0, 64]], dtype=np.uint8)
cv2.imwrite("output_data_image.png", data_2d)

# practical 2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Read sample image
img = data.astronaut()  
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV operations
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

# a
negative = 255 - gray

plt.imshow(negative, cmap="gray")
plt.title("Negative Image")
plt.axis("off")
plt.show()

# b
flip_h = cv2.flip(img, 1)   # Horizontal flip
flip_v = cv2.flip(img, 0)   # Vertical flip

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(flip_h, cv2.COLOR_BGR2RGB)); plt.title("Horizontal Flip"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(flip_v, cv2.COLOR_BGR2RGB)); plt.title("Vertical Flip"); plt.axis("off")
plt.show()

# c
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(bw, cmap="gray")
plt.title("Binary Thresholded Image")
plt.axis("off")
plt.show()

# d
I_min = np.min(gray)
I_max = np.max(gray)

contrast_stretched = ((gray - I_min) / (I_max - I_min)) * 255
contrast_stretched = contrast_stretched.astype(np.uint8)

plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretched Image")
plt.axis("off")
plt.show()


# Code (PRACTICAL 3)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Two sample images (same size)
img1 = data.astronaut()
img2 = data.camera()
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# a. Addition of two images
added = cv2.add(img1, img2)
plt.imshow(added)
plt.title("Added Image")
plt.axis("off")
plt.show()

# b. Subtract one image from another
subtracted = cv2.subtract(img1, img2)

plt.imshow(subtracted)
plt.title("Subtracted Image")
plt.axis("off")
plt.show()

# c. Calculate mean value of image
mean_val = np.mean(img1)
print("Mean Pixel Value:", mean_val)

# d. Change brightness by changing mean value
bright = cv2.add(img1, 50)   # Increase brightness
dark   = cv2.subtract(img1, 50)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(bright); plt.title("Brighter Image"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(dark);   plt.title("Darker Image"); plt.axis("off")
plt.show()

#practical 5
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ---------------------------------------------------
# LOAD TWO SAMPLE IMAGES (convert to same size)
# ---------------------------------------------------
img1 = data.camera()          # grayscale image 1
img2 = data.coins()           # grayscale image 2

# Resize img2 to match img1 dimensions
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Display original images
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img1, cmap='gray'); plt.title("Image 1"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(img2, cmap='gray'); plt.title("Image 2"); plt.axis("off")
plt.show()

# ---------------------------------------------------
# AND Operation
# ---------------------------------------------------
# Performs bitwise AND → keeps only common bright regions
and_img = cv2.bitwise_and(img1, img2)

plt.imshow(and_img, cmap='gray')
plt.title("AND Operation Output")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# OR Operation
# ---------------------------------------------------
# Combines all bright regions from both images
or_img = cv2.bitwise_or(img1, img2)

plt.imshow(or_img, cmap='gray')
plt.title("OR Operation Output")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# INTERSECTION (same as AND)
# ---------------------------------------------------
# Intersection = common areas in both images
intersection = cv2.bitwise_and(img1, img2)

plt.imshow(intersection, cmap='gray')
plt.title("Intersection of Images")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# WATERMARKING using XOR Operation
# ---------------------------------------------------
# XOR highlights the differences → used for watermarking
xor_img = cv2.bitwise_xor(img1, img2)

plt.imshow(xor_img, cmap='gray')
plt.title("Watermarking using XOR")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# NOT Operation → Negative Image
# ---------------------------------------------------
# Inverts pixel values: 255 - pixel
not_img = cv2.bitwise_not(img1)

plt.imshow(not_img, cmap='gray')
plt.title("NOT Operation (Negative Image)")
plt.axis("off")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ---------------------------------------------------
# LOAD IMAGE (Grayscale)
# ---------------------------------------------------
img = data.camera()     # sample grayscale image

plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# HISTOGRAM CALCULATION USING INBUILT FUNCTION
# ---------------------------------------------------
# Using cv2.calcHist to calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title("Histogram (Inbuilt Function)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------
# HISTOGRAM EQUALIZATION USING INBUILT FUNCTION
# ---------------------------------------------------
equalized_img = cv2.equalizeHist(img)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(equalized_img, cmap="gray"); plt.title("Equalized"); plt.axis("off")
plt.show()

# ---------------------------------------------------
# MANUAL HISTOGRAM CALCULATION (WITHOUT INBUILT FUNCTION)
# ---------------------------------------------------
# Count frequency of each intensity value (0–255)
manual_hist = np.zeros(256)

for pixel in img.flatten():
    manual_hist[pixel] += 1

plt.plot(manual_hist)
plt.title("Manual Histogram Calculation")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------
# MANUAL HISTOGRAM EQUALIZATION
# ---------------------------------------------------
# Step 1: Normalize histogram → PDF
pdf = manual_hist / manual_hist.sum()

# Step 2: Cumulative distribution → CDF
cdf = np.cumsum(pdf)

# Step 3: Multiply by max intensity (255)
equal_map = np.uint8(255 * cdf)

# Step 4: Apply lookup table to each pixel
manual_eq_img = equal_map[img]

# Display result
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Original Image"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(manual_eq_img, cmap="gray"); plt.title("Manual Equalization"); plt.axis("off")
plt.show()

# PRACTICAL 6
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ---------------------------------------------------
# LOAD IMAGE (RGB → BGR → RGB for display correctness)
# ---------------------------------------------------
img = data.astronaut()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# TRANSLATION (Shift image)
# ---------------------------------------------------
# Move image right by 50 and down by 70 pixels
tx, ty = 50, 70
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, translation_matrix,
                            (img.shape[1], img.shape[0]))

plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
plt.title("Translated Image (Right 50, Down 70)")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# SCALING (Resize)
# ---------------------------------------------------
scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image (1.5x)")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# ROTATION
# ---------------------------------------------------
# Rotate around center by 45 degrees
(h, w) = img.shape[:2]
center = (w//2, h//2)

rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)  # 45° rotation
rotated = cv2.warpAffine(img, rotation_matrix, (w, h))

plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45°)")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# SHRINKING (Reduce Size)
# ---------------------------------------------------
shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

plt.imshow(cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB))
plt.title("Shrunk Image (0.5x)")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# ZOOMING (Cropping + Resizing)
# ---------------------------------------------------
# Crop the center region
h, w = img.shape[:2]
crop = img[h//4:3*h//4, w//4:3*w//4]

# Resize cropped part to original size = zoom effect
zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
plt.title("Zoomed Image (Center Region)")
plt.axis("off")
plt.show()

# PRACTICAL 7
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util, restoration

# ---------------------------------------------------
# LOAD IMAGE (Convert to grayscale)
# ---------------------------------------------------
img = data.camera()
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# ADD SALT & PEPPER NOISE
# ---------------------------------------------------
sp_noisy = util.random_noise(img, mode='s&p', amount=0.1)   # salt & pepper 10%
sp_noisy = (sp_noisy * 255).astype(np.uint8)

plt.imshow(sp_noisy, cmap="gray")
plt.title("Salt & Pepper Noisy Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# REMOVE SALT & PEPPER NOISE (Median Filter)
# ---------------------------------------------------
# Median filter removes S&P noise effectively
median_restored = cv2.medianBlur(sp_noisy, 3)

plt.imshow(median_restored, cmap="gray")
plt.title("Restored (Median Filter)")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# ADD GAUSSIAN NOISE
# ---------------------------------------------------
gauss_noisy = util.random_noise(img, mode='gaussian', var=0.01)
gauss_noisy = (gauss_noisy * 255).astype(np.uint8)

plt.imshow(gauss_noisy, cmap="gray")
plt.title("Gaussian Noisy Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# MINIMIZE GAUSSIAN NOISE (Gaussian Blur)
# ---------------------------------------------------
gaussian_restored = cv2.GaussianBlur(gauss_noisy, (5,5), 1)

plt.imshow(gaussian_restored, cmap="gray")
plt.title("Restored (Gaussian Blur)")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# MEDIAN FILTER (General Noise Removal)
# ---------------------------------------------------
median_filtered = cv2.medianBlur(img, 5)

plt.imshow(median_filtered, cmap="gray")
plt.title("Median Filter Output")
plt.axis("off")
plt.show()


# ---------------------------------------------------
# WIENER FILTER (De-noising with skimage)
# ---------------------------------------------------
# Wiener filter requires float image
img_float = img.astype(np.float32) / 255.0
wiener_restored = restoration.wiener(img_float, psf=np.ones((5,5))/25, balance=0.5)

plt.imshow(wiener_restored, cmap="gray")
plt.title("Wiener Filter Output")
plt.axis("off")
plt.show()


#PRACTICAL 8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.signal import convolve2d

# ---------------------------------------------------
# LOAD IMAGE (Grayscale)
# ---------------------------------------------------
img = data.camera()
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# 1-D Convolution Example (row vector)
# ---------------------------------------------------
row = img[100, :]  # take row 100
kernel_1d = np.array([1, 0, -1])  # simple edge detection kernel
conv_1d = np.convolve(row, kernel_1d, mode='same')

plt.plot(conv_1d)
plt.title("1-D Convolution of Row 100")
plt.xlabel("Column Index")
plt.ylabel("Intensity")
plt.show()

# ---------------------------------------------------
# 2-D Convolution Example (Low-pass filter / Smoothing)
# ---------------------------------------------------
low_pass_kernel = np.ones((3,3), dtype=np.float32) / 9  # 3x3 averaging mask
low_pass_img = convolve2d(img, low_pass_kernel, mode='same', boundary='symm')

plt.imshow(low_pass_img, cmap="gray")
plt.title("Low-pass Filtered Image (Smoothing)")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# 2-D Convolution Example (High-pass filter / Sharpening)
# ---------------------------------------------------
high_pass_kernel = np.array([[ -1, -1, -1],
                             [ -1,  8, -1],
                             [ -1, -1, -1]])

high_pass_img = convolve2d(img, high_pass_kernel, mode='same', boundary='symm')
high_pass_img = np.clip(high_pass_img, 0, 255)

plt.imshow(high_pass_img, cmap="gray")
plt.title("High-pass Filtered Image (Sharpening)")
plt.axis("off")
plt.show()


# PRACTICAL 9
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ---------------------------------------------------
# LOAD IMAGE (Grayscale)
# ---------------------------------------------------
img = data.camera()
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# APPLY FFT (Frequency Domain Transformation)
# ---------------------------------------------------
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  # shift zero frequency to center
magnitude_spectrum = 20*np.log(np.abs(fshift)+1)

plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum (FFT)")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# CREATE LOW-PASS FILTER (Ideal)
# ---------------------------------------------------
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask_lp = np.zeros((rows, cols), np.uint8)
r = 30  # radius
cv2.circle(mask_lp, (ccol, crow), r, 1, thickness=-1)

# Apply mask (low-pass)
fshift_lp = fshift * mask_lp
magnitude_lp = 20*np.log(np.abs(fshift_lp)+1)

plt.imshow(magnitude_lp, cmap="gray")
plt.title("Low-pass Filtered Spectrum")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# CREATE HIGH-PASS FILTER (Ideal)
# ---------------------------------------------------
mask_hp = 1 - mask_lp  # inverse of low-pass

fshift_hp = fshift * mask_hp
magnitude_hp = 20*np.log(np.abs(fshift_hp)+1)

plt.imshow(magnitude_hp, cmap="gray")
plt.title("High-pass Filtered Spectrum")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# APPLY IFFT TO RECONSTRUCT IMAGE
# ---------------------------------------------------
# Low-pass reconstruction
img_lp = np.fft.ifft2(np.fft.ifftshift(fshift_lp))
img_lp = np.abs(img_lp)

# High-pass reconstruction
img_hp = np.fft.ifft2(np.fft.ifftshift(fshift_hp))
img_hp = np.abs(img_hp)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(img_lp, cmap="gray"); plt.title("Low-pass Reconstructed"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(img_hp, cmap="gray"); plt.title("High-pass Reconstructed"); plt.axis("off")
plt.show()


# PRACTICAL 10
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import ndimage

# ---------------------------------------------------
# LOAD IMAGE (Grayscale)
# ---------------------------------------------------
img = data.camera()
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# SOBEL EDGE DETECTION
# ---------------------------------------------------
# Compute gradient along x and y
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Compute magnitude
sobel_mag = np.sqrt(sobelx**2 + sobely**2)
sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

plt.imshow(sobel_mag, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# PREWITT EDGE DETECTION (using convolution)
# ---------------------------------------------------
prewitt_kernelx = np.array([[ -1, 0, 1],
                            [ -1, 0, 1],
                            [ -1, 0, 1]])
prewitt_kernely = np.array([[ 1, 1, 1],
                            [ 0, 0, 0],
                            [-1,-1,-1]])

prewittx = ndimage.convolve(img.astype(float), prewitt_kernelx)
prewitty = ndimage.convolve(img.astype(float), prewitt_kernely)
prewitt_mag = np.sqrt(prewittx**2 + prewitty**2)
prewitt_mag = np.uint8(np.clip(prewitt_mag, 0, 255))

plt.imshow(prewitt_mag, cmap="gray")
plt.title("Prewitt Edge Detection")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# LAPLACIAN EDGE DETECTION
# ---------------------------------------------------
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Edge Detection")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# CANNY EDGE DETECTION
# ---------------------------------------------------
canny = cv2.Canny(img, 100, 200)

plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()


# PRACTICAL 11
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ---------------------------------------------------
# LOAD IMAGE (Binary/Grayscale)
# ---------------------------------------------------
img = data.camera()
# Convert to binary for morphological operations
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# DEFINE STRUCTURING ELEMENT (3x3)
# ---------------------------------------------------
kernel = np.ones((3,3), np.uint8)

# ---------------------------------------------------
# EROSION (Shrinks bright regions)
# ---------------------------------------------------
erosion = cv2.erode(binary, kernel, iterations=1)

plt.imshow(erosion, cmap="gray")
plt.title("Eroded Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# DILATION (Expands bright regions)
# ---------------------------------------------------
dilation = cv2.dilate(binary, kernel, iterations=1)

plt.imshow(dilation, cmap="gray")
plt.title("Dilated Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# OPENING (Erosion followed by Dilation)
# ---------------------------------------------------
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

plt.imshow(opening, cmap="gray")
plt.title("Opening (Erosion → Dilation)")
plt.axis("off")
plt.show()

# ---------------------------------------------------
# CLOSING (Dilation followed by Erosion)
# ---------------------------------------------------
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

plt.imshow(closing, cmap="gray")
plt.title("Closing (Dilation → Erosion)")
plt.axis("off")
plt.show()
