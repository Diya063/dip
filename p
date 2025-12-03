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
