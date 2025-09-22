import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


img = cv2.imread('/workspaces/base_ros2/src/image_processing/images/denali.jpg')

# ### Analyze dimensions
# print(img.shape) # prints image dimensions
# print(img.dtype) # Print the data type of the values

# blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # Split channels
# print(blue.shape)  # dimensions of a single channel

# ### Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
print(gray.shape) # Examin the dimensions

# ### Eaxmin the pixel statistics
# print(f"max pixel value: {gray.max()}")
# print(f"min pixel value: {gray.min()}")
# print(f"mean pixel value: {gray.mean()}")

# ### Examin the pixel-value histogram
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.savefig('histogram.png')
# plt.close()

# ### Analyze the histogram
# peaks, _ = find_peaks(hist.flatten()) # find all peaks
# print(f"number of peaks: {peaks.size}")

# peaks, _ = find_peaks(hist.flatten(), threshold=100, distance=50) # find peaks with filters
# print(f"number of peaks after filters: {peaks.size}")
# print(f"position of peaks: {peaks}")

# ### Monadic operations 

# ## Thresholding
# # Set threshold
# threshold = 200

# # Apply binary thresholding
# gray[gray > threshold] = 255
# gray[gray <= threshold] = 0

# # Save result
# cv2.imwrite('monadic_threshold_binary.jpg', gray)

## Skewing
# Apply gamma correction to skew histogram towards 255
gamma = 0.5  # Gamma < 1 brightens image
img_normalized = gray / 255.0 # Normalize the image
skewed = np.power(img_normalized, gamma) * 255.0
skewed = skewed.astype(np.uint8)

# Save the skewed image
cv2.imwrite('skewed_gray.jpg', skewed)

hist = cv2.calcHist([skewed], [0], None, [256], [0, 256])
plt.plot(hist)
plt.title('Skewed Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('histogram_skewed_gray.png')
plt.close()



