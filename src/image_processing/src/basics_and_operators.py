import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import convolve2d

import pdb

img = cv2.imread('/workspaces/base_ros2/src/image_processing/images/denali.jpg')

# print(img)
# ### Analyze dimensions
# print(img.shape) # prints image dimensions
# print(img.dtype) # Print the data type of the values

# blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # Split channels

# ### Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
# print(gray.shape) # Examin the dimensions

# cv2.imwrite('grayscale_img_sliced.png', gray[0:200, 0:200])

# ### Eaxmin the pixel statistics
# print(f"max pixel value: {gray.max()}")
# print(f"min pixel value: {gray.min()}")
# print(f"mean pixel value: {gray.mean()}")

### Examin the pixel-value histogram
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
# threshold = 100

# # # Apply binary thresholding
# gray[gray > threshold] = 255
# gray[gray <= threshold] = 0

# # # Save result
# cv2.imwrite('monadic_threshold_binary.jpg', gray)

# ## Skewing
# # Apply gamma correction to skew histogram towards 255
# gamma = 0.1  # Gamma < 1 brightens image
# img_normalized = gray / 255.0 # Normalize the image
# skewed = np.power(img_normalized, gamma) * 255.0
# skewed = skewed.astype(np.uint8)

# # # Save the skewed image
# cv2.imwrite('skewed_gray.jpg', skewed)

# hist = cv2.calcHist([skewed], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.title('Skewed Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.savefig('histogram_skewed_gray.png')
# plt.close()


# ### Dyadic operations 

# ## Chroma Keying
meme_dog = cv2.imread('/workspaces/base_ros2/src/image_processing/images/meme_dog.jpg')

# # # Compute green chromaticity
# sum_rgb = np.sum(meme_dog, axis=2, keepdims=True)  # Calculate B+G+R pixel values
# sum_rgb = np.where(sum_rgb == 0, 1, sum_rgb)  # Avoid division by zero
# g_chroma = (meme_dog[:, :, 1] / sum_rgb[:, :, 0]) # g = G/(R+G+B)

# # Plot the chromacity histogram
# plt.hist(g_chroma.flatten(), bins=100, range=(0, 1))
# plt.title('Green Chromaticity Histogram')
# plt.xlabel('Green Chromaticity (g)')
# plt.ylabel('Frequency')
# plt.savefig('meme_dog_green_chroma_hist.png')
# plt.close()


# # Based on the histogram, select a cut to create the mask
# mask = (g_chroma < 0.6)

# copy_of_img = img.copy() # Otherwise, we'd be changing the img itself.

# # # Apply the same mask on both images pixel by pixel
# copy_of_img[mask == True] = meme_dog[mask == True]

# # # Save the results as an image
# cv2.imwrite('meme_dog_in_denali.jpg', copy_of_img)


### Spacial  operation

## Edge detecting - basic
gray_dog = cv2.cvtColor(meme_dog, cv2.COLOR_BGR2GRAY)

# kernel = np.array([[-0.5, 0, 0.5]])
# dog_edges = cv2.filter2D(gray_dog, -1, kernel)
# cv2.imwrite('edges.jpg', dog_edges)


# ## Sobel kernels
# sobel_x = np.array([[0.125, 0, -0.125], [-0.25, 0, 0.25], [-0.125, 0, 0.125]])
# sobel_y = np.array([[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]])

# # Apply Sobel kernels
# edges_x = cv2.filter2D(gray_dog, -1, sobel_x)
# edges_y = cv2.filter2D(gray_dog, -1, sobel_y)

# # Combine edges (\sqrt(x^2 +))
# edges = np.sqrt(np.square(edges_x) + np.square(edges_y)).astype(np.uint8)

# # Normalize to 0-255 range for better visibility
# edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

# cv2.imwrite('sobel_edges.jpg', edges)


# Gaussian kernels with 2nd derivative laplanced.
# kernel = np.array([[2,4,5,4,2],
#                    [4,9,12,9,4],
#                    [5,12,15,12,5],
#                    [4,9,12,9,4],
#                    [2,4,5,4,2]]) / 159
# smoothed = cv2.filter2D(gray_dog, -1, kernel)

# laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]])
# edges = cv2.filter2D(smoothed, -1, laplacian)
# edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

# cv2.imwrite('laplance_edges.jpg', np.abs(edges))

# ## Gaussian with Sobel
# # Define Sobel kernels
# sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# # Define 5x5 Gaussian kernel
# gaussian = cv2.getGaussianKernel(5, sigma=1.0)
# gaussian = gaussian @ gaussian.T  # Create 5x5 kernel

# # Convolve Sobel and Gaussian kernels
# kernel_x = convolve2d(sobel_x, gaussian, mode='same')
# kernel_y = convolve2d(sobel_y, gaussian, mode='same')

# # Apply convolved kernels
# edges_x = cv2.filter2D(gray_dog, cv2.CV_32F, kernel_x)
# edges_y = cv2.filter2D(gray_dog, cv2.CV_32F, kernel_y)

# # Compute edge magnitude
# edges = np.sqrt(np.square(edges_x) + np.square(edges_y))

# # Normalize to 0-255
# edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # Save result
# cv2.imwrite('sobel_gaussian_edges.jpg', edges)


# ### Morphological Operators
# ## Boundary detection
# # Binarize image (threshold to create a binary mask)
# _, binary = cv2.threshold(gray_dog, 127, 255, cv2.THRESH_BINARY)
# cv2.imwrite('thresholded_img.jpg', binary)

# # Define circular structuring element (approximated as a disk)
# radius = 5
# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

# # Erode the binary image
# eroded = cv2.erode(binary, se)

# # Compute boundary (original - eroded)
# boundary = binary - eroded

# # Save result
# cv2.imwrite('boundaries_morphological.jpg', boundary)


