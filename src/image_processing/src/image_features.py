import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

img = cv2.imread('/workspaces/base_ros2/src/image_processing/images/tomatoes.jpg')
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
# gray = img[:,:,2] 
# ### Examin the pixel-value histogram
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.savefig('histogram.png')
# plt.close()

# # # Set threshold
# threshold = 170

# # # # Apply binary thresholding
# gray[gray > threshold] = 255
# gray[gray <= threshold] = 0

# # # Save result
# cv2.imwrite('sign_threshold_binary.jpg', gray)

## Color image classification

# tmt = cv2.imread('/workspaces/base_ros2/src/image_processing/images/tomatoes.jpg')

# # # Convert the BGR image to LAB
# lab_tmt = cv2.cvtColor(tmt, cv2.COLOR_BGR2LAB)

# # # Extract a* and b* planes (2-channel image)
# ab_tmt = lab_tmt[:, :, 1:3]  # Shape: (height, width, 2)_tmt
# b_tmt = lab_tmt[:, :, 1:2]  # Shape: (height, width, 2)_tmt

# # # Create a dummy L* channel with neutral value 50, matching the shape of the first channel of ab_tmt
# l_channel = np.full_like(ab_tmt[:, :, 0:1], 50)

# # # Concatenate the dummy L* channel with ab_tmt along the third axis to form a 3-channel L*a*b* image
# lab_dummy = np.concatenate((l_channel, ab_tmt), axis=2)

# # # Convert the L*a*b* image back to BGR and save it as 'ab_tmt.jpg'
# cv2.imwrite('ab_tmt.jpg', cv2.cvtColor(lab_dummy, cv2.COLOR_Lab2BGR))

# # # Commented out: Calculate histogram of b_tmt (grayscale image)
# hist = cv2.calcHist([b_tmt], [0], None, [256], [0, 256])

# # Commented out: Plot histogram
# plt.plot(hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.savefig('b_tmt_hist.png')
# plt.close()

# # # Get height and width from ab_tmt shape (excluding channel dimension)
# height, width = ab_tmt.shape[:2]

# # Reshape ab_tmt into a 2D array of a*b* points for k-means clustering
# ab_flat = ab_tmt.reshape((height * width, 2)).astype(np.float32)

# # Define k-means termination criteria: epsilon or max 10 iterations with precision 1.0
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# # Perform k-means clustering with 3 clusters, random centers, 10 attempts; returns labels and centers
# _, labels, centers = cv2.kmeans(ab_flat, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Reshape the cluster labels back to the original image dimensions
# clustered = labels.reshape((height, width))

# # Save the clustered image, scaled to 0-255 range by multiplying by 85 (255//3)
# cv2.imwrite('clustered_ab.png', clustered * (255 // 3))

# # Create a mask initialized with 255 (white) for all pixels
# mask = np.ones_like(clustered, dtype=np.uint8) * 255

# # Set pixels where cluster label is 1 to 0 (black) in the mask
# mask[(clustered == 1)] = 0

# # Save the mask image as 'clsutered_mask.png' (note: typo 'clsutered' should be 'clustered')
# cv2.imwrite('clustered_mask.png', mask)

# # Define a structuring element for erosion with an ellipse of radius 7
# radius = 7
# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

# # Erode the binary mask to smooth edges and remove small noise
# mask_eroded = cv2.erode(mask, se)

# # Save the eroded mask as 'clsutered_mask_smoothed.jpg' (note: typo 'clsutered' should be 'clustered')
# cv2.imwrite('clustered_mask_smoothed.jpg', mask_eroded)

# # Apply the inverted mask to an assumed 'tmt' image (assuming tmt is defined RGB image)
# # Note: 'tmt' should be defined earlier as cv2.imread('tmt.jpg') for this to work
# masked_tmt = cv2.bitwise_and(tmt, tmt, mask=cv2.bitwise_not(mask_eroded))

# # Save the masked tmt image
# cv2.imwrite('tmt_masked.jpg', masked_tmt)


# ## Object Instance Representation

# # tmt_binary = cv2.imread('/workspaces/base_ros2/tmt_masked.jpg', cv2.IMREAD_GRAYSCALE)
# # tmt_binary = cv2.imread('/workspaces/base_ros2/src/image_processing/images/tomatoes.jpg', cv2.IMREAD_GRAYSCALE)

# tmt_binary = mask_eroded.copy()
# _, tmt_binary = cv2.threshold(tmt_binary, 127,255, cv2.THRESH_BINARY)
# # tmt = cv2.imread('/workspaces/base_ros2/tmt_masked.jpg')

# cv2.imwrite('tmt_binay.jpg', tmt_binary)

# # Detect blobs using SimpleBlobDetector
# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 1000
# params.maxArea = 200000
# params.minCircularity = 0.3
# params.filterByCircularity = False
# params.filterByConvexity = False  # Disabled for broader detection
# params.filterByInertia = False  # Disabled for flexibility

# detector = cv2.SimpleBlobDetector_create(params)

# # # Detect blobs
# keypoints = detector.detect(tmt_binary)
# # Define colors for different blobs (cycle through red, green, blue)
# colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR format

# # print(keypoints)

# # Draw each blob with a different color
# for i, kp in enumerate(keypoints):
#     color = colors[i % len(colors)]
#     center = (int(kp.pt[0]), int(kp.pt[1]))
#     radius = int(kp.size / 2)
#     cv2.circle(tmt, center, radius, color, 10)  # Draw circle outline
# # Save or display result
# cv2.imwrite('blobs_detected.jpg', tmt)

