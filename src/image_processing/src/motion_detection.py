import cv2
import os
import numpy as np

video_folder_path = '/workspaces/base_ros2/src/image_processing/split/'

background = cv2.imread('/workspaces/base_ros2/src/image_processing/split/output_0001.png')
bkg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

white = np.full_like(bkg_gray, 0)
# Create an image of size(background) with white pixels in it

for i in range(1, 87):
    frame_name = f"output_{i:04d}.png"
    frame_path = os.path.join(video_folder_path, frame_name)

    current_frame = cv2.imread(frame_path)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = current_frame - bkg_gray
    white = white + diff
    
    clipped_diff = np.clip(diff, -20, 20)

    bkg_gray = bkg_gray + clipped_diff

cv2.imwrite('Background_Estimate.jpg', bkg_gray)
cv2.imwrite('diff.jpg', white)

# save the image bkg_gray
# Save the white image

# ffmpeg to concatenate and produce a video 