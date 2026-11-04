import cv2
import numpy as np
import torch
from torchvision import models, transforms

# Load the person image and the shirt image
person_img = cv2.imread("data/W+NK+SS+TEE+OC+BB+VINTAGE.png")
shirt_img = cv2.imread("data/Screenshot_2024-10-29_at_5.06.08_PM-removebg-preview.png", cv2.IMREAD_UNCHANGED)

# Ensure the shirt image has an alpha channel
if shirt_img.shape[2] == 3:
    # Add an alpha channel
    alpha_channel = np.ones(shirt_img.shape[:2], dtype=shirt_img.dtype) * 255
    shirt_img = np.dstack([shirt_img, alpha_channel])

# Load the pre-trained keypoint detection model
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

# Transform the person image
person_img_tensor = transform(person_img)

# Add a batch dimension
person_img_tensor = person_img_tensor.unsqueeze(0)

# Perform keypoint detection
with torch.no_grad():
    outputs = model(person_img_tensor)

# Extract keypoints for shoulders and hips
keypoints = outputs[0]['keypoints'][0].numpy()
left_shoulder = keypoints[5][:2]
right_shoulder = keypoints[6][:2]
left_hip = keypoints[11][:2]
right_hip = keypoints[12][:2]

# Define source points on the shirt image (these points should be manually chosen based on the shirt image)
src_points = np.float32([
    [50, 50],  # Left shoulder on shirt image
    [shirt_img.shape[1] - 50, 50],  # Right shoulder on shirt image
    [50, shirt_img.shape[0] - 50],  # Left hip on shirt image
    [shirt_img.shape[1] - 50, shirt_img.shape[0] - 50]  # Right hip on shirt image
])

# Define destination points on the person image using the detected keypoints
dst_points = np.float32([
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip
])

# Compute the affine transformation matrix
M = cv2.getAffineTransform(src_points[:3], dst_points[:3])

# Apply the affine transformation to the shirt image
warped_shirt_img = cv2.warpAffine(shirt_img, M, (person_img.shape[1], person_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

# Create a mask from the alpha channel of the warped shirt image
alpha_channel = warped_shirt_img[:, :, 3]
_, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

# Ensure the mask is of type CV_8U
mask = mask.astype(np.uint8)
mask_inv = cv2.bitwise_not(mask)

# Black-out the area of the shirt in the person image
person_bg = cv2.bitwise_and(person_img, person_img, mask=mask_inv)

# Take only the region of the shirt from the warped shirt image
shirt_fg = cv2.bitwise_and(warped_shirt_img[:, :, :3], warped_shirt_img[:, :, :3], mask=mask)

# Combine the person image background and the shirt foreground
result = cv2.add(person_bg, shirt_fg)

# Define the output file path
output_file_path = 'output/dress_model_2_result_image.png'

# Save the result image to the output file path
cv2.imwrite(output_file_path, result)

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()