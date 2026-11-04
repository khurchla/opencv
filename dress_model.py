import cv2
import numpy as np
import mediapipe as mp

# Load the person image and the shirt image
person_img = cv2.imread("data/W+NK+SS+TEE+OC+BB+VINTAGE.png")
shirt_img = cv2.imread("data/Screenshot_2024-10-29_at_5.06.08_PM-removebg-preview.png")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Perform pose estimation
results = pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))

# Extract keypoints for shoulders and hips
keypoints = {}
if results.pose_landmarks:
    for landmark in mp_pose.PoseLandmark:
        if landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]:
            keypoints[landmark.name] = (
                int(results.pose_landmarks.landmark[landmark].x * person_img.shape[1]),
                int(results.pose_landmarks.landmark[landmark].y * person_img.shape[0])
            )

# Define source points on the shirt image (these points should be manually chosen based on the shirt image)
src_points = np.float32([
    [50, 50],  # Left shoulder on shirt image
    [shirt_img.shape[1] - 50, 50],  # Right shoulder on shirt image
    [50, shirt_img.shape[0] - 50],  # Left hip on shirt image
    [shirt_img.shape[1] - 50, shirt_img.shape[0] - 50]  # Right hip on shirt image
])

# Define destination points on the person image using the detected keypoints
dst_points = np.float32([
    keypoints['LEFT_SHOULDER'],
    keypoints['RIGHT_SHOULDER'],
    keypoints['LEFT_HIP'],
    keypoints['RIGHT_HIP']
])

# Compute the affine transformation matrix
M = cv2.getAffineTransform(src_points[:3], dst_points[:3])

# Apply the affine transformation to the shirt image
warped_shirt_img = cv2.warpAffine(shirt_img, M, (person_img.shape[1], person_img.shape[0]))

# Convert the shirt image to HSV color space for better background removal
hsv_shirt = cv2.cvtColor(warped_shirt_img, cv2.COLOR_BGR2HSV)

# Define the color range for the shirt background (adjust these values as needed)
lower_color = np.array([0, 0, 0])
upper_color = np.array([180, 255, 30])

# Create a mask for the shirt background
mask = cv2.inRange(hsv_shirt, lower_color, upper_color)

# Invert the mask to get the shirt area
mask_inv = cv2.bitwise_not(mask)

# Ensure the mask is of type CV_8U
mask_inv = mask_inv.astype(np.uint8)

# Black-out the area of the shirt in the person image
person_bg = cv2.bitwise_and(person_img, person_img, mask=mask)

# Take only the region of the shirt from the warped shirt image
shirt_fg = cv2.bitwise_and(warped_shirt_img, warped_shirt_img, mask=mask_inv)

# Combine the person image background and the shirt foreground
result = cv2.add(person_bg, shirt_fg)

# Define the output file path
output_file_path = 'output/result_image.png'

# Save the result image to the output file path
cv2.imwrite(output_file_path, result)

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()