''' Detect a light purple/violet dot, record its appearance, and track its position relative to the video dimensions. '''

import cv2
import numpy as np
import json

video_path = 'data/explainer-powerpoint-a-touchpoints-01.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

flashing_dot_info = []

'''
Define thresholds for color detection (adjust for your specific flashing dot color)
Example: for a light purple or violet dot, use HSV color space
Hue (H) for purple/violet is typically around 130 to 160 (for light shades, it might shift towards blue slightly).
Saturation (S) should be medium to high for a distinct purple, so values around 60-200 work well for light purple.
Value (V) controls the brightness, so we’ll set it relatively high to capture the lightness of the purple dot.

How to Fine-Tune the HSV Bounds:
Hue (H): Shift between 130 and 160 for purple/violet hues. If the dot is more bluish, increase this range slightly.
Saturation (S): Lower values may help detect lighter purples. Reduce the lower bound if the dot is very light or pastel-like.
Value (V): Increasing the lower bound might help eliminate darker regions if the dot is distinctly light.
'''
# Define thresholds for detecting light purple/violet in HSV color space
lower_bound = np.array([130, 60, 100])   # Adjust as needed for light purple hue
upper_bound = np.array([160, 200, 255])  # Upper bound for purple/violet

# Variables to track state
dot_present = False
previous_position = None
current_dot = None  # Store the current dot details (for start timestamp)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for the light purple color (violet)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours (blobs) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the current timestamp
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds

    # If contours are found, assume the dot is present
    if contours:
        # Get the largest contour (assuming it's the flashing dot)
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)

        # Calculate the relative position (top-left corner of bounding box)
        relative_x = x / video_width
        relative_y = y / video_height
        relative_w = w / video_width
        relative_h = h / video_height

        # Check if this is a new appearance or a new position
        if not dot_present or (previous_position and (
            abs(relative_x - previous_position['x']) > 0.01 or
            abs(relative_y - previous_position['y']) > 0.01)):

            # If this is a new appearance, save the start timestamp and position
            if not dot_present:
                current_dot = {
                    'dot_start_timestamp': timestamp,
                    'x': relative_x,
                    'y': relative_y,
                    'width': relative_w,
                    'height': relative_h
                }

            # Update the state
            dot_present = True
            previous_position = {'x': relative_x, 'y': relative_y}

    else:
        # If the dot was previously present and now disappears, record the stop time
        if dot_present:
            current_dot['dot_end_timestamp'] = timestamp
            flashing_dot_info.append(current_dot)  # Save the current dot info
            current_dot = None  # Reset current dot state

        # Reset the dot_present flag when the dot is not detected
        dot_present = False

cap.release()

output_file = 'output/flashing_dot_data.json'
try:
    with open(output_file, 'w') as f:
        json.dump(flashing_dot_info, f, indent=4)
    print(f"Flashing dot data has been written to {output_file}.")
except IOError as e:
    print(f"Failed to write flashing dot data to {output_file}: {e}")
