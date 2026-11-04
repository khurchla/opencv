import cv2
import numpy as np
import json

video_path = 'data/explainer-powerpoint-a-touchpoints-01.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize an empty list to store timestamps and positions of the flashing dot
flashing_dot_info = []

# Play the video until the user manually pauses it or quits
print("Playing the video. Press 'p' to pause and select the flashing dot, or 'q' to quit.")
paused = False
roi_selected = False
tracker = None
current_dot = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display the current frame
    cv2.imshow("Video", frame)

    # Wait for user input
    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):
        # Pause the video when 'p' is pressed
        paused = True
        print("Video paused. Please select the flashing dot.")

        # If there is a previously tracked dot, save the end timestamp
        if current_dot:
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            current_dot['dot_end_timestamp'] = current_timestamp
            flashing_dot_info.append(current_dot)
            current_dot = None

        # Allow the user to select the new region of interest (ROI)
        roi = cv2.selectROI("Select Flashing Dot", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Flashing Dot")

        # Extract the region where the dot is located
        (x, y, w, h) = roi
        roi_selected = True

        # Initialize the tracker for the new dot
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, roi)

        # Record the new dot's start timestamp and relative position
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        current_dot = {
            'dot_start_timestamp': current_timestamp,
            'relative_left': round(x / video_width * 100, 2),  # Multiply by 100 and round
            'relative_top': round(y / video_height * 100, 2),  # Multiply by 100 and round
            'relative_width': round(w / video_width * 100, 2),  # Multiply by 100 and round
            'relative_height': round(h / video_height * 100, 2)  # Multiply by 100 and round
        }

        paused = False  # Resume the video after selecting the dot

    elif key == ord('q'):
        # Quit the process when 'q' is pressed
        print("Quit command received. Exiting the process.")
        break

    elif roi_selected and tracker:
        # If the tracker is active, continue tracking the dot
        success, roi = tracker.update(frame)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get current timestamp

        if success:
            # Update the tracker and get the new position
            x, y, w, h = [int(v) for v in roi]

            # Calculate relative positions and round to two decimal places
            relative_left = round(x / video_width * 100, 2)
            relative_top = round(y / video_height * 100, 2)
            relative_width = round(w / video_width * 100, 2)
            relative_height = round(h / video_height * 100, 2)

            # Update current dot information if it moved
            current_dot.update({
                'relative_left': relative_left,
                'relative_top': relative_top,
                'relative_width': relative_width,
                'relative_height': relative_height
            })

        else:
            # If the tracker loses the dot, close the current dot record
            if current_dot:
                current_dot['dot_end_timestamp'] = timestamp
                flashing_dot_info.append(current_dot)
                current_dot = None
                roi_selected = False

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# If the video ends while tracking the dot, save the last instance
if current_dot:
    current_dot['dot_end_timestamp'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    flashing_dot_info.append(current_dot)

output_file = 'output/flashing_dot_data.json'
try:
    with open(output_file, 'w') as f:
        json.dump(flashing_dot_info, f, indent=4)
    print(f"Flashing dot data has been written to {output_file}.")
except IOError as e:
    print(f"Failed to write flashing dot data to {output_file}: {e}")