import cv2
import numpy as np

# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the first frame and convert it to grayscale
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_skip = 3  # Process every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1

     # If the frame is not returned, break out of the loop
    if not ret:
        break
    elif frame_count % frame_skip != 0:
        continue   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set a threshold to pick only the larger motion vectors
    mask = magnitude > 2

    # Visualize the mask
    vis = np.uint8(mask * 255)
    cv2.imshow("Mask", vis)

    prev_gray = gray

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
