import cv2
import numpy as np

# Load the video
video_path = './video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    
    # If the frame is not returned, break out of the loop
    if not ret:
        break

    # Resize the frame
    frame_resized = cv2.resize(frame, (640, 480))

    # Split the frame into its RGB channels
    r_channel, g_channel, b_channel = cv2.split(frame_resized)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_3d = cv2.merge([gray, gray, gray])

    # Empty section (black image of same size)
    empty_section = np.zeros_like(frame_resized)

    # Merge the channels to display them in a grid
    top_row = np.hstack([cv2.merge([r_channel, r_channel, r_channel]), 
                         cv2.merge([g_channel, g_channel, g_channel]), 
                         cv2.merge([b_channel, b_channel, b_channel])])
    
    bottom_row = np.hstack([gray_3d, frame_resized, empty_section])
    combined = np.vstack([top_row, bottom_row])

    cv2.imshow('R | G | B | Gray | Original | Empty', combined)

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
