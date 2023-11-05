import cv2
import numpy as np

def update(val=0):
    '''Callback function to be called whenever the trackbars are moved.'''
    pass

# Load the video
video_path = './video.mp4'
cap = cv2.VideoCapture(video_path)

# Create a window for displaying the video
cv2.namedWindow('R | G | B | Gray | Original | Canny', cv2.WINDOW_NORMAL)

# Add trackbars for Canny thresholds
cv2.createTrackbar('Lower', 'R | G | B | Gray | Original | Canny', 50, 255, update)
cv2.createTrackbar('Upper', 'R | G | B | Gray | Original | Canny', 150, 255, update)

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

    # Get the current positions of the trackbars
    lower_threshold = cv2.getTrackbarPos('Lower', 'R | G | B | Gray | Original | Canny')
    upper_threshold = cv2.getTrackbarPos('Upper', 'R | G | B | Gray | Original | Canny')

    # Apply Canny edge detection using the trackbar values
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    edges_3d = cv2.merge([edges, edges, edges])

    # Merge the channels to display them in a grid
    top_row = np.hstack([cv2.merge([r_channel, r_channel, r_channel]), 
                        cv2.merge([g_channel, g_channel, g_channel]), 
                        cv2.merge([b_channel, b_channel, b_channel])])

    bottom_row = np.hstack([gray_3d, frame_resized, edges_3d])
    combined = np.vstack([top_row, bottom_row])

    cv2.imshow('R | G | B | Gray | Original | Canny', combined)

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()