import cv2
import numpy as np

def update(val=0):
    '''Callback function to be called whenever the trackbars are moved.'''
    pass

# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the first frame and convert it to grayscale
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (640, 480))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create a window for displaying the video
cv2.namedWindow('R | G | B | Gray | Original | Canny', cv2.WINDOW_NORMAL)

# Add trackbars for Canny thresholds
cv2.createTrackbar('Lower', 'R | G | B | Gray | Original | Canny', 50, 255, update)
cv2.createTrackbar('Upper', 'R | G | B | Gray | Original | Canny', 150, 255, update)

frame_skip = 4  # Process every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    
    # If the frame is not returned, break out of the loop
    if not ret:
        break
    elif frame_count % frame_skip != 0:
        continue


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

    # convert to hsv
    frame_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    r_channel_hsv, g_channel_hsv, b_channel_hsv = cv2.split(frame_hsv)

    #############################

    # Define range for red color
    lower_red1 = np.array([0,50,50])
    upper_red1 = np.array([10,255,255])
    red_mask1 = cv2.inRange(frame_hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160,50,50])
    upper_red2 = np.array([180,255,255])
    red_mask2 = cv2.inRange(frame_hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Define range for green color
    lower_green = np.array([35,50,50])
    upper_green = np.array([85,255,255])
    green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

    # Define range for blue color
    lower_blue = np.array([85,50,50])
    upper_blue = np.array([170,255,255])
    blue_mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

    # Extract the color's objects from the grayscale frame
    r_result_white = cv2.bitwise_and(gray, gray, mask=red_mask)
    g_result_white = cv2.bitwise_and(gray, gray, mask=green_mask)
    b_result_white = cv2.bitwise_and(gray, gray, mask=blue_mask)

    ########################

    # Convert the binary results to 3-channel BGR for visualization
    r_result_c2 = cv2.merge([r_result_white, r_result_white, r_result_white])
    g_result_c2 = cv2.merge([g_result_white, g_result_white, g_result_white])
    b_result_c2 = cv2.merge([b_result_white, b_result_white, b_result_white])

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set a threshold to pick only the larger motion vectors
    mask = magnitude > 2

    vis = np.uint8(mask * 255)

    # kernel = np.ones((3,3),np.uint8)
    # vis = cv2.erode(vis, kernel, iterations = 1)
    # vis = cv2.dilate(vis, kernel, iterations = 1)

    # Extract moving regions from the original frame using the optical flow mask
    moving_objects = cv2.bitwise_and(frame_resized, frame_resized, mask=vis)

    vis_3d = cv2.merge([vis, vis, vis])

    # Empty section (black image of same size)
    empty_section = np.zeros_like(frame_resized)

    # Merge the channels to display them in a grid
    top_row = np.hstack([cv2.merge([r_channel, r_channel, r_channel]), 
                        cv2.merge([g_channel, g_channel, g_channel]), 
                        cv2.merge([b_channel, b_channel, b_channel]), moving_objects])
    middle_row = np.hstack([gray_3d, frame_resized, edges_3d, empty_section])
    bottom_row = np.hstack([r_result_c2, g_result_c2, b_result_c2, vis_3d])

    combined = np.vstack([top_row, middle_row, bottom_row])

    cv2.imshow('R | G | B | Gray | Original | Canny', combined)

    prev_gray = gray

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
