import cv2
import numpy as np

def update(val=0):
    '''Callback function to be called whenever the trackbars are moved.'''
    pass

# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Create a window for displaying the video
cv2.namedWindow('R | G | B | Gray | Original | Canny', cv2.WINDOW_NORMAL)

# Add trackbars for Canny thresholds
cv2.createTrackbar('Lower', 'R | G | B | Gray | Original | Canny', 50, 255, update)
cv2.createTrackbar('Upper', 'R | G | B | Gray | Original | Canny', 150, 255, update)

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
    # Set a threshold to create a binary mask for each channel
    _, r_mask = cv2.threshold(r_channel_hsv, 160, 180, cv2.THRESH_BINARY)
    _, g_mask = cv2.threshold(g_channel_hsv, 35, 80, cv2.THRESH_BINARY)
    _, b_mask = cv2.threshold(b_channel_hsv, 85, 170, cv2.THRESH_BINARY)

    # Use the binary masks to extract the color's objects from the original frame
    r_result = cv2.bitwise_and(gray, gray, mask=r_mask)
    g_result = cv2.bitwise_and(gray, gray, mask=g_mask)
    b_result = cv2.bitwise_and(gray, gray, mask=b_mask)

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

    # Threshold the grayscale results to convert non-black areas to white
    # _, r_result_w = cv2.threshold(r_result, 1, 255, cv2.THRESH_BINARY)
    # _, g_result_w = cv2.threshold(g_result, 1, 255, cv2.THRESH_BINARY)
    # _, b_result_w = cv2.threshold(b_result, 1, 255, cv2.THRESH_BINARY)

    # Convert the binary results to 3-channel BGR for visualization
    # r_result_colored = cv2.merge([r_result_w, r_result_w, r_result_w])
    # g_result_colored = cv2.merge([g_result_w, g_result_w, g_result_w])
    # b_result_colored = cv2.merge([b_result_w, b_result_w, b_result_w])

    # Convert the binary results to 3-channel BGR for visualization
    r_result_c2 = cv2.merge([r_result_white, r_result_white, r_result_white])
    g_result_c2 = cv2.merge([g_result_white, g_result_white, g_result_white])
    b_result_c2 = cv2.merge([b_result_white, b_result_white, b_result_white])


    # Merge the channels to display them in a grid
    top_row = np.hstack([cv2.merge([r_channel, r_channel, r_channel]), 
                        cv2.merge([g_channel, g_channel, g_channel]), 
                        cv2.merge([b_channel, b_channel, b_channel])])
    middle_row = np.hstack([gray_3d, frame_resized, edges_3d])
    # bottom_row = np.hstack([r_result_colored, g_result_colored, b_result_colored])
    bottom_row2 = np.hstack([r_result_c2, g_result_c2, b_result_c2])

    combined = np.vstack([top_row, middle_row, bottom_row2])

    cv2.imshow('R | G | B | Gray | Original | Canny', combined)

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
