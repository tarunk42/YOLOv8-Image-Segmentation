import cv2
import numpy as np
import youtube_dl
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

def update(val=0):
    '''Callback function to be called whenever the trackbars are moved.'''
    pass

def get_live_stream_url(youtube_url):
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']
    
class_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
              8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
              14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
              22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
              29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
              36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
              44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 
              53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
              62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 
              71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# Replace with your live YouTube video URL
# https://www.youtube.com/watch?v=5_XSYlAfJZM&ab_channel=BradPhillips
# https://www.youtube.com/watch?v=EuOZeHQmg-4&ab_channel=%E6%9C%9D%E6%97%A5%E6%96%B0%E8%81%9ELIVE
youtube_url = 'https://www.youtube.com/watch?v=5_XSYlAfJZM&ab_channel=BradPhillips'
stream_url = get_live_stream_url(youtube_url)


# Load the video
# video_path = 'video.mp4'
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(stream_url)

# Get the first frame and convert it to grayscale
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (640, 480))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create a window for displaying the video
cv2.namedWindow('R | G | B | Gray | Original | Canny', cv2.WINDOW_NORMAL)

# Add trackbars for Canny thresholds
cv2.createTrackbar('Lower', 'R | G | B | Gray | Original | Canny', 50, 255, update)
cv2.createTrackbar('Upper', 'R | G | B | Gray | Original | Canny', 150, 255, update)

frame_skip = 1  # Process every 3rd frame
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

    # YOLO
    results = model(frame_resized, device='mps')
    annotated_frame = results[0].plot(boxes=False)

    # Split the frame into its RGB channels
    r_channel, g_channel, b_channel = cv2.split(frame_resized)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_3d = cv2.merge([gray, gray, gray])

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set a threshold to pick only the larger motion vectors
    mask = magnitude > 2

    vis = np.uint8(mask * 255)

    # Extract moving regions from the original frame using the optical flow mask
    moving_objects = cv2.bitwise_and(frame_resized, frame_resized, mask=vis)

    vis_3d = cv2.merge([vis, vis, vis])

    # drawing rectangles
    # Find contours in the motion mask
    contours, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define a color for the rectangles
    rect_color = (0, 255, 0)  # Green color

    new_frame = frame_resized.copy()

    # Iterate over the contours
    for contour in contours:
        # Compute the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the rectangle on the original frame
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), rect_color, 2)

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
    r_result_white = cv2.bitwise_and(vis, vis, mask=red_mask)
    g_result_white = cv2.bitwise_and(vis, vis, mask=green_mask)
    b_result_white = cv2.bitwise_and(vis, vis, mask=blue_mask)

    ########################

    # Convert the binary results to 3-channel BGR for visualization
    r_result_c2 = cv2.merge([r_result_white, r_result_white, r_result_white])
    g_result_c2 = cv2.merge([g_result_white, g_result_white, g_result_white])
    b_result_c2 = cv2.merge([b_result_white, b_result_white, b_result_white])

    # Empty section (black image of same size)
    empty_section = np.zeros_like(frame_resized)

    # After constructing each section, add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # White color
    thickness = 2
    position_offset = (10, 25)

    cv2.putText(r_channel, 'Red Channel', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(g_channel, 'Green Channel', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(b_channel, 'Blue Channel', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(gray_3d, 'Gray', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame_resized, 'Original', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(edges_3d, 'Canny Edges', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(r_result_c2, 'Red Objects', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(g_result_c2, 'Green Objects', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(b_result_c2, 'Blue Objects', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(vis_3d, 'Motion', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(moving_objects, 'Motion Masked', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(new_frame, 'Detected Vehicles', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(annotated_frame, 'YOLOv8n-Seg', position_offset, font, font_scale, color, thickness, cv2.LINE_AA)


    # Merge the channels to display them in a grid
    top_row = np.hstack([cv2.merge([r_channel, r_channel, r_channel]), 
                        cv2.merge([g_channel, g_channel, g_channel]), 
                        cv2.merge([b_channel, b_channel, b_channel]), moving_objects, annotated_frame])
    middle_row = np.hstack([gray_3d, frame_resized, edges_3d, new_frame, empty_section])
    bottom_row = np.hstack([r_result_c2, g_result_c2, b_result_c2, vis_3d, empty_section])

    combined = np.vstack([top_row, middle_row, bottom_row])

    cv2.imshow('R | G | B | Gray | Original | Canny', combined)

    prev_gray = gray

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
