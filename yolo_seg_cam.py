import cv2
from ultralytics import YOLO
import youtube_dl

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

def get_live_stream_url(youtube_url):
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']
    
youtube_url = 'https://www.youtube.com/watch?v=5_XSYlAfJZM&ab_channel=BradPhillips'
stream_url = get_live_stream_url(youtube_url)

# Open the video file
cap = cv2.VideoCapture(stream_url)

frame_skip = 2  # Process every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    
    # If the frame is not returned, break out of the loop
    if not ret:
        break
    elif frame_count % frame_skip != 0:
        continue

    results = model(frame, device='mps')

    annotated_frame = results[0].plot(boxes=False)


    if not ret:
        break
    cv2.imshow("Img", annotated_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()