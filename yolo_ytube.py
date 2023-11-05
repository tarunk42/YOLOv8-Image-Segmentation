# pip3 install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"

import cv2
import numpy as np
import youtube_dl
from ultralytics import YOLO


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


def get_live_stream_url(youtube_url):
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

# Replace with your live YouTube video URL
# https://www.youtube.com/watch?v=1fiF7B6VkCk&ab_channel=SharxDemo
# https://www.youtube.com/watch?v=5_XSYlAfJZM&ab_channel=BradPhillips
# https://www.youtube.com/watch?v=EuOZeHQmg-4&ab_channel=%E6%9C%9D%E6%97%A5%E6%96%B0%E8%81%9ELIVE
# https://www.youtube.com/watch?v=1EiC9bvVGnk&ab_channel=SeeJacksonHole
# https://www.youtube.com/watch?v=w_DfTc7F5oQ&ab_channel=Teleport.camera
# https://www.youtube.com/watch?v=B0YjuKbVZ5w&ab_channel=ColoradoMountainCollege
# https://www.youtube.com/watch?v=ByED80IKdIU&ab_channel=CityofColdwaterMI
# https://www.youtube.com/watch?v=nE8qkw9u7tQ&ab_channel=kidneybeansglobal

youtube_url = 'https://www.youtube.com/watch?v=nE8qkw9u7tQ&ab_channel=kidneybeansglobal'
stream_url = get_live_stream_url(youtube_url)

cap = cv2.VideoCapture(stream_url)
model = YOLO("yolov8m.pt")

frame_skip = 2  # Process every 8th frame for efficiency
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        break
    elif frame_count % frame_skip != 0:
        continue

    results = model(frame, device='mps')
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")


    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        class_name = class_dict[cls]
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 1)
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


    cv2.imshow("Live YouTube Feed", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
