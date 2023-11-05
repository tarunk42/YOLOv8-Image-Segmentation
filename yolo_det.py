import cv2
from ultralytics import YOLO
import numpy as np


cap = cv2.VideoCapture("video.mp4")
model = YOLO("yolov8m.pt")

frame_skip = 6  # Process every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    
    # If the frame is not returned, break out of the loop
    if not ret:
        break
    elif frame_count % frame_skip != 0:
        continue

    results = model(frame)
    result = results[0]
    bboxes = result.boxes.xyxy

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    # print(bboxes)

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)


    if not ret:
        break
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
