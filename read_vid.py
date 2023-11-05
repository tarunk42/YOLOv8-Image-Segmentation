import cv2

# Load the video
video_path = './video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    
    # If the frame is not returned, break out of the loop
    if not ret:
        break

    cv2.imshow('Video Playback', frame)

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
