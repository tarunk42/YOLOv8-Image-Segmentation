import cv2
import os

video_path = './video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a directory to save the frames
if not os.path.exists('frames'):
    os.mkdir('frames')

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        print("Video playback completed.")
        break

    # Display the current frame
    cv2.imshow('Video', frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, save the current frame
    if key == ord('c'):
        frame_name = os.path.join('frames', f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_name, frame)
        frame_count += 1
        print(f"Captured frame_{frame_count}.jpg")

    # If 'q' is pressed, exit the loop
    elif key == ord('q'):
        print("Exiting...")
        break

# Release video and close all windows
cap.release()
cv2.destroyAllWindows()
