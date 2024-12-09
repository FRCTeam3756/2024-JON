import cv2
import time
import os

video_url = "http://10.37.56.11:5800/"
save_folder = "yolov11n/dataset/images/train"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 1
save_interval = 5
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    current_time = time.time()
    
    if current_time - last_save_time >= save_interval:
        frame_filename = os.path.join(save_folder, f"fifth_frame_{frame_count}.jpg")
        
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

        frame_count += 1
        last_save_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
