import cv2
import os

############################################################

VIDEO_PATH = 'yolo/Blsga6NTnb4.mp4'
OUTPUT_FOLDER = 'yolo/dataset/images/train'
SPACE_BETWEEN_FRAMES = 60

############################################################

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    if frame_count % SPACE_BETWEEN_FRAMES == 0:
        frame_file_path = os.path.join(OUTPUT_FOLDER, f"third_{frame_count:05d}.jpg")
        cv2.imwrite(frame_file_path, frame)

    frame_count += 1

video.release()
print(f"Extracted {frame_count//SPACE_BETWEEN_FRAMES} frames to '{OUTPUT_FOLDER}'")