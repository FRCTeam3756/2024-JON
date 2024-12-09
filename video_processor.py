import cv2
import math
import time
import torch
import logging
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from decision_matrix import DecisionMatrix
from stereo_video import StereoVision
from mono_video import MonoVision
from trackable_objects import Note
from rio_communication import post_to_network_tables

###############################################################


class Config:
    """Configuration settings for video processing."""
    COVERAGE_THRESHOLD = 0.4
    CONFIDENCE_THRESHOLD = 0.7
    DISPLAY = True
    VIDEO_PATH = "http://limelight.local:5800" #"video.mp4" #0 #
    LABEL_COLORS = {
        "0": [0, 155, 255],
        "1": [0, 0, 255],
        "2": [255, 0, 0]
    }

class Logger:
    @staticmethod
    def setup_logging():
        logging.basicConfig(level=logging.INFO, filename="logs/test.log", filemode="w")
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLODetector:
    def __init__(self, device):
        self.device = device
        self.model = YOLO('yolov11n/runs/train/weights/best.pt').to(self.device)

    def detect(self, frame):
        """Run detection on a frame and return processed results."""
        results = self.model(frame)[0]
        boxes, confidences, class_ids = self.extract_detections(results)
        return boxes, confidences, class_ids

    def extract_detections(self, results):
        """Extract bounding boxes, confidences, and class IDs."""
        boxes, confidences, class_ids = [], [], []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence >= Config.CONFIDENCE_THRESHOLD:
                boxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(class_id)

        return np.array(boxes), np.array(confidences), np.array(class_ids)

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.2)

    def track(self, frame, boxes, confidences, class_ids):
        """Track objects in the frame."""
        detections = [[x1, y1, x2 - x1, y2 - y1, confidences[i], class_ids[i]]
                      for i, (x1, y1, x2, y2) in enumerate(boxes)]
        tracked_objects = self.tracker.update_tracks(detections, frame=frame)
        return tracked_objects

class VideoDisplay:
    @staticmethod
    def show_frame(frame):
        if Config.DISPLAY:
            cv2.imshow('Video', frame)

    @staticmethod
    def annotate_frame(frame, boxes, class_ids):
        """Annotate the frame with bounding boxes and labels."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = Config.LABEL_COLORS.get(str(class_ids[i]), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    @staticmethod
    def draw_angle_line(frame, angle):
        """Draws a line at a given angle from the bottom center of the screen."""
        height, width = frame.shape[:2]
        start_point = (width // 2, height - 1)
        length = 100
        end_x = int(start_point[0] + length * math.sin(math.radians(angle)))
        end_y = int(start_point[1] - length * math.cos(math.radians(angle)))
        cv2.line(frame, start_point, (end_x, end_y), (0, 155, 255), 2)

class FrameProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Logger.setup_logging()
        logging.info(f'Using device: {"GPU" if torch.cuda.is_available() else "CPU"}')
        self.detector = YOLODetector(self.device)
        self.tracker = DeepSortTracker()
        self.decision_matrix = DecisionMatrix()
        self.property_calculation = MonoVision()
        self.depth_estimation = StereoVision()
        self.notes = []
        self.frame_count = 0
        self.start_time = time.time()  

    def process_frame(self, frame):
        """Processes a single frame for detections and annotations."""
        frame = cv2.resize(frame, (640, 640))
        boxes, confidences, class_ids = self.detector.detect(frame)
        
        if boxes.size > 0:
            indices = self.apply_nms(boxes, confidences)
            VideoDisplay.annotate_frame(frame, boxes[indices], class_ids)
            self.update_notes(boxes[indices], confidences, class_ids)

        if self.notes:
            note = self.decision_matrix.compute_best_game_piece(*self.notes)
        else:
            note = None
        return frame, note

    def update_notes(self, boxes, confidences, class_ids):
        """Update notes with detection data for game piece selection."""
        self.notes.clear()
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == 0:
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                scale = ((x2 - x1) + (y2 - y1)) / 2
                ratio = (x2 - x1) / (y2 - y1)
                
                note = Note()
                note.update_frame_location(center_x, center_y, scale, ratio)
                note.update_confidence(conf)
                distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                note.update_physical_location(distance, angle)
                self.notes.append(note)

    def apply_nms(self, boxes, confidences):
        """Apply Non-Maximum Suppression to filter bounding boxes."""
        if boxes.size == 0:
            return np.array([])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32, device=self.device)
        indices = torch.ops.torchvision.nms(boxes_tensor, confidences_tensor, Config.COVERAGE_THRESHOLD)
        return indices.cpu().numpy() if indices.numel() > 0 else np.array([])

    def calculate_frame_rate(self):
        """Calculate and log the frame processing rate."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            logging.info(f"Processing FPS: {fps:.2f}")
            self.start_time = time.time()
            self.frame_count = 0


def main():
    Logger.setup_logging()
    processor = FrameProcessor()
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        logging.error(f"Error opening video: {Config.VIDEO_PATH}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream.")
                break

            processed_frame, note = processor.process_frame(frame)
            if note:
                post_to_network_tables((note.distance, note.angle))
                
            if Config.DISPLAY:
                if note:
                    VideoDisplay.draw_angle_line(frame, note.angle)
                VideoDisplay.show_frame(processed_frame)
                
            processor.calculate_frame_rate()

            if Config.DISPLAY & cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

###############################################################

if __name__ == "__main__":
    main()