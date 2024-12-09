import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

########################################################

IOU_THRESHOLD = 0.3     # Higher is more selective
MAX_TRACKER_AGE = 2     # Lower is more selective
MIN_HIT_STREAK = 3      # Higher is more selective

########################################################

def solve_linear_assignment(cost_matrix):
    """Solve the linear assignment problem."""
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return np.column_stack((row_indices, col_indices))

def compute_iou_matrix(bboxes_test, bboxes_gt):
    """Compute the Intersection over Union (IoU) matrix for a batch of bounding boxes."""
    bboxes_test = np.expand_dims(bboxes_test, 1)
    bboxes_gt = np.expand_dims(bboxes_gt, 0)

    x_min_intersection = np.maximum(bboxes_test[..., 0], bboxes_gt[..., 0])
    y_min_intersection = np.maximum(bboxes_test[..., 1], bboxes_gt[..., 1])
    x_max_intersection = np.minimum(bboxes_test[..., 2], bboxes_gt[..., 2])
    y_max_intersection = np.minimum(bboxes_test[..., 3], bboxes_gt[..., 3])

    intersection_width = np.maximum(0., x_max_intersection - x_min_intersection)
    intersection_height = np.maximum(0., y_max_intersection - y_min_intersection)
    intersection_area = intersection_width * intersection_height

    bboxes_test_area = (bboxes_test[..., 2] - bboxes_test[..., 0]) * (bboxes_test[..., 3] - bboxes_test[..., 1])
    bboxes_gt_area = (bboxes_gt[..., 2] - bboxes_gt[..., 0]) * (bboxes_gt[..., 3] - bboxes_gt[..., 1])

    iou = intersection_area / (bboxes_test_area + bboxes_gt_area - intersection_area)

    return iou

def convert_bbox_to_center_form(bbox):
    """Convert bounding box from corner to center format."""
    x1, y1, x2, y2 = bbox[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    scale = width * height
    ratio = width / height

    return np.array([center_x, center_y, scale, ratio]).reshape((4, 1))

def convert_center_format_to_bbox(center_format_bbox):
    """Convert a bounding box in center format to corner format."""
    center_x, center_y, scale, ratio = center_format_bbox[:4]
    width = np.sqrt(scale * ratio)
    height = scale / width
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    if len(center_format_bbox) > 4:
        score = center_format_bbox[4]
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))
    return np.array([x1, y1, x2, y2]).reshape((1, 4))

def associate_detections_to_trackers(detections, trackers, iou_threshold=IOU_THRESHOLD):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = compute_iou_matrix(detections, trackers)
    matching_matrix = (iou_matrix > iou_threshold).astype(int)
    
    if matching_matrix.sum(1).max() == 1 and matching_matrix.sum(0).max() == 1:
        matched_indices = np.argwhere(matching_matrix)
    else:
        matched_indices = solve_linear_assignment(-iou_matrix)

    unmatched_detections = [i for i in range(len(detections)) if i not in matched_indices[:, 0]]
    unmatched_trackers = [i for i in range(len(trackers)) if i not in matched_indices[:, 1]]

    matched_detections = set(matched_indices[:, 0])
    matched_trackers = set(matched_indices[:, 1])

    unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
    unmatched_trackers = [i for i in range(len(trackers)) if i not in matched_trackers]

    matches = [match for match in matched_indices if iou_matrix[match[0], match[1]] >= iou_threshold]
    matches = np.array(matches) if matches else np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.setup_kalman_filter()
        self.kf.x[:4] = convert_bbox_to_center_form(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def setup_kalman_filter(self):
        """Initialize Kalman Filter parameters."""
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], 
                              [0, 1, 0, 0, 0, 1, 0], 
                              [0, 0, 1, 0, 0, 0, 1], 
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10
        self.kf.P[4:, 4:] *= 1000
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

    def update(self, bbox):
        """Update the state vector with observed bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_center_form(bbox))

    def predict(self):
        """Predict the next bounding box."""
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_center_format_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return the current bounding box estimate."""
        return convert_center_format_to_bbox(self.kf.x)

class Sort(object):
    def __init__(self):
        self.trackers = []       # List of active KalmanBoxTrackers
        self.frame_count = 0

    def update(self, detections=np.empty((0, 5))):
        """Update all trackers with new detections."""
        self.frame_count += 1
        predicted_bboxes = np.zeros((len(self.trackers), 5))
        to_delete = []
        output_tracks = []
        
        for tracker_idx, pred_bbox in enumerate(predicted_bboxes):
            predicted_state = self.trackers[tracker_idx].predict()[0]
            pred_bbox[:] = [predicted_state[0], predicted_state[1], predicted_state[2], predicted_state[3], 0]
            if np.any(np.isnan(predicted_state)):
                to_delete.append(tracker_idx)
        
        predicted_bboxes = np.ma.compress_rows(np.ma.masked_invalid(predicted_bboxes))
        for tracker_idx in reversed(to_delete):
            self.trackers.pop(tracker_idx)
        
        matched_indices, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, predicted_bboxes)

        for match in matched_indices:
            self.trackers[match[1]].update(detections[match[0], :])

        for detection_idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[detection_idx, :])
            self.trackers.append(new_tracker)

        for tracker_idx in reversed(range(len(self.trackers))):
            current_tracker = self.tracker[tracker_idx]
            track_state = current_tracker.get_state()[0]
            if current_tracker.time_since_update < 1 and (current_tracker.hit_streak >= MIN_HIT_STREAK or self.frame_count <= MIN_HIT_STREAK):
                output_tracks.append(np.concatenate((track_state, [current_tracker.id + 1])).reshape(1, -1))
            if current_tracker.time_since_update > MAX_TRACKER_AGE:
                self.trackers.pop(tracker_idx)
        
        if len(output_tracks) > 0:
            return np.concatenate(output_tracks)
        return np.empty((0, 5))