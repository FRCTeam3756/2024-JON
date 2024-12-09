import math

class StereoVision:
    def __init__(self):
        self.inches_between_cameras = 36   # Distance between the two cameras in inches
        self.frame_width = 640             # Width of the camera frame in pixels
        self.frame_height = 480            # Height of the camera frame in pixels
        self.focal_length_in_mm = 2.75     # Focal length of the camera lens in mm
        self.sensor_width_in_mm = 4        # Sensor width in mm

    def focal_length_in_pixels(self):
        return (self.frame_width / self.sensor_width_in_mm) * self.focal_length_in_mm

    def calculate_disparity(self, left_camera_box, right_camera_box):
        x1 = left_camera_box[0]
        x2 = right_camera_box[0]
        disparity = abs(x1 - x2)
        return disparity

    def calculate_distance(self, disparity):
        if disparity == 0:
            return float('inf')  # Return infinity if object is too far

        focal_length_in_pixels = self.focal_length_in_pixels()
        distance = (focal_length_in_pixels *
                    self.inches_between_cameras) / disparity
        
        return distance

    def calculate_angle(self, left_camera_box, right_camera_box):
        center_x = (left_camera_box[0] + right_camera_box[0]) / 2
        deviation = center_x - (self.frame_width / 2)

        if abs(deviation) < 1e-6:
            return 0.0

        focal_length_in_pixels = self.focal_length_in_pixels()
        angle_radians = math.atan(deviation / focal_length_in_pixels)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def analyze(self, left_camera_box, right_camera_box):
        disparity = self.calculate_disparity(left_camera_box, right_camera_box)
        distance = self.calculate_distance(disparity)
        angle = self.calculate_angle(left_camera_box, right_camera_box)

        return distance, angle