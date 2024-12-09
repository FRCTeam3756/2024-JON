import math

image_width = 640        # in pixels
image_height = 640       # in pixels
fov_horizontal = 59.6    # in degrees
note_width = 12          # in inches


class MonoVision:
    def __init__(self):
        self.focal_length_px = self.calculate_focal_length()

    def find_distance_and_angle(self, object_x, object_width):
        distance = self.calculate_distance(object_width)
        angle_offset = self.calculate_angular_offset(object_x)

        return distance, angle_offset

    def calculate_focal_length(self):
        fov_rad = math.radians(fov_horizontal)
        focal_length = image_width / (2 * math.tan(fov_rad / 2))

        return focal_length

    def calculate_distance(self, object_width):
        return (note_width * self.focal_length_px) / object_width

    def calculate_angular_offset(self, object_x):
        center_x = image_width / 2
        x_offset = object_x - center_x
        angle_offset = math.degrees(math.atan(x_offset / self.focal_length_px))
        return angle_offset
