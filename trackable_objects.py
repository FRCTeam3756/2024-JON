import math

class Note:
    def __init__(self):
        self.x = None
        self.y = None
        self.scale = None
        self.ratio = None
        self.confidence = None
        self.distance = None
        self.angle = None
        self.timestamp = None

    def update_frame_location(self, x, y, s, r, timestamp):
        self.x = x
        self.y = y
        self.scale = s
        self.ratio = r
        self.timestamp = timestamp

    def update_confidence(self, conf):
        self.confidence = conf

    def update_relative_location(self, distance, angle):
        self.distance = distance
        self.angle = angle

class Robot:
    def __init__(self, team_colour):
        self.team_colour = team_colour
        self.x = None
        self.y = None
        self.scale = None
        self.ratio = None
        self.confidence = None
        self.distance = None
        self.angle = None
        self.travel_angle = None
        self.travel_speed = None
        self.velocity_x = None
        self.velocity_y = None
        self.acceleration_x = None
        self.acceleration_y = None
        self.timestamp = None
    
    def update_frame_location(self, x, y, s, r, timestamp):
        if self.timestamp is not None and self.x is not None:
            time_diff = timestamp - self.timestamp
            new_velocity_x = (x - self.x) / time_diff
            new_velocity_y = (y - self.y) / time_diff
            if self.velocity_x is not None and self.velocity_y is not None:
                self.acceleration_x = (new_velocity_x - self.velocity_x) / time_diff
                self.acceleration_y = (new_velocity_y - self.velocity_y) / time_diff
            self.velocity_x, self.velocity_y = new_velocity_x, new_velocity_y
        self.x, self.y, self.scale, self.ratio, self.timestamp = x, y, s, r, timestamp

    def update_confidence(self, conf):
        self.confidence = conf

    def update_relative_location(self, distance, angle):
        self.distance = distance
        self.angle = angle
    
    def update_movement(self, travel_angle, travel_speed):
        self.travel_angle = travel_angle
        self.travel_speed = travel_speed
    
    def calculate_speed(self):
        if self.velocity_x is not None and self.velocity_y is not None:
            return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        return None
    
    def predict_position(self, time_step):
        """Predicts future position based on current velocity and travel angle."""
        if self.velocity_x is not None and self.velocity_y is not None:
            pred_x = self.x + self.velocity_x * time_step
            pred_y = self.y + self.velocity_y * time_step
            return pred_x, pred_y
        elif self.travel_angle is not None and self.travel_speed is not None:
            angle_rad = math.radians(self.travel_angle)
            pred_x = self.x + self.travel_speed * math.cos(angle_rad) * time_step
            pred_y = self.y + self.travel_speed * math.sin(angle_rad) * time_step
            return pred_x, pred_y
        return None, None
    
    def is_data_recent(self, current_time):
        time_threshold = 1 #second
        return (current_time - self.timestamp) <= time_threshold if self.timestamp is not None else False