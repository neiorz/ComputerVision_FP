import numpy as np

class PoseAnalyzer:
    def __init__(self):
        # Dictionary to store the previous positions of each person (using their track_id)
        self.history = {}

        # Thresholds to distinguish between different types of movement
        self.velocity_threshold_run = 0.15
        self.velocity_threshold_walk = 0.03

        # Threshold for the knee angle (Running involves more bending)
        self.run_angle_threshold = 130

    def calculate_velocity(self, current_points, track_id):
        """Calculates the movement speed by comparing current and previous ankle positions"""
        if track_id not in self.history:
            self.history[track_id] = current_points
            return 0

        prev_points = self.history[track_id]

        # Calculate Euclidean distance for left (15) and right (16) ankles
        dist_l = np.linalg.norm(current_points[15] - prev_points[15])
        dist_r = np.linalg.norm(current_points[16] - prev_points[16])

        # Take the average distance as the final velocity
        velocity = (dist_l + dist_r) / 2

        # Update history with current points for the next frame
        self.history[track_id] = current_points
        return velocity

    def calculate_angle(self, a, b, c):
        """Calculates the angle at joint 'b' using the coordinates of points a, b, and c"""
        a, b, c = np.array(a), np.array(b), np.array(c)

        # Use arctan2 to find the angle in radians, then convert to degrees
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        # Ensure the angle is the smaller interior angle (0-180)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def classify_behavior(self, keypoints, track_id):
        """Main logic to determine the behavior based on speed and joint angles"""

        # 1. Get current movement speed
        velocity = self.calculate_velocity(keypoints, track_id)

        # 2. Calculate Knee Angles (11/12: Hips, 13/14: Knees, 15/16: Ankles)
        left_knee_angle = self.calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        right_knee_angle = self.calculate_angle(keypoints[12], keypoints[14], keypoints[16])

        # Use the most bent knee for detection
        min_angle = min(left_knee_angle, right_knee_angle)

        # 3. Behavior Logic Tree
        if velocity < self.velocity_threshold_walk:
            status = "Static"
        elif velocity > self.velocity_threshold_run or min_angle < self.run_angle_threshold:
            status = "Running"
        else:
            status = "Walking"

        # 4. Fighting Detection (Check if hands are high and person is moving)
        # 5/6: Shoulders, 15/16: Wrists
        hands_above_shoulders = keypoints[15][1] < keypoints[5][1] or keypoints[16][1] < keypoints[6][1]

        if hands_above_shoulders and velocity > 0.04:
            status = "Fighting!"

        return status
