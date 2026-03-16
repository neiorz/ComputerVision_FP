import numpy as np

class BehaviorAnalyzer:
    def __init__(self):
        # Thresholds for Knee Angle
        self.angle_threshold = 145 
        # Thresholds for Box Displacement (Pixel distance)
        self.move_threshold = 2.0  # Walking
        self.run_threshold = 6.0   # Running
        # Dictionary to store previous center points for each ID
        self.prev_centers = {}

    def calculate_knee_angle(self, hip, knee, ankle):
        a, b, c = np.array(hip), np.array(knee), np.array(ankle)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_behavior(self, track_id, keypoints, current_box):
        """
        Combines Box Movement and Knee Angles for high accuracy.
        current_box: [x1, y1, x2, y2]
        """
        # --- PART 1: BOX MOVEMENT CALCULATION ---
        current_center = np.array([(current_box[0] + current_box[2])/2, 
                                   (current_box[1] + current_box[3])/2])
        
        movement_speed = 0
        if track_id in self.prev_centers:
            # Calculate distance between current and last frame
            movement_speed = np.linalg.norm(current_center - self.prev_centers[track_id])
        
        self.prev_centers[track_id] = current_center # Update memory

        # --- PART 2: KNEE ANGLE CALCULATION ---
        try:
            l_angle = self.calculate_knee_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
            r_angle = self.calculate_knee_angle(keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
            avg_angle = (l_angle + r_angle) / 2
        except:
            avg_angle = 180 # Default straight legs

        # --- PART 3: COMBINED LOGIC (HYBRID) ---
        # 1. Running: High speed OR very low angle while moving
        if movement_speed > self.run_threshold:
            return "Running", (0, 0, 255) # Red
        
        # 2. Walking: Medium speed OR knees bending
        elif movement_speed > self.move_threshold or avg_angle < self.angle_threshold:
            return "Walking", (0, 255, 0) # Green
            
        # 3. Static: No movement and straight legs
        else:
            return "Static", (255, 255, 255) # White
