import numpy as np

class BehaviorAnalyzer:
    def __init__(self):
        # Threshold for the knee angle to distinguish between standing and moving
        self.angle_threshold = 145 

    def calculate_knee_angle(self, hip, knee, ankle):
        """Calculates the angle at the knee joint."""
        # Convert to numpy arrays
        a = np.array(hip)
        b = np.array(knee)
        c = np.array(ankle)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Use cosine rule to find the angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def get_leg_behavior(self, keypoints):
        """
        Determines if the person is Static or Moving based on knee angles.
        COCO Keypoint Indices: 11/12 (Hips), 13/14 (Knees), 15/16 (Ankles)
        """
        try:
            # Extract coordinates (x, y)
            l_hip, l_knee, l_ankle = keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
            r_hip, r_knee, r_ankle = keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]

            # Calculate angles for both legs
            left_angle = self.calculate_knee_angle(l_hip, l_knee, l_ankle)
            right_angle = self.calculate_knee_angle(r_hip, r_knee, r_ankle)
            
            avg_angle = (left_angle + right_angle) / 2

            # Logic: If knees are bent (low angle), the person is likely moving
            if avg_angle < self.angle_threshold:
                return "Active/Moving", (0, 255, 0) # Green
            else:
                return "Static/Standing", (255, 255, 255) # White
        except:
            return "Detecting...", (127, 127, 127)
