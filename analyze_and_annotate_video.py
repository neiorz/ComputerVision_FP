import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import os
import sys

# Add the directory containing pose_analysis.py to sys.path
sys.path.insert(0, '/content/ComputerVision_FP/')

from pose_analysis import PoseAnalyzer

def analyze_and_annotate_video(input_video_path, model_path='/content/yolov8l-pose.pt', output_base_dir='runs/pose/annotated_videos'):
    # Load the YOLOv8 pose model
    model = YOLO(model_path)

    # Initialize the PoseAnalyzer
    analyzer = PoseAnalyzer()

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output directory and create if it doesn't exist
    output_full_dir = os.path.join('/content/', output_base_dir) # Save outside ComputerVision_FP for clarity
    os.makedirs(output_full_dir, exist_ok=True)
    
    # Define the output video path
    output_filename = os.path.basename(input_video_path).replace('.', '_annotated.') if '.' in os.path.basename(input_video_path) else os.path.basename(input_video_path) + '_annotated.mp4'
    output_video_path = os.path.join(output_full_dir, output_filename)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return

    print(f"Processing video: {input_video_path}")
    print(f"Saving annotated video to: {output_video_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0: # Print progress every 100 frames
            print(f"Processed {frame_count} frames...")

        # Perform prediction
        results = model.predict(frame, conf=0.25, iou=0.7, verbose=False, save=False)

        # Create an annotator object to draw on the frame
        annotator = Annotator(frame)

        # Process results and apply custom analysis
        for r in results:
            if r.keypoints is not None and r.boxes.id is not None and len(r.keypoints.data) > 0:
                kp = r.keypoints.xyn.cpu().numpy() # Normalized keypoints
                ids = r.boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(ids):
                    # Your custom analysis logic
                    keypoints_xy = kp[i] # This is already normalized (x,y)
                    # Note: PoseAnalyzer expects non-normalized keypoints typically for pixel distances/angles.
                    # For this example, let's pass scaled keypoints to the analyzer.
                    # Scale keypoints back to original frame dimensions for analyzer
                    scaled_keypoints = np.copy(keypoints_xy)
                    scaled_keypoints[:, 0] = scaled_keypoints[:, 0] * width
                    scaled_keypoints[:, 1] = scaled_keypoints[:, 1] * height

                    label = analyzer.classify_behavior(scaled_keypoints, track_id)

                    # Draw the bounding box and the label on the frame
                    box = r.boxes.xyxy[i] # Unscaled box coordinates
                    annotator.box_label(box, f"ID:{track_id} {label}", color=(0, 255, 0))
                    
                    # Draw keypoints (skeleton) for visual annotation
                    keypoints_with_conf = r.keypoints.data[i].cpu().numpy() # x, y, confidence for current person
                    annotator.kpt_bbox(keypoints_with_conf, orig_shape=(width, height), pil=False)

        # Get the final frame with all drawings applied
        annotated_frame = annotator.result()
        
        # Write the processed frame to the output video
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Analysis complete. Output video saved to: {output_video_path}")

if __name__ == '__main__':
    # Example usage: Ensure the model and video paths are correct
    video_input = "/content/5962271-hd_1920_1080_30fps.mp4"
    analyze_and_annotate_video(video_input)
