import os
import cv2
import freenect
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path
from pose_analysis import BehaviorAnalyzer

# --- Load the Custom ID Segmentation Model ---
id_seg_model = YOLO('id_seg_model.pt') 

def pose_estimation(
    model, source, is_video=False, view_img=False, save_img=False, exist_ok=False
):  
    is_kinect = (source == "kinect")
    
    if is_kinect:
        fps, frame_width, frame_height = 30, 640, 480

    if source != "kinect" and source != 0 and not Path(str(source)).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if is_video or source == 0 or is_kinect:

        # Video setup for files, webcam, or Kinect
        if not is_kinect:
            cap = cv2.VideoCapture(source)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30 
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output directory and file setup
        save_dir = increment_path(Path("output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_filename = "kinect_output.mp4" if is_kinect else ("webcam_output.mp4" if source == 0 else f"{Path(source).stem}.mp4")
        output_path = str(save_dir / output_filename)
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

        track_history = {}
        analyzer = BehaviorAnalyzer()

        while True:
            if is_kinect:

                # Get frame from Kinect sensor and convert to BGR for OpenCV
                frame, _ = freenect.sync_get_video()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                success = True
            else:
                success, frame = cap.read()

            if success:

                # Run YOLOv8 Pose Tracking
                results = model.track(
                    frame, conf=0.5, iou=0.7, device="cpu", imgsz=640,
                    tracker="bytetrack.yaml", 
                    persist=True, retina_masks=True, augment=True,
                )

                img_annotated = results[0].plot(boxes=True)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    keypoints_data = results[0].keypoints.data

                    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        
                        # --- 1. ID Segmentation Logic ---
                        # Crop the chest region (top 40% of the bounding box)
                        roi_h = int((y2 - y1) * 0.4)
                        chest_roi = frame[max(0, y1):y1+roi_h, max(0, x1):x2]
                        
                        is_staff = False
                        if chest_roi.size > 0:

                            # Run segmentation model on the cropped chest area
                            id_results = id_seg_model(chest_roi, conf=0.5, verbose=False)
                            
                            # If a mask is detected, categorize as STAFF
                            if id_results[0].masks is not None:
                                is_staff = True
                        
                        # Set display text and color based on authorization status
                        status_txt = "STAFF" if is_staff else "VISITOR"
                        status_color = (0, 255, 0) if is_staff else (0, 0, 255) # Green for Staff, Red for Visitor
                        
                        # --- 2. Behavior Analysis Logic ---
                        current_kpts = keypoints_data[i].cpu().numpy()
                        current_box = box.tolist()
                        behavior, b_color = analyzer.get_behavior(track_id, current_kpts, current_box)
                        
                        # Draw status and behavior info on the frame
                        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), status_color, 2)
                        cv2.putText(img_annotated, f"ID:{track_id} {status_txt}", 
                                    (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                        cv2.putText(img_annotated, f"Act: {behavior}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 1)

                        # --- 3. Tracking Lines Logic ---
                        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        track = track_history.get(track_id, [])
                        track.append((float(bbox_center[0]), float(bbox_center[1])))
                        if len(track) > 10: track.pop(0)
                        track_history[track_id] = track
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_annotated, [points], isClosed=False, color=(0, 0, 255), thickness=2)

                if view_img:
                    cv2.imshow("Yaqz Security - Full Analysis", img_annotated)
                if save_img:
                    video_writer.write(img_annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        video_writer.release()
        if not is_kinect: cap.release()
        cv2.destroyAllWindows()
    else:
        # Static image processing
        image = cv2.imread(source)
        results = model.predict(image, conf=0.5, iou=0.7)
        img_annotated = results[0].plot()
        if view_img:
            cv2.imshow("Image Pose Estimation", img_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--source", type=str, default="0") # 0 for webcam, "kinect" for Kinect
    parser.add_argument("--is_video", action="store_true")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()

def main(local_opt):
    model = YOLO(local_opt.model)
    if local_opt.source == "kinect":
        pose_estimation(model, "kinect", 
                        is_video=local_opt.is_video, view_img=local_opt.view_img, 
                        save_img=local_opt.save_img, exist_ok=local_opt.exist_ok)
    elif os.path.isdir(local_opt.source):
        for filename in os.listdir(local_opt.source):
            pose_estimation(model, os.path.join(local_opt.source, filename), 
                            is_video=local_opt.is_video, view_img=local_opt.view_img, 
                            save_img=local_opt.save_img, exist_ok=local_opt.exist_ok)
    else:
        src = int(local_opt.source) if local_opt.source == "0" else local_opt.source
        pose_estimation(model, src, is_video=local_opt.is_video, view_img=local_opt.view_img, 
                        save_img=local_opt.save_img, exist_ok=local_opt.exist_ok)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
