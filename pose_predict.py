import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils.files import increment_path
# Import the behavior logic from your other file
from pose_analysis import BehaviorAnalyzer

def pose_estimation(
    model, source, is_video=False, view_img=False, save_img=False, exist_ok=False
):
    if source != 0 and not Path(str(source)).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if is_video or source == 0:
        # Video setup
        cap = cv2.VideoCapture(source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # Output setup
        save_dir = increment_path(Path("output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_filename = (
            "webcam_output.mp4" if source == 0 else f"{Path(source).stem}.mp4"
        )
        output_path = str(save_dir / output_filename)
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )

        track_history = {}
        # Initialize the Behavior Analyzer
        analyzer = BehaviorAnalyzer()

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = model.track(
                    frame,
                    conf=0.5,
                    iou=0.7,
                    device="cpu",
                    imgsz=640,
                    tracker="bytetrack.yaml",
                    persist=True,
                    retina_masks=True,
                    augment=True,
                )

                img_annotated = results[0].plot(boxes=True)

                # Process tracking and behavior analysis
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    keypoints_data = results[0].keypoints.data # [N, 17, 3]

                    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                        # 1. Get keypoints and box for behavior analysis
                        current_kpts = keypoints_data[i].cpu().numpy()
                        current_box = box.tolist()

                        # 2. Get behavior from the analyzer (Hybrid: Angle + Movement)
                        behavior, color = analyzer.get_behavior(track_id, current_kpts, current_box)
                        
                        # 3. Draw the Behavior label on the frame
                        cv2.putText(img_annotated, f"ID:{track_id} {behavior}", 
                                    (int(box[0]), int(box[1] - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # 4. Process tracking lines (Original Logic)
                        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        if source != 0:
                            track = track_history.get(track_id, [])
                            track.append((float(bbox_center[0]), float(bbox_center[1])))

                            if len(track) > 10:
                                track.pop(0)

                            track_history[track_id] = track

                            points = (
                                np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            )
                            cv2.polylines(
                                img_annotated,
                                [points],
                                isClosed=False,
                                color=(0, 0, 255),
                                thickness=2,
                            )

                if view_img:
                    cv2.imshow("Yaqz Security - Behavior Analysis", img_annotated)

                if save_img:
                    video_writer.write(img_annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
    else:  # if image file
        image = cv2.imread(source)
        results = model.predict(image, conf=0.5, iou=0.7, device="cpu", imgsz=640, retina_masks=True)
        img_annotated = results[0].plot(boxes=True)

        if view_img:
            cv2.imshow("Image Pose Estimation", img_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_img:
            save_dir = Path("output_pose") / "exp"
            save_dir.mkdir(parents=True, exist_ok=True)
            img_path = save_dir / (Path(source).stem + "_pose.jpg")
            cv2.imwrite(str(img_path), img_annotated)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8l-pose.pt")
    parser.add_argument("--source", type=str, default="People-Walking-2.mp4")
    parser.add_argument("--is_video", action="store_true")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()

def main(local_opt):
    model = YOLO(local_opt.model)
    if os.path.isdir(local_opt.source):
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
