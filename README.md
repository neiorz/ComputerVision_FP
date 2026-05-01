# YOLOv8 Human Pose Estimation 

Welcome to the YOLOv8-Human-Pose-Estimation Repository!  This project is dedicated to improving the prediction of the pre-trained YOLOv8l-pose model from Ultralytics. Here, you'll find scripts specifically written to address and mitigate common challenges like reducing False Positives, filling gaps in Missing Detections across consecutive frames, and stabilizing Detections in dynamic environments .

## Installation 

### Option 1: Using pip or make 

```bash
pip install -r requirements.txt

# or 

make install
```

## Usage 

### 1. `pose_predict.py` 

- **Description**: Integrated security surveillance script that combines real-time Human Pose Estimation with Staff Authorization. It tracks individuals, analyzes their behavior (e.g., static, running, walking), and performs real-time ID Card Segmentation to verify authorization.
- **Use Case**: Ideal for autonomous patrolling robots and high-security zones where the system needs to distinguish between authorized staff members and unauthorized visitors while monitoring their actions.
#### **Example Command**:
```
python3 pose_predict.py --model yolov8n-pose.pt --source kinect --is_video --view-img                    #Kinect camera

python pose_predict.py --model yolov8l-pose.pt --source 0 --is_video --view-img                          # webcam

python pose_predict.py --model yolov8l-pose.pt --source video.mp4 --is_video --save-img --view-img       # Video source

python pose_predict.py --model yolov8l-pose.pt --source img_name.jpg --save-img --view-img               # single image

python pose_predict.py --model yolov8l-pose.pt --source folder_name --save-img                           # folder of images

```

### 2. `pose_valid.py` 

- **Description**: Automates the evaluation of the YOLOv8 pose model across multiple confidence thresholds to determine the most effective setting.

- **Use Case**: Essential for optimizing model accuracy by identifying the ideal confidence threshold through systematic testing and metric analysis.

- **Features**:
  - 🎚 **Automated Threshold Testing**: Runs the model validation over a series of confidence thresholds.
  - 📈 **Performance Metrics Recording**: Collects and logs important metrics  like Precision (P), Recall (R), mean Average Precision (mAP), and F1 Score for each threshold inside a csv file.

**Example Command**:
  ```bash
  python pose_valid.py --model_file yolov8l-pose.yaml --weights yolov8l-pose.pt --dataset coco8-pose.yaml
  ```

### 3. `pose_fusion_predict.py` 

- **Description**: Combines object tracking with an auxiliary segmentation network to enhance pose estimation results from the YOLOv8 model.

- **Use Case**: Ideal for scenarios where a single model's output needs refinement, especially in terms of accuracy and stability in pose detection.

**Note**: This script is still under development and serves as a prototype for the proposed method. It offers a glimpse into the fusion technique for improved pose estimation.

- **Example Command**:
  ```bash
  python pose_fusion_predict.py --pose_model yolov8l-pose.pt --seg_model yolov8l-seg.pt --source video.mp4 --is_video --save-img --view-img
  ```

### 4. `pose_custom_data_train.py` 

- **Description**: Fine-tune the YOLOv8 pose detection model on a custom dataset. This process involves retraining the pre-trained model with data that's more specific to the task, enhancing model specificity and accuracy.

- **Use Case**: Optimal for scenarios requiring the model to adapt to unique environments or objects. Especially beneficial in improving detection precision and reducing false positives/negatives in challenging or borderline cases.

**Note**: Utilizing tools like Intel's CVAT for keypoint annotation.

- **Example Command**:
  ```bash
  python pose_custom_data_train.py --model_file yolov8l-pose.yaml --weights yolov8l-pose.pt --dataset your_custom_dataset.yaml
  ```

### 5. `pose_custom_data_tune.py` 

- **Description**: hyperparameter tuning of the YOLOv8 pose detection model using custom datasets. This script can help refines the model by adjusting specific parameters to align closely with the unique features of the data.

- **Use Case**: Ideal when seeking to enhance the precision and effectiveness of the YOLOv8 model for specific use cases, particularly where default settings might not yield optimal results.




**Note**: This script is designed to optimize the model's learning process through tailored hyperparameter adjustments, ensuring a more accurate and efficient performance on the custom data.

- **Example Command**:
  ```bash
  python pose_custom_data_tune.py --weights yolov8l-pose.pt --dataset your_custom_dataset.yaml
  ```




  ### 🛠️ How it Works (System Pipeline)
The system follows a hierarchical processing chain to maximize efficiency:

1. **Human Detection & Tracking:** The system first uses **YOLOv8-Pose** to identify people in the frame and assign them a unique `Track ID`.
2. **Pose Analysis:** Keypoints are extracted to determine the person's current behavior (analyzed via `pose_analysis.py`).
3. **Dynamic ROI Cropping:** Instead of scanning the whole frame for an ID, the system uses the person's shoulder keypoints to crop a **Region of Interest (ROI)** around the chest area.
4. **Staff Verification:** A custom-trained **YOLOv8-Segmentation** model scans only the cropped ROI for a company ID card.
5. **Authorization:** - If an ID is detected: Status is set to 🟢 **STAFF**.
   - If no ID is detected: Status remains 🔴 **VISITOR**.
