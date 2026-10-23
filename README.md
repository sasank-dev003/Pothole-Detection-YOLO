# YOLO Pothole Detection Model

An end-to-end computer vision pipeline for detecting road potholes in dashcam and traffic videos. This project trains a lightweight YOLO26s model to identify road damage and runs bounding-box inference on video files.

**Performance & Training Results:** 
*   **Model:** YOLO26s (Small) trained on a local RTX 4050 GPU (6GB VRAM)[cite: 17, 18].
*   **Accuracy:** Achieved ~80% mAP50 and ~84% Precision[cite: 18]. 
*   **True Positive Rate:** The normalized confusion matrix shows a 96% accuracy rate for correctly predicting actual potholes[cite: 14].
*   **Dataset:** Trained on a 665-image custom dataset (80/10/10 split)[cite: 18]. 
*   **Training Setup:** Configured with 640x640 image size, batch size 16, and early stopping (patience=10)[cite: 17].

## File Structure
* `train.py`: Trains the YOLO26s model with automatic hardware detection, early stopping, and dynamic output logging[cite: 17, 18].
* `detect_v2.py`: Runs the trained model weights on an input video (`sample vid.mp4`) and outputs an annotated video file (`detected vid.avi`) with pothole bounding boxes.
* `weights/`: Directory containing the final trained YOLO model weights (`best.pt`) used for inference.
* `.gitignore`: Configuration file that blocks local datasets, heavy training outputs, and video files from being uploaded to GitHub.

## How to Run Inference
1. Clone this repository and install dependencies:
   `pip install ultralytics`
2. Place your test video in the project folder and update the filename inside `detect_v2.py`.
3. Run the detection script:
   `python detect_v2.py`
4. The annotated video will be generated and saved to your specified output directory.