from ultralytics import YOLO
from pathlib import Path

def detect_potholes_in_video(
    model_path: str = "weights/best.pt", # Path to your best training weights
    video_path: str = "sample vid.mp4",                    # Replace with your actual video filename
    output_dir: str = "output_results"
):
    """
    Loads your trained YOLO model and performs pothole detection on an input video.
    """
    # 1. Verify files exist
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model weights not found at: {model_path}. Check your training outputs!")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found at: {video_path}. Provide a valid video path.")

    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)

    print(f"Processing video: {video_path} ...")
    
    # 2. Run prediction / inference on the video
    results = model.predict(
        source=video_path,
        conf=0.25,          # Lowered confidence threshold to catch tricky/distant potholes
        save=True,          # Save the annotated video result
        project=output_dir, # Output directory folder name
        name="video_run2",   # Sub-folder name inside project
        stream=False       # Stream can be False for standard dashcam clips
    )

    print(f"\nProcessing complete! Annotated video saved inside: {Path(output_dir) / 'video_run'}")

if __name__ == "__main__":
    detect_potholes_in_video(
        video_path = r"C:\Users\sasi3\Desktop\poth hole detection\sample clip 4.webm" # <--- Put your real video file name here!
    )