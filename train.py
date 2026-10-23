import time
import torch
from pathlib import Path
from ultralytics import YOLO


def train_pothole_detector(
    # =========================================================================
    # 1. CENTRALIZED CONFIGURATION (Change your parameters ONLY here!)
    # =========================================================================
    yaml_path: str = r"C:\Users\sasi3\Desktop\poth hole detection\dataset1\PothHole_dataset_1\dataset.yaml",  # Path to your dataset YAML file
    model_weights: str = "yolo26s.pt",                   # Pretrained YOLO weights (yolo26n.pt or yolo11n.pt)
    epochs: int = 150,                                    # Maximum number of training epochs
    patience: int = 10,                                  # Early stopping: epochs to wait with no improvement
    imgsz: int = 640,                                    # Input image resolution (640x640 is standard)
    batch_size: int = 16,                                # Batch size (reduce to 8 or 4 if GPU runs out of memory)
    workers: int = 4,                                    # Number of CPU threads for loading data
    save_results: bool = True,                           # Save model checkpoints (best.pt & last.pt)
    save_plots: bool = True,                             # Save graphs, confusion matrices, and val batch images
    save_period: int = -1,                               # Save backup weights every X epochs (-1 disables this)
    output_project: str = r"C:\Users\sasi3\Desktop\poth hole detection\training_outputs_1",            # Main root folder for saving all training outputs
    run_name: str = "pothole_yolo26_run3"                     # Subfolder name for this specific training run
):
    """
    Trains an Ultralytics YOLO model on the custom pothole dataset.
    Includes automatic GPU/CPU hardware detection, dynamic parameter logging,
    early stopping, checkpoint saving, and execution time tracking.
    """
    print("=" * 65)
    print("        Starting Pothole Detection Model Training Pipeline")
    print("=" * 65)

    # =========================================================================
    # 2. AUTOMATIC HARDWARE DETECTION (Laptop GPU vs CPU)
    # =========================================================================
    if torch.cuda.is_available():
        device_id = 0  # Use primary NVIDIA GPU
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Hardware Detection] NVIDIA GPU detected : {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"[Hardware Detection] Compute Device      : GPU (device={device_id})")
    else:
        device_id = "cpu"
        print("[Hardware Detection] Warning: No CUDA-compatible GPU detected.")
        print("[Hardware Detection] Compute Device      : CPU (Training will be slower)")

    # =========================================================================
    # 3. DATASET CONFIGURATION VERIFICATION
    # =========================================================================
    dataset_yaml = Path(yaml_path)
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"\n[Error] Dataset YAML not found at: {dataset_yaml.resolve()}\n"
            f"Please ensure '{yaml_path}' exists and your dataset script finished running."
        )

    # =========================================================================
    # 4. INITIALIZE YOLO MODEL
    # =========================================================================
    print(f"\n[Model Setup] Loading pretrained model weights ({model_weights})...")
    # Ultralytics automatically downloads the weight file if it is not in your folder
    model = YOLO(model_weights)

    # =========================================================================
    # 5. DYNAMIC PARAMETER LOGGING (Automatically reflects your changes!)
    # =========================================================================
    save_dir = Path(output_project) / run_name
    print("\n[Training Setup] Launching training with dynamic parameters:")
    print(f" - Dataset Config : {dataset_yaml}")
    print(f" - Base Model     : {model_weights}")
    print(f" - Max Epochs     : {epochs}")
    print(f" - Early Stopping : patience={patience} epochs")
    print(f" - Image Size     : {imgsz}x{imgsz}")
    print(f" - Batch Size     : {batch_size}")
    print(f" - Workers        : {workers}")
    print(f" - Save Results   : {save_results} (Plots={save_plots})")
    print(f" - Output Folder  : {save_dir}/\n")

    # =========================================================================
    # 6. EXECUTE TRAINING & TRACK TIME
    # =========================================================================
    start_time = time.time()

    try:
        results = model.train(
            data=str(dataset_yaml),     # Path to dataset.yaml
            epochs=epochs,              # Dynamically uses 'epochs' parameter
            patience=patience,          # Dynamically uses 'patience' parameter for early stopping
            imgsz=imgsz,                # Dynamically uses 'imgsz' parameter
            batch=batch_size,           # Dynamically uses 'batch_size' parameter
            device=device_id,           # Auto-detected GPU (0) or 'cpu'
            workers=workers,            # Data loading worker threads
            save=save_results,          # Save training checkpoints (best.pt, last.pt)
            save_period=save_period,    # Periodic epoch checkpoint saving
            plots=save_plots,           # Generate loss curves, PR curves, and confusion matrix
            project=output_project,     # Root output directory
            name=run_name,              # Specific experiment folder name
            exist_ok=True,              # Reuse output folder cleanly without creating duplicates
            verbose=True                # Print detailed per-epoch metrics table in terminal
        )
    except Exception as e:
        print(f"\n[Error] Training interrupted due to an error: {e}")
        return

    # Calculate total training duration in minutes
    elapsed_minutes = (time.time() - start_time) / 60

    # =========================================================================
    # 7. FINAL TRAINING SUMMARY
    # =========================================================================
    print("\n" + "=" * 65)
    print("                 Training Completed Successfully!")
    print("=" * 65)
    print(f" - Total Time Taken    : {elapsed_minutes:.2f} minutes")
    print(f" - All Outputs Saved To: {save_dir.resolve()}")
    print(f" - Best Model Weights  : {(save_dir / 'weights' / 'best.pt').resolve()}")
    print(f" - Last Model Weights  : {(save_dir / 'weights' / 'last.pt').resolve()}")
    print(f" - Training Graphs/CSV : {(save_dir / 'results.png').resolve()}")
    print("=" * 65)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Calling with empty parentheses () uses all the default settings defined at the top!
    # You only ever need to edit lines 11-22 at the top of the file.
    train_pothole_detector()