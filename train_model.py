from ultralytics import YOLO
import torch

def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    PROJECT_NAME = 'mission2_obstacle_model'
    DATA_CONFIG  = 'dataset_final/data.yaml'
    EPOCHS       = 50    # Number of times to review the data
    IMG_SIZE     = 640   # Standard resolution for YOLO
    BATCH_SIZE   = 8    # Reduce to 8 or 4 if you get "Out of Memory" errors
    
    # Check if GPU is available 
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on device: {device}")

    model = YOLO('yolov8n.pt') 

    print("Training started... this may take 30-60 minutes.")
    
    try:
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            name=PROJECT_NAME,
            patience=10,      # Stop early if accuracy stops improving (saves time)
            save=True,        # Save the best model automatically
            verbose=True      # Show progress details
        )
        print("\nSUCCESS! Training Complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print("Tip: If the error is 'CUDA out of memory', change BATCH_SIZE to 8 or 4.")

if __name__ == '__main__':
    main()