from ultralytics import YOLO
import torch

def main():
    PROJECT_NAME = 'mission2_obstacle_model'
    DATA_CONFIG  = 'dataset_final/data.yaml'
    EPOCHS       = 50    
    IMG_SIZE     = 640   
    BATCH_SIZE   = 8    
    
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
            patience=10,      
            save=True,      
            verbose=True      
        )
        print("\nSUCCESS! Training Complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        print("Tip: If the error is 'CUDA out of memory', change BATCH_SIZE to 8 or 4.")

if __name__ == '__main__':

    main()
