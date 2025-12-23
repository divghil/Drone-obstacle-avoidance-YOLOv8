# Drone Obstacle Avoidance System 
This project uses a custom-trained YOLOv8 Nano model to detect obstacles (Trees, Poles, Wires, Vehicles) and navigate a drone autonomously using DroneKit.

- **Architecture:** YOLOv8 Nano
- **Trained on:** Custom dataset (~1,200 images) + TTPLA Dataset
- **Accuracy (mAP50):** 74.3%
- **Classes:** `Pole`, `Wire`, `Tree`, `Vehicle`

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
