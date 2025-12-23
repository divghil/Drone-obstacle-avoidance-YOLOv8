from ultralytics import YOLO
import collections
# This line fixes the crash on Python 3.10 and 3.11
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import cv2

# ================= CONFIGURATION =================
# Connect string: Use '127.0.0.1:14550' for SITL (Simulation)
# Use '/dev/ttyACM0' or '/dev/ttyUSB0' for real Raspberry Pi -> Pixhawk
CONNECTION_STRING = '127.0.0.1:14550' 
MODEL_PATH = 'best.pt' # Put your 5.95MB file next to this script
TARGET_SPEED = 1.0     # m/s (Fly slow!)
# =================================================

def send_velocity(vehicle, vx, vy, vz):
    """
    Move vehicle in direction based on velocity vectors.
    vx = Forward(+)/Back(-)
    vy = Right(+)/Left(-)
    vz = Down(+)/Up(-)
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # Frame relative to drone (Body frame)
        0b0000111111000111, # Type mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        vx, vy, vz, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not used)
        0, 0)    # yaw, yaw_rate (not used)
    vehicle.send_mavlink(msg)

def main():
    print(f"Connecting to drone on {CONNECTION_STRING}...")
    try:
        # wait_ready=True ensures we actually get a connection
        vehicle = connect(CONNECTION_STRING, wait_ready=True)
        print("✅ Drone Connected!")
    except:
        print("❌ Could not connect to drone. (Are you running SITL?)")
        print("Running in 'Headless Mode' (Camera Only) for testing...")
        vehicle = None

    print(f"Loading Model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Open Webcam (Change to 0 for laptop webcam, or 'video.mp4')
    cap = cv2.VideoCapture(0)
    
    # Image Center (Target)
    CENTER_X = 640 / 2 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. RUN YOLO
        results = model(frame, verbose=False, conf=0.5)
        
        # 2. DECISION LOGIC
        move_x = TARGET_SPEED # Default: Move Forward
        move_y = 0            # Default: Don't turn
        
        detected = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # We found an object!
                detected = True
                x1, y1, x2, y2 = box.xyxy[0]
                box_center_x = int((x1 + x2) / 2)
                
                # Draw box for visualization
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # AVOIDANCE LOGIC:
                # If object is in the MIDDLE, we must dodge.
                # Threshold: 100 pixels from center
                
                if box_center_x < (CENTER_X - 100):
                    # Object is on the LEFT -> Move RIGHT
                    print("⚠️ Obstacle LEFT -> Dodging RIGHT")
                    move_y = 1.0 # 1 m/s Right
                    
                elif box_center_x > (CENTER_X + 100):
                    # Object is on the RIGHT -> Move LEFT
                    print("⚠️ Obstacle RIGHT -> Dodging LEFT")
                    move_y = -1.0 # 1 m/s Left
                    
                else:
                    # Object is DEAD AHEAD -> STOP or CLIMB
                    # Simple Logic: Stop forward, drift Right
                    print("🚨 Obstacle AHEAD -> STOPPING Forward!")
                    move_x = 0.0  # Stop moving forward
                    move_y = 1.0  # Drift right to bypass
        
        # 3. SEND COMMAND TO DRONE
        if vehicle and vehicle.mode.name == 'GUIDED':
            send_velocity(vehicle, move_x, move_y, 0)
        elif not vehicle:
            # Just print what we WOULD do
            print(f"CMD: Fwd={move_x}, Right={move_y}")

        # Show the camera view
        cv2.imshow("Drone Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()