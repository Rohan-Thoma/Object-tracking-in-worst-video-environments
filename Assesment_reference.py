import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects
import argparse

# Dictionary to store previous positions and motion states
tracked_objects_history = {}


def main(video_path, model_path):

    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)

    # Initialize Norfair tracker
    tracker = Tracker(
        distance_function="mean_euclidean",
        distance_threshold=50,
        hit_counter_max=60,
    )

    # Motion detection parameters
    MOTION_THRESHOLD = 5  # Minimum pixel movement to consider as motion
    MOTION_HISTORY = 5    # Number of frames to consider for motion detection
    track_history = {}    # Stores tracking history for each ID

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1440, 1080))
        results = model(frame)
        
        # Prepare detections and boxes
        detections = []
        class_0_boxes = []  # Stores class 0 detections
        class_32_centers = []  # Stores class 32 centers for tracking
        
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                if cls == 0:  # Store class 0 boxes normally
                    x1, y1, x2, y2 = box
                    class_0_boxes.append((x1, y1, x2, y2, conf))
                
                elif cls == 32:  # Only track class 32
                    x1, y1, x2, y2 = box
                    center = np.array([(x1 + x2)/2, (y1 + y2)/2])
                    detections.append(Detection(points=center, scores=np.array([conf]), label=int(cls)))
                    class_32_centers.append(center)

        # Update tracker with class 32 detections
        tracked_objects = tracker.update(detections=detections)
        
        # Start with clean frame
        annotated_frame = frame.copy()
        
        # Draw class 0 boxes normally
        for box in class_0_boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(annotated_frame,(int(x1), int(y1)),(int(x2), int(y2)),(255, 0, 0),2)

            label = f"{model.names[0]} {conf:.2f}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame,(int(x1), int(y1) - h - 10),(int(x1) + w, int(y1)),(255, 0, 0),-1)
            cv2.putText(annotated_frame,label,(int(x1), int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),1)

        # Draw only tracking IDs and action status for class 32 (no boxes)
        for obj in tracked_objects:
            obj_id = obj.id
            center_x, center_y = obj.estimate[0]
            
            # Update tracking history for motion detection
            if obj_id not in track_history:
                track_history[obj_id] = []
            
            track_history[obj_id].append(obj.estimate[0])
            if len(track_history[obj_id]) > MOTION_HISTORY:
                track_history[obj_id].pop(0)
            
            # Determine motion state
            is_moving = False
            if len(track_history[obj_id]) > 1:
                total_movement = sum(
                    np.linalg.norm(track_history[obj_id][i] - track_history[obj_id][i-1])
                    for i in range(1, len(track_history[obj_id]))
                )
                is_moving = total_movement > MOTION_THRESHOLD
            
            # Draw tracking ID and action status
            action_text = "ACTION" if is_moving else "STATIONARY"
            label = f"ID:{obj_id} {action_text}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw center point
            cv2.circle(annotated_frame,(int(center_x), int(center_y)),5,(0, 0, 255) if is_moving else (0, 255, 0),-1 )
            
            cv2.putText(annotated_frame,label,(int(center_x+20), int(center_y)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

        cv2.imshow("Class 0 (Boxes) + Class 32 (Tracking Only)", annotated_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Object Tracking with YOLOv8 and Norfair')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolo11l.pt', help='Path to YOLO model file')
    
    args = parser.parse_args()
    
    # Run main function with provided arguments
    main(args.video, args.model)












