import cv2
import numpy as np
from collections import defaultdict

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")
track_history = defaultdict(lambda: [])

# Open the video file
video_path = "100m.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, save=False)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            #print("track:",track)
            #print("X:",x)
            #print("box:",box)
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #print("points: ",points)

            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            empt = np.array([[0, 0]])

            difference = np.array(points[0] - points[-1])
            
            print(difference)
            
        # Visualize the results on the frame
        cv2.namedWindow('YOLOv11 Tracking', cv2.WINDOW_KEEPRATIO)
        # Display the annotated frame
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        window = cv2.resizeWindow('YOLOv11 Tracking',1240, 700)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()