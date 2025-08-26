from ultralytics import YOLO
import cv2
import time

# Load trained YOLOv8 model
model = YOLO("best.pt")  # adjust path if needed

# Open webcam (0) or video file (replace with path)
cap = cv2.VideoCapture(0)  # use "conveyor.mp4" for a conveyor simulation

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # Run detection
    results = model(frame, conf=0.3, verbose=False)

    # Annotated frame
    annotated = results[0].plot()

    # Object counts
    object_counts = {}

    # Draw pick points
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Draw crosshair for pick point
        cv2.drawMarker(
            annotated, (cx, cy), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2
        )

        # Count detected objects
        cls_name = model.names[int(cls)]
        object_counts[cls_name] = object_counts.get(cls_name, 0) + 1

    # FPS calculation
    fps = 1.0 / (time.time() - start)

    # Dashboard overlay
    y0 = 30
    for cls_name, count in object_counts.items():
        cv2.putText(annotated, f"{cls_name}: {count}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y0 += 30
    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Scrap Classifier Simulation", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
