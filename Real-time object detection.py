import cv2
from ultralytics import YOLO

# Load YOLOv8 model (you can choose 'yolov8n.pt' for speed or 'yolov8s.pt' for better accuracy)
model = YOLO("yolov8n.pt")  # Downloads automatically if not available

# Check available camera
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        cap.release()

# Open webcam with AVFoundation (macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Draw results (bounding boxes and labels)
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press Esc to quit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()