from ultralytics import YOLO
import cv2
import numpy as np

# Wczytanie modelu YOLOv8
model = YOLO("yolov8s.pt")

def detect_person():
    # url = "http://10.154.231.94:8080/video"
    # cap = cv2.VideoCapture(url)
    # ret, frame = cap.read()
    # cap.release()

    frame = cv2.imread("test.jpg")
    ret = True

    if not ret:
        return None, {"person_detected": False}

    people_detected = False
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            confidence = box.conf[0]

            if label == "person" and confidence > 0.5:
                people_detected = True

    return frame, {"person_detected": people_detected}

# Test
frame, result = detect_person()
print(result["person_detected"])
if frame is not None:
    cv2.imshow("Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Błąd: Nie udało się pobrać obrazu")
