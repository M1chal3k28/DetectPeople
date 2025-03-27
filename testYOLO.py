from ultralytics import YOLO
import cv2

# Wczytanie modelu YOLOv8
model = YOLO("yolov8s.pt")  # Możesz użyć yolov8s.pt dla lepszej dokładności

# Uruchomienie kamery
cap = cv2.VideoCapture("http://10.154.231.94:8080/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Wykrywanie obiektów
    results = model(frame)

    # Rysowanie wykrytych obiektów
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            confidence = box.conf[0]

            if label == "person" and confidence > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()