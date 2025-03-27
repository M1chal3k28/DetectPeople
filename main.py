from flask import Flask, jsonify, Response
import cv2

# Utility to make rout accesible only on debug 
from functools import wraps
from flask import current_app, abort
def debug_only(f):
    @wraps(f)
    def wrapped(**kwargs):
        if not current_app.debug:
            abort(404)

        return f(**kwargs)

    return wrapped

# Import YOLO models
from ultralytics import YOLO

# Create app on Flask
app = Flask(__name__)

# Load nano model of yolo version 8 smallest model for fastest calculations
model = YOLO("yolov8n.pt")

# Function which reads image from camera
def getImg():
    # TODO: Change url to existing and connected web cam on the device ! 
    url = "/"
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()

    # Returns frame from camera to be tested
    return frame if ret else None

# Function that runs model to check if any people is in sight of view
def detect_person():
    # Get image of interest
    image = getImg()
    if image is None:
        return None, {"person_detected": False}

    # Flag to return
    people_detected = False
    
    # Run model
    results = model(image)

    # Check results for people
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            confidence = box.conf[0]

            # If person detected and confidence is greater than half report True
            if label == "person" and confidence > 0.5:
                people_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Person: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, {"Detected": people_detected }

# Returns json with true if person is in sight of view or false if not
@app.route('/detect', methods=['GET'])
def detect():
    _, result = detect_person()
    return jsonify(result)

# Printst image in html (DEGUB ONLY) 
@app.route('/detect_image', methods=['GET'])
@debug_only
def detect_image():
    image, _ = detect_person()
    if image is None:
        return "Error: Can't download image from camera!", 500

    _, buffer = cv2.imencode('.jpg', image)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# Flask main
if __name__ == '__main__':
    app.run(debug=False)
