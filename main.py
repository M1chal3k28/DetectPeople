from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import torch
import os

# Title is mandatory for restart.bat script cause it terminates by name
os.system("title PeopleDetection")

# restarting API
restart = False
def RESTART():
    global restart
    if not restart: 
        print("Restart not enabled !")
        return
    
    # Only for windows run restart script
    os.system("start /min Misc\\Restart.bat")


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
# CORS !!!!
CORS(app)

# Load nano model of yolo version 8 smallest model for fastest calculations
model = YOLO("yolov8n.pt")
# Load model to gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Global camera variable
cap = None

# Function which reads image from camera
def getImg():
    ret, frame = cap.read()
    if not ret:
        # If camera dies during execution start restart script to restart API
        print("Couldn't read frame from camera restarting !")
        RESTART()
        
    # Returns frame from camera to be tested
    return frame if ret else None

# Function that runs model to check if any people is in sight of view
def detect_person():
    # Get image of interest
    image = getImg()
    if image is None:
        return None, {"Detected": False}

    # Flag to return
    people_detected = False
    # Amount
    people_amount = 0
    
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
                people_amount+=1

    return {"Detected": people_detected, "Amount": people_amount}

# Function that runs model to check if any people is in sight of view
def detect_person_image():
    # Get image of interest
    image = getImg()
    if image is None:
        return None, {"Detected": False}

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
    result = detect_person()
    return jsonify(result)

# Printst image in html (DEGUB ONLY) 
@app.route('/detect_image', methods=['GET'])
@debug_only
def detect_image():
    image, _ = detect_person_image()
    if image is None:
        return "Error: Can't download image from camera!", 500

    _, buffer = cv2.imencode('.jpg', image)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# Flask main
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', dest='isDeployed', type=int, help='Run in deployment/debug mode (0 - debug, 1 - deployment)')
    parser.add_argument('--restart', dest='restartEnable', type=int, help="1 - API will try to recover from error warning, 0 - omit errors and warnings")
    args = parser.parse_args()

    if args.restartEnable == 1:
        # if restart is enabled change global flag
        global restart
        restart = True
    
    # Set Globally camera source
    global cap
    # Load first available camera device
    # ! May not work if you have more than one camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Couldn't load camera restarting !")
        RESTART()

    if (args.isDeployed) :
        print("People detection is running in deployment mode")
        from waitress import serve
        # Run deploy on localhost
        serve(app, host="127.0.0.1", port=8080)
    else :
        app.run(debug=True)

    cap.release()
    cv2.destroyAllWindows()

    global model
    del model

if __name__ == '__main__':
    main()
