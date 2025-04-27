import cv2
import numpy as np
from twilio.rest import Client
import firebase_admin
from firebase_admin import credentials, firestore
import time
import datetime

# Twilio Client
account_sid = 'ACdf1086f0539977324e2dc952a855f358'
auth_token = '8f626a93d781ad0a515af725abf3d4e6'
twilio_client = Client(account_sid, auth_token)
twilio_number = '+12792064935'
alert_recipient_number = '+919783735904'

# Firebase Initialization
cred = credentials.Certificate(r'C:\Users\theda\Documents\phantompulse-firebase-adminsdk-fbsvc-875e865453.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load YOLO Model for Bottle Detection
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Video Capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(0)

# Last call timestamp
last_call_time = 0  # Initially, no call has been made
CALL_DELAY = 30 * 60  # 30 minutes in seconds


# Function to make an automated call alert
def make_call_alert():
    global last_call_time
    current_time = time.time()

    # Check if 30 minutes have passed since the last call
    if current_time - last_call_time >= CALL_DELAY:
        call = twilio_client.calls.create(
            twiml='<Response><Say>trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. trash detected in area 1! Please clean the area. </Say></Response>',
            from_=twilio_number,
            to=alert_recipient_number
        )
        last_call_time = current_time  # Update last call time
        print(f"Call initiated to {alert_recipient_number}")
    else:
        print("Call skipped due to delay limit.")


# Function to log bottle detection in Firebase
def log_bottle_complaint(location, timestamp):
    doc_ref = db.collection('bottle_detections').document()
    doc_ref.set({
        'location': location,
        'timestamp': timestamp,
        'status': 'Pending'
    })
    print(f"Bottle complaint logged in Firebase: {location} at {timestamp}")


# Main loop for bottle detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Detect only bottles (Class ID 39 in COCO dataset)
            if confidence > 0.5 and classes[class_id] == "bottle":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = "Bottle"
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Alert & Logging
            print(f"Trash detected with {confidence:.2f} confidence.")

            # Make a call alert (only if 30 min delay is met)
            make_call_alert()

            # Log to Firebase
            location = "Detected TRASH at Area 1`"
            timestamp = datetime.datetime.now().isoformat()
            log_bottle_complaint(location, timestamp)

    cv2.imshow('Bottle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
