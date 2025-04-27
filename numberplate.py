import cv2
import pytesseract
from twilio.rest import Client
import time

# Set up Twilio client
account_sid = 'ACdf1086f0539977324e2dc952a855f358'
auth_token = '8f626a93d781ad0a515af725abf3d4e6'
client = Client(account_sid, auth_token)

# Your phone number to receive the call
to_phone = '+12792064935'
from_phone = '+919783735904'

# Set up Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change if necessary

# List of no-parking plates (just an example)
no_parking_plates = ['ABC123', 'XYZ789']


# Function to call via Twilio
def make_call():
    call = client.calls.create(
        to=to_phone,
        from_=from_phone,
        url="http://demo.twilio.com/docs/voice.xml"
    )
    print(f"Call SID: {call.sid}")


# Function to detect license plates in live video
def detect_license_plate(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges in the frame
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Only consider rectangles with a certain aspect ratio (like a license plate)
        if 2 < w / h < 6:
            # Extract the region of interest (ROI) from the frame
            plate_roi = frame[y:y + h, x:x + w]
            # Use pytesseract to extract text from the ROI (assumed to be a plate)
            plate_text = pytesseract.image_to_string(plate_roi, config='--psm 8')
            plate_text = plate_text.strip()

            if plate_text:
                print(f"Detected Plate: {plate_text}")
                return plate_text
    return None


# Main function to process the live video stream
def process_live_video():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Change 0 to the appropriate camera ID if needed

    while True:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect license plate in the frame
        plate = detect_license_plate(frame)

        if plate:
            # Check if the detected plate is in the no-parking list
            if plate in no_parking_plates:
                print(f"Car with plate {plate} is parked in a no-parking zone!")
                make_call()  # Call if the car is in a no-parking zone
                time.sleep(10)  # Wait a bit to prevent multiple calls in quick succession

        # Display the frame with the detected license plate (if any)
        cv2.imshow('Live License Plate Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


# Run the live video processing
process_live_video()
