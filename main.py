import cv2
import numpy as np
import pygame
import time
from ultralytics import YOLO
from twilio.rest import Client
import keys

# Twilio Setup
client = Client(keys.account_sid, keys.auth_token)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Alarm sound path
path_alarm = "Alarm/alarm.wav"

# Initialize pygame
pygame.init()
pygame.mixer.init()

try:
    pygame.mixer.music.load(path_alarm)
    print("Alarm loaded successfully.")
except pygame.error as e:
    print(f"Alarm load error: {e}")

# Video capture
cap = cv2.VideoCapture("Test Videos/Pencuri.mp4")
frame_skip = 4
frame_counter = 0

# Detection and alert settings
target_classes = ['person']
count = 0
number_of_photos = 3
sms_cooldown = 30  # seconds
last_sms_time = 0
pygame_alarm_playing = False

# Hardcoded polygon (adjust to fit your video frame)
pts = [(100, 100), (540, 100), (540, 300), (100, 300)]

def send_sms_alert():
    try:
        message = client.messages.create(
            body="Alert! Person detected in the monitored area.",
            from_=keys.twilio_number,
            to=keys.my_number
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def inside_polygon(point, polygon):
    if len(polygon) >= 3:
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 360))
    frame_copy = frame.copy()
    person_detected = False

    results = model(frame)

    for r in results:
        for obj in r.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0].tolist())
            label = int(obj.cls)
            name = model.names[label]
            conf = float(obj.conf)

            if name in target_classes:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if inside_polygon((center_x, center_y), pts):
                    print("Person detected inside polygon.")
                    person_detected = True

                    if count < number_of_photos:
                        cv2.imwrite(f"Detected Photos/detected_{count}.jpg", frame_copy)
                        print(f"Saved photo detected_{count}.jpg")
                        count += 1

                    current_time = time.time()
                    if current_time - last_sms_time > sms_cooldown:
                        send_sms_alert()
                        last_sms_time = current_time

    # Play alarm if person detected and alarm not playing
    if person_detected and not pygame_alarm_playing:
        pygame.mixer.music.play(-1)
        print("Alarm started.")
        pygame_alarm_playing = True

    # Stop alarm if no person detected
    if not person_detected and pygame_alarm_playing:
        pygame.mixer.music.stop()
        print("Alarm stopped.")
        pygame_alarm_playing = False

    # Draw polygon
    if len(pts) >= 3:
        cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        pygame.mixer.music.stop()
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()