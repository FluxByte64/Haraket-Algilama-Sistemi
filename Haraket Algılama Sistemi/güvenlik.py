import cv2
import time
from playsound import playsound
import numpy as np
from pushbullet import Pushbullet 

pb = Pushbullet("") 


def send_push_notification(title, message):
    push = pb.push_note(title, message)

camera = cv2.VideoCapture(0)
time.sleep(1)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

motion_threshold = 2000 
movement_speed_threshold = 15  
min_area = 1500 

prev_center = None

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    fgmask = fgbg.apply(gray_frame)

    fgmask = cv2.dilate(fgmask, None, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:  
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        movement_area = w * h 
        center = (x + w // 2, y + h // 2)

        if prev_center is None:
            prev_center = center
            continue

        delta_x = center[0] - prev_center[0]
        delta_y = center[1] - prev_center[1]
        movement_speed = np.sqrt(delta_x**2 + delta_y**2)

        if movement_speed > movement_speed_threshold:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            playsound("ireland-eas-alarm-264351.mp3", block=False)

            send_push_notification("UYARI: Hareket Algılandı!", "İzinsiz Giriş Yapıldı!")

        prev_center = center  
    cv2.imshow("Hareket Algılayıcı", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

