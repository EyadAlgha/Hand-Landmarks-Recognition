import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import utils
from utils import recognise_gesture
import time

cap = cv2.VideoCapture(0)

# Model Creation
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
model = vision.HandLandmarker.create_from_options(options)

fps = 0
prev_time = 0

# Open Cam and Process
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR image to RGB for the model to process
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    image = np.copy(mp_image.numpy_view())

    result = model.detect(mp_image) # Detect hand landmarks

    if result.hand_landmarks:
        image = utils.draw_hand_landmarks_on_image(image, result)
        recognise_gesture(image, result)

    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # show the processed image

    key = cv2.waitKey(1)
    if (key and 0xFF == ord('q')) or key == 27 or cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1: # Exit on 'q'
        break

cap.release()
cv2.destroyAllWindows()
