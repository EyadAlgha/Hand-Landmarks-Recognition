import threading
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import pyautogui
from pynput.mouse import Button, Controller

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

SCREEN_HEIGHT = 720
SCREEN_WIDTH = 1280

mouse = Controller()
last_left_click_time = 0

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_pose_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def calculate_angle(a, b, c):
  return np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))

def calculate_distance(landmarks: [NormalizedLandmark]):
  if len(landmarks) < 2:
    return

  p1 = np.array([landmarks[0].x, landmarks[0].y]) #x, y pair
  p2 = np.array([landmarks[1].x, landmarks[1].y]) #x, y pair

  return np.linalg.norm(p1 - p2)

def find_finger_landmark(result, id):
  if result.hand_landmarks:
    hand_landmarks_list = result.hand_landmarks
    return hand_landmarks_list[0][id]

  return None

def draw_rect(img, x, y, color):
  # x, y are given normalized

  x_norm, y_norm = (x, y)
  width_norm, height_norm = 0.03, 0.03  # width and height of normalized rectangle

  x = int(x_norm * SCREEN_WIDTH)
  y = int(y_norm * SCREEN_HEIGHT)
  width = int(width_norm * SCREEN_WIDTH)
  height = int(height_norm * SCREEN_HEIGHT)

  start_point = (x - width // 2, y - height // 2)  # Top-left corner
  end_point = (x + width // 2, y + height // 2)  # Bottom-right corner
  thickness = 2

  cv2.rectangle(img, start_point, end_point, color, thickness)

def move_mouse(x, y):
  current_x, current_y = pyautogui.position()

  x_diff = x - current_x
  y_diff = y - current_y

  pyautogui.moveRel(x_diff, y_diff, duration=0.1)

def perform_left_click():
  global last_left_click_time
  current_time = time.time()

  if current_time - last_left_click_time > 0.5:
    mouse.press(Button.left)
    mouse.release(Button.left)
    last_left_click_time = current_time

def recognise_gesture(img, result):

  # Skip gesture recognition if ANY landmark is outside the frame.
  #for hand_landmarks in result.hand_landmarks:
  #  for landmark in hand_landmarks:
  #    if not (0 <= landmark.x < 1 and 0 <= landmark.y < 1):
  #      #print("Landmark out of bounds, skipping gesture recognition.")
  #      cv2.putText(img, f"Skipping", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  #      return

  idx_finger_tip = find_finger_landmark(result, 8) # 8 is the index fingertip
  t1 = find_finger_landmark(result, 4)
  t2 = find_finger_landmark(result, 11)
  thumb_idx_dist = calculate_distance([t1, t2])
  thumb_idx_angle = calculate_angle([find_finger_landmark(result, 5).x, find_finger_landmark(result, 5).y],
                                    [find_finger_landmark(result, 6).x, find_finger_landmark(result, 6).y],
                                    [find_finger_landmark(result, 8).x, find_finger_landmark(result, 8).y])

  middle_fing_angle = calculate_angle([find_finger_landmark(result, 9).x, find_finger_landmark(result,9).y],
                                      [find_finger_landmark(result, 10).x, find_finger_landmark(result, 10).y],
                                      [find_finger_landmark(result, 12).x, find_finger_landmark(result, 12).y])

  draw_rect(img, idx_finger_tip.x, idx_finger_tip.y, (255, 0, 0)) # R
  draw_rect(img, t1.x, t1.y, (0, 255, 0)) # G
  draw_rect(img, t2.x, t2.y, (0, 0, 255)) # B
  #print(round(thumb_idx_dist, 3), thumb_idx_angle)
  #print(middle_fing_angle)
  #print(f'thumb angle {round(thumb_idx_angle, 3)} middle angle {round(middle_fing_angle, 3)} thumb dist {round(thumb_idx_dist, 3)}')

  action = 'None'

  # Mouse Movement
  if thumb_idx_dist < 0.045 and thumb_idx_angle < -160:
    x = int(idx_finger_tip.x * SCREEN_WIDTH) # Increase x-axis movement
    y = int(idx_finger_tip.y * SCREEN_HEIGHT)
    threading.Thread(target = move_mouse, args = (x, y)).start()
    action = 'Movement'

  elif thumb_idx_angle < 55 and middle_fing_angle < 30 and thumb_idx_dist < 0.045:
    perform_left_click()
    action = 'Left Click'

  cv2.putText(img, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)





