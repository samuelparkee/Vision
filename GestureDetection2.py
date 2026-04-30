import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time

import pyautogui
import ctypes
import tkinter as tk

# --- Colors ---
# -- Extra Screen Window Configure --
#BLACK = 0x00000000
BLACK_HEX = "#000000" # Currently Used
#LIGHT_GRAY = 0x00D3D3D3
#LIGHT_GRAY_HEX = "#D3D3D3"

# -- Hand Landmark Changes --
PEACH_COLOR = np.array([180, 229, 255])
ORANGE_COLOR = np.array([0, 128, 255])
YELLOW_COLOR = np.array([0, 204, 255])
PINK_COLOR = np.array([102, 0, 204])

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MY_SCREEN_WIDTH, MY_SCREEN_HEIGHT = pyautogui.size()

CAMERA_WINDOW_TOP_LEFT_POINT_WIDTH = int(MY_SCREEN_WIDTH/2 - CAMERA_WIDTH/2)
CAMERA_WINDOW_TOP_LEFT_POINT_HEIGHT = int(MY_SCREEN_HEIGHT/2 - CAMERA_HEIGHT/2)

latest_result = None
latest_gesture = None

FINGER_CLOSENESS_THRESHOLD = 0.17 # temp number, can change
MAX_FINGER_CLOSENESS_NUMBER = 0.0000000 # used to check how close the numbers get to threshold
finger_position = (0,0)

start_time_ms = time.time_ns() // 1000000

BORDER_VAL = 0.2

screen_window = tk.Tk()
screen_window.title("Gesture Keyboard")
screen_window.geometry("1000x400+50+50")

screen_window.configure(bg=BLACK_HEX)
screen_window.update() # i think this is needed before i change the title bar color for the window

gesture_text_to_speech_enable = 0 # won't implement for now but will do it in the future

model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

def store_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=store_result,
    num_hands=2)

def distance_between_points(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def replace_color(image, old_color, new_color, tolerance=1):
    mask = cv2.inRange(image, old_color-tolerance, old_color+tolerance)
    image[mask > 0] = new_color

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    gestures_list = detection_result.gestures

    global MAX_FINGER_CLOSENESS_NUMBER, FINGER_CLOSENESS_THRESHOLD, latest_gesture

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        gesture = gestures_list[idx][0]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # replacing the color change with function call
        replace_color(annotated_image, PEACH_COLOR, ORANGE_COLOR)
        replace_color(annotated_image, YELLOW_COLOR, PINK_COLOR)

       # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        label = f"{gesture.category_name} ({gesture.score:.2f})"

        if gesture.category_name == "None":
            palm_size = distance_between_points((x_coordinates[0],y_coordinates[0]),(x_coordinates[9],y_coordinates[9]))
            thumb_to_index_distance = distance_between_points((x_coordinates[4],y_coordinates[4]),(x_coordinates[8],y_coordinates[8]))
            normalized_thumb_index = thumb_to_index_distance / palm_size
            thumb_to_middle_distance = distance_between_points((x_coordinates[4], y_coordinates[4]),
                                                              (x_coordinates[12], y_coordinates[12]))
            normalized_thumb_middle = thumb_to_middle_distance / palm_size

            if normalized_thumb_index < FINGER_CLOSENESS_THRESHOLD:
                if normalized_thumb_index > MAX_FINGER_CLOSENESS_NUMBER:
                    MAX_FINGER_CLOSENESS_NUMBER = normalized_thumb_index

                if normalized_thumb_middle < FINGER_CLOSENESS_THRESHOLD:
                    # action is done if previous and current gesture was pinch
                    if latest_gesture == "Pinch":
                        pyautogui.leftClick()

                    latest_gesture = "Pinch"
                    label = f"{latest_gesture} ({gesture.score:.2f}) ({round(normalized_thumb_index, 2)}) ({round(normalized_thumb_middle, 2)})"
                else:
                    latest_gesture = "Okay"
                    label = f"{latest_gesture} ({gesture.score:.2f}) ({round(normalized_thumb_index, 2)})"

                    x_coord_to_move = (x_coordinates[8] - BORDER_VAL) / (1 - 2 * BORDER_VAL)
                    y_coord_to_move = (y_coordinates[8] - BORDER_VAL) / (1 - 2 * BORDER_VAL)
                    x_move = min(max(x_coord_to_move * 1920, 0), 1920)
                    y_move = min(max(y_coord_to_move * 1080, 0), 1080)
                    pyautogui.moveTo(x_move, y_move)

            elif normalized_thumb_middle < FINGER_CLOSENESS_THRESHOLD:
                latest_gesture = "Thumb Middle"
                label = f"{latest_gesture} ({gesture.score:.2f}) ({round(normalized_thumb_middle, 2)})"
            else:
                latest_gesture = gesture.category_name
        else:
            latest_gesture = gesture.category_name

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, label,
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    label_2h = None
    if len(hand_landmarks_list) == 2:
        hand0_x = [lm.x for lm in hand_landmarks_list[0]]
        hand0_y = [lm.y for lm in hand_landmarks_list[0]]
        hand1_x = [lm.x for lm in hand_landmarks_list[1]]
        hand1_y = [lm.y for lm in hand_landmarks_list[1]]

        palm0_size = distance_between_points((hand0_x[0], hand0_y[0]), (hand0_x[9], hand0_y[9]))
        palm1_size = distance_between_points((hand1_x[0], hand1_y[0]), (hand1_x[9], hand1_y[9]))
        avg_palm_size = (palm1_size + palm0_size) / 2

        dist_h0_thumb_h1_index = distance_between_points((hand0_x[4], hand0_y[4]), (hand1_x[8], hand1_y[8]))
        dist_h0_index_h1_thumb = distance_between_points((hand0_x[8], hand0_y[8]), (hand1_x[4], hand1_y[4]))
        norm_h0h1_ti = dist_h0_thumb_h1_index/avg_palm_size
        norm_h0h1_it = dist_h0_index_h1_thumb/avg_palm_size

        if norm_h0h1_ti < FINGER_CLOSENESS_THRESHOLD and norm_h0h1_it < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"Polygon ({round(norm_h0h1_ti, 2)}) ({round(norm_h0h1_it, 2)})"

        elif norm_h0h1_ti < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"H0 Thumb H1 Index ({round(norm_h0h1_ti, 2)})"

        elif norm_h0h1_it < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"H0 Index H1 Thumb ({round(norm_h0h1_it, 2)})"

        if label_2h is not None:
            cv2.putText(annotated_image, label_2h,(10,30), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

# ==================================================================================================================================================================
# created to make the frame appear in the center of my screen (1920x1080)
# (also approximately adjusted for taskbar height)
cv2.namedWindow("frame")
cv2.moveWindow("frame", CAMERA_WINDOW_TOP_LEFT_POINT_WIDTH, CAMERA_WINDOW_TOP_LEFT_POINT_HEIGHT)
#cv2.moveWindow("frame", 640,0)

hwnd = ctypes.windll.user32.FindWindowW(None,"frame")
style = ctypes.windll.user32.GetWindowLongW(hwnd,-16)
ctypes.windll.user32.SetWindowLongW(hwnd,-16,style & ~0x00C00000)
ctypes.windll.user32.SetWindowPos(hwnd,None,CAMERA_WINDOW_TOP_LEFT_POINT_WIDTH,CAMERA_WINDOW_TOP_LEFT_POINT_HEIGHT,CAMERA_WIDTH,CAMERA_HEIGHT,0x0027)
## ==================================================================================================================================================================

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        frame_timestamp_ms = (time.time_ns() // 1000000) - start_time_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        if latest_result is not None:
            marked_image = draw_landmarks_on_image(frame, latest_result)
            cv2.imshow("frame", marked_image)
        else:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) in [ord('q'),ord('Q')]:
            break
cap.release()
cv2.destroyAllWindows()
