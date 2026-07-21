import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time

import pyautogui
import ctypes
from ctypes import wintypes
import tkinter as tk

from dataclasses import dataclass

import os

# ================================= CONSTANTS/VARIABLES =================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYBOARD_LABELS_FILE_PATH = os.path.join(SCRIPT_DIR, "keyboard_labels.txt")

# for Keyboard Gesture
FINGER_TIP_JOINT = {
    "index": (8,6),
    "middle": (12,10),
    "ring": (16,14),
    "pinky": (20,18)
}

# ------------------ CAMERA ------------------
# ------ CAMERA DISPLAY SIZE ------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ------ MOUSE POSITION BORDER ------
MARGIN = 10  # pixels
BORDER_VAL = 0.2

# ------ FINGER CLOSENESS & POSITION ------
FINGER_CLOSENESS_THRESHOLD = 0.17 # temp number, can change
MAX_FINGER_CLOSENESS_NUMBER = 0.0000000 # used to check how close the numbers get to threshold
finger_position = (0,0)

# ------ TEXT DISPLAY ------
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# ------------------ Colors ------------------
# ------ Extra Screen Window Configure ------
#BLACK = 0x00000000
BLACK_HEX = "#000000" # Currently Used
CHARCOAL = 0X00252525
#CHARCOAL_HEX = #252525
#LIGHT_GRAY = 0x00D3D3D3
#LIGHT_GRAY_HEX = "#D3D3D3"

# ------ Hand Landmark Changes ------
PEACH_COLOR = np.array([180, 229, 255])
ORANGE_COLOR = np.array([0, 128, 255])
YELLOW_COLOR = np.array([0, 204, 255])
PINK_COLOR = np.array([102, 0, 204])

# ------------------ GLOBALS ------------------
# ------ VARIABLES ------
@dataclass
class GlobalVariables:
    latest_result: object = None
    latest_gesture: str = None
    keyboard_window_open: bool = False
    gesture_text_to_speech_enable: bool = False # won't implement for now but will do it in the future
    
    # Screen Size
    screen_width: int = None
    screen_height: int = None

gbv = GlobalVariables()

# ------------------ MEDIAPIPE SETUP ------------------
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# ================================= functions =================================
# ------------------ FINGER CHECKING ------------------
def is_finger_up (y_coords, tip_idx, joint_idx):
    # less than because bigger number = lower on monitor
    return y_coords[tip_idx] < y_coords[joint_idx]

def is_finger_down (y_coords, tip_idx, joint_idx):
    return not is_finger_up(y_coords,tip_idx,joint_idx)

def fingers_curled_except_index(y_coordinates):
    # this return all checks until the first False then return false if found, True if not
    # return any would check until first True then return true if found, False if not
    return all(
        is_finger_down(y_coordinates,tip,joint)
        for name, (tip,joint) in FINGER_TIP_JOINT.items()
        if name != "index"
    )

# ------------------ KEYBOARD ------------------
#KEYBOARD_LABELS = []
def get_keyboard_labels(filepath):
    key_labels = []
    with open(filepath) as f:
        key_labels = [line.split() for line in f if line.strip()]
    return key_labels

# ------------------ COLOR CHANGE ------------------
def replace_color(image, old_color, new_color, tolerance=1):
    mask = cv2.inRange(image, old_color-tolerance, old_color+tolerance)
    image[mask > 0] = new_color

# ------------------ GESTURES ------------------
def store_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gbv.latest_result = result

def distance_between_points(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    gestures_list = detection_result.gestures

    global MAX_FINGER_CLOSENESS_NUMBER, FINGER_CLOSENESS_THRESHOLD

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

        if gesture.category_name == "Pointing_Up":
            x_coord_to_move = (x_coordinates[8] - BORDER_VAL) / (1 - 2 * BORDER_VAL)
            y_coord_to_move = (y_coordinates[8] - BORDER_VAL) / (1 - 2 * BORDER_VAL)
            x_move = min(max(x_coord_to_move * gbv.screen_width, 1), gbv.screen_width - 1)
            y_move = min(max(y_coord_to_move * gbv.screen_height, 1), gbv.screen_height - 1)
            pyautogui.moveTo(x_move, y_move, _pause=False)

        elif gesture.category_name == "None":
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
                    if gbv.latest_gesture == "Pinch":
                        print("Pinched")
                        # ================================== temporary until i decide what to do here

                    gbv.latest_gesture = "Pinch"
                    label = f"{gbv.latest_gesture} ({gesture.score:.2f}) ({round(normalized_thumb_index, 2)}) ({round(normalized_thumb_middle, 2)})"
                else:
                    # Okay gesture (index and thumb touching) = Left Click
                    gbv.latest_gesture = "Okay"
                    label = f"{gbv.latest_gesture} ({gesture.score:.2f}) ({round(normalized_thumb_index, 2)})"
                    gbv.latest_gesture = "Okay"
                    pyautogui.leftClick()

            elif normalized_thumb_middle < FINGER_CLOSENESS_THRESHOLD:
                 # Thumb + Middle finger touching = right click
                gbv.latest_gesture = "Thumb Middle"
                gbv.latest_gesture = "Thumb Middle"
                label = f"{gbv.latest_gesture} ({gesture.score:.2f}) ({round(normalized_thumb_middle, 2)})"
                pyautogui.rightClick()
            else:
                gbv.latest_gesture = gesture.category_name
        else:
            gbv.latest_gesture = gesture.category_name

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
        dist_h0_thumb_h1_thumb = distance_between_points((hand0_x[4], hand0_y[4]), (hand1_x[4], hand1_y[4]))
        dist_h0_index_h1_index = distance_between_points((hand0_x[8], hand0_y[8]), (hand1_x[8], hand1_y[8]))
        norm_h0h1_thumb = dist_h0_thumb_h1_thumb/avg_palm_size
        norm_h0h1_index = dist_h0_index_h1_index/avg_palm_size

        if norm_h0h1_ti < FINGER_CLOSENESS_THRESHOLD and norm_h0h1_it < FINGER_CLOSENESS_THRESHOLD:
            # only if thumb and index fingers of both hands touch
            label_2h = f"Polygon ({round(norm_h0h1_ti, 2)}) ({round(norm_h0h1_it, 2)})"

        elif norm_h0h1_ti < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"H0 Thumb H1 Index ({round(norm_h0h1_ti, 2)})"

        elif norm_h0h1_it < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"H0 Index H1 Thumb ({round(norm_h0h1_it, 2)})"

        elif norm_h0h1_thumb < FINGER_CLOSENESS_THRESHOLD and norm_h0h1_index < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"Diamond ({round(norm_h0h1_thumb, 2)}) ({round(norm_h0h1_index, 2)})"

        elif norm_h0h1_thumb < FINGER_CLOSENESS_THRESHOLD:
            # in here means thumbs touch
            index_up_both_hands = is_finger_up(hand0_y,8,5) and is_finger_up(hand1_y,8,5)
            other_fingers_curled_both_hands = fingers_curled_except_index(hand0_y) and fingers_curled_except_index(hand1_y)
            if index_up_both_hands and other_fingers_curled_both_hands:
                label_2h = f"Keyboard"
                if not gbv.keyboard_window_open:
                    screen_window.deiconify()
                    gbv.keyboard_window_open = True
            elif not index_up_both_hands and other_fingers_curled_both_hands:
                label_2h = f"Close Keyboard"
                if gbv.keyboard_window_open:
                    screen_window.withdraw()
                    gbv.keyboard_window_open = False
            else:
                label_2h = f"Touching Thumbs ({round(norm_h0h1_thumb, 2)})"

        elif norm_h0h1_index < FINGER_CLOSENESS_THRESHOLD:
            label_2h = f"Touching Index Fingers ({round(norm_h0h1_index, 2)})"

        if label_2h is not None:
            cv2.putText(annotated_image, label_2h,(10,30), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

if __name__ == "__main__":
    start_time_ms = time.time_ns() // 1000000

    gbv.screen_width, gbv.screen_height = pyautogui.size()
    print(gbv.screen_width, gbv.screen_height)

    CAMERA_WINDOW_TOP_LEFT_POINT_WIDTH = int(gbv.screen_width / 2 - CAMERA_WIDTH / 2)
    CAMERA_WINDOW_TOP_LEFT_POINT_HEIGHT = int(gbv.screen_height / 2 - CAMERA_HEIGHT / 2)

    screen_window = tk.Tk()
    screen_window.title("Gesture Keyboard")
    screen_window.geometry("1000x400+50+50")

    screen_window.attributes('-alpha', 0.7)
    screen_window.configure(bg=BLACK_HEX)

    screen_window.update()  # i think this is needed before i change the title bar color for the window
    screen_window.withdraw()  # hides screen_window right after update

    screen_window_id = ctypes.windll.user32.GetParent(screen_window.winfo_id())
    window_title_bar_id = 35
    screen_window_title_color = wintypes.DWORD(CHARCOAL)
    ctypes.windll.dwmapi.DwmSetWindowAttribute(screen_window_id, window_title_bar_id,
                                               ctypes.byref(screen_window_title_color),
                                               ctypes.sizeof(screen_window_title_color))

    model_path = 'gesture_recognizer.task'

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=store_result,
        num_hands=2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # ==================================================================================================================================================================
    cv2.namedWindow("frame")
    cv2.moveWindow("frame", CAMERA_WINDOW_TOP_LEFT_POINT_WIDTH, CAMERA_WINDOW_TOP_LEFT_POINT_HEIGHT)
    # cv2.moveWindow("frame", 640,0)

    hwnd = ctypes.windll.user32.FindWindowW(None, "frame")
    style = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
    ctypes.windll.user32.SetWindowLongW(hwnd, -16, style & ~0x00C00000)
    ctypes.windll.user32.SetWindowPos(hwnd, None, CAMERA_WINDOW_TOP_LEFT_POINT_WIDTH,
                                      CAMERA_WINDOW_TOP_LEFT_POINT_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, 0x0027)
    ## ==================================================================================================================================================================
    try:
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
                if gbv.latest_result is not None:
                    marked_image = draw_landmarks_on_image(frame, gbv.latest_result)
                    cv2.imshow("frame", marked_image)
                else:
                    cv2.imshow('frame', frame)

                screen_window.update_idletasks()
                screen_window.update()

                if cv2.waitKey(1) in [ord('q'), ord('Q')]:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        screen_window.destroy()