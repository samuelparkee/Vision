import os
from gesturedetection_2hands import get_keyboard_labels

KEYBOARD_LABELS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keyboard_labels.txt")
KEYBOARD_LABELS = []
KEYBOARD_LABELS = get_keyboard_labels(KEYBOARD_LABELS_FILE_PATH)

print(KEYBOARD_LABELS)