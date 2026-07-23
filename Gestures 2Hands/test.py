import os
import tkinter as tk
from gesturedetection_2hands_withmain import get_keyboard_labels
from gesturedetection_2hands_withmain import do_keyboard_placement

'''
#KEYBOARD_LABELS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keyboard_labels.txt")
KEYBOARD_LABELS = []
KEYBOARD_LABELS = get_keyboard_labels()


print(KEYBOARD_LABELS)
print(KEYBOARD_LABELS[1][1])
'''


BLACK_HEX = "#000000" # Currently Used

# ------------------ Keyboard Size ------------------
KEYBOARD_ROWS = 6
KEYBOARD_COLUMNS = 15


screen_window = tk.Tk()
screen_window.title("Gesture Keyboard")
screen_window.geometry("1000x400+50+50")

screen_window.attributes('-alpha', 0.7)
screen_window.configure(bg=BLACK_HEX)

keyboard_labels = get_keyboard_labels()

for i in range (KEYBOARD_ROWS):
    screen_window.grid_rowconfigure(i, weight=1)
for i in range (KEYBOARD_COLUMNS):
    screen_window.grid_columnconfigure(i, weight=1)

do_keyboard_placement(screen_window, keyboard_labels)

screen_window.update()  # i think this is needed before i change the title bar color for the window
screen_window.mainloop()