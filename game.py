import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from pynput.keyboard import Controller,Key
from handtrackingmodule import HandDetector
import time

keyboard=Controller()
# Function to map the position of hand landmarks to the screen resolution
def map_position(x, y, width, height):
    return int(np.interp(x, (0, width), (0, pyautogui.size()[0]))), int(np.interp(y, (0, height), (0, pyautogui.size()[1])))

# Load the hand detector
detector = HandDetector(max_hands=1, detection_confidence=0.7)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for mouse control
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smooth_factor = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    #make the frame smaller

    # Use the hand detector to find hands in the frame
    frame = detector.find_hands(frame)
    landmarks = detector.find_position(frame)

    # If hands are detected, you can extract landmarks and use them for further processing
    if landmarks:
        # Get the position of the index finger landmark (landmark index 8)
        x, y = landmarks[0][1], landmarks[0][2]
        x_thumb, y_thumb = landmarks[4][1], landmarks[4][2]
        x_index, y_index = landmarks[8][1], landmarks[8][2]
        x_middle, y_middle = landmarks[12][1], landmarks[12][2]
        x_ring, y_ring = landmarks[16][1], landmarks[16][2]
        x_pinky, y_pinky = landmarks[20][1], landmarks[20][2]
        x_end_ring, y_end_ring = landmarks[13][1], landmarks[13][2]
        indexUp=detector.isIndexUp()
        thumbUp=detector.isThumbUp()
        middleUp=detector.isMiddleUp()
        allDown=detector.isAllDown()
        pinkyUp=detector.isPinkyUp()
        ringUp=detector.isRingUp()

        # Smooth the hand movement
        curr_x = prev_x + (x - prev_x) / smooth_factor
        curr_y = prev_y + (y - prev_y) / smooth_factor
        if(indexUp):
            keyboard.release(Key.down)
            keyboard.press(Key.up)
            # keyboard.release(Key.up)
        if indexUp and not thumbUp:
            keyboard.release(Key.left)
        if indexUp and not middleUp:
            keyboard.release(Key.right)
        if(thumbUp and not indexUp and not middleUp and not allDown):
            keyboard.release(Key.right)
            keyboard.press(Key.left)
            # keyboard.release(Key.left)
        if(middleUp and not indexUp and not thumbUp and not allDown):
            keyboard.release(Key.left)
            keyboard.press(Key.right)
        if (pinkyUp):
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.press(Key.down);
        if ringUp:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.release(Key.down)
            keyboard.press('f')
            time.sleep(0.1)
            keyboard.release('f')
        # if indexUp:
        #     keyboard.press(Key.up)
        #     time.sleep(0.1)
        #     keyboard.release(Key.up)
        # if thumbUp:
        #     keyboard.press(Key.left)
        #     time.sleep(0.1)
        #     keyboard.release(Key.left)
        # if middleUp:
        #     keyboard.press(Key.right)
        #     time.sleep(0.1)
        #     keyboard.release(Key.right)
        # if pinkyUp:
        #     keyboard.press(Key.down)
        #     time.sleep(0.1)
        #     keyboard.release(Key.down)
        # if np.linalg.norm(np.array([x_thumb, y_thumb]) - np.array([x_end_ring, y_end_ring])) < 50:
        #     keyboard.release(Key.left)
        #     keyboard.release(Key.right)
        #     keyboard.release(Key.up)
        #     keyboard.press(Key.down)
        #     keyboard.press('f')
        #     time.sleep(0.1)
        #     keyboard.release('f')
        
        # Map the hand position to the screen resolution
        mapped_x, mapped_y = map_position(curr_x, curr_y, frame.shape[1], frame.shape[0])

        # Move the mouse cursor
        pyautogui.moveTo(mapped_x, mapped_y)

        # Update previous positions
        prev_x, prev_y = curr_x, curr_y

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
