import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from handtrackingmodule import HandDetector

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
        #get the position of palm landmark (landmark index 0)
        x, y = landmarks[0][1], landmarks[0][2]
        # Get the position of the index finger landmark (landmark index 8)
        x_index, y_index = landmarks[8][1], landmarks[8][2]
        # get the position of thumb landmark (landmark index 4)
        x_thumb, y_thumb = landmarks[4][1], landmarks[4][2]
        #get the positon of the middle finger landmark (landmark index 12)
        x_middle, y_middle = landmarks[12][1], landmarks[12][2]
        x_end_ring, y_end_ring = landmarks[13][1], landmarks[13][2]
        x_ring, y_ring = landmarks[16][1], landmarks[16][2]
        #if the thumb is closer to the index finger, perform a click action
        pyautogui.FAILSAFE = False
        if np.linalg.norm(np.array([x_index, y_index]) - np.array([x_thumb, y_thumb])) < 50:
            pyautogui.click()
        #if the thumb  and little finger are closer, perform a scroll up action
        elif np.linalg.norm(np.array([landmarks[20][1], landmarks[20][2]]) - np.array([landmarks[4][1], landmarks[4][2]])) < 50:
            pyautogui.scroll(-20)
        #if the thumb and ring finger are closer, perform a scroll down action
        elif np.linalg.norm(np.array([landmarks[16][1], landmarks[16][2]]) - np.array([landmarks[4][1], landmarks[4][2]])) < 50:
            pyautogui.scroll(+20)
        elif np.linalg.norm(np.array([x_thumb, y_thumb]) - np.array([x_end_ring, y_end_ring])) < 50:
            pyautogui.rightClick()
        elif np.linalg.norm(np.array([x_thumb, y_thumb]) - np.array([landmarks[5][1],landmarks[5][2]])) < 50:
            pyautogui.mouseDown();

        # Smooth the hand movement
        curr_x = prev_x + (x - prev_x) / smooth_factor
        curr_y = prev_y + (y - prev_y) / smooth_factor

        # Map the hand position to the screen resolution
        mapped_x, mapped_y = map_position(curr_x+40, curr_y-40, frame.shape[1], frame.shape[0])

        # Move the mouse cursor
        pyautogui.moveTo(mapped_x, mapped_y)

        # Update previous positions
        prev_x, prev_y = curr_x, curr_y

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
