import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.5, track_confidence=0.5):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        # Initialize Mediapipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=self.max_hands, min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        self.results = self.hands.process(rgb_frame)
        # Draw landmarks if specified
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, hand_number=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, landmark in enumerate(hand.landmark):
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, cx, cy])
        return landmark_list
    def isIndexUp(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            index=hand.landmark[8]
            thumb=hand.landmark[4]
            pinky=hand.landmark[20]
            middle=hand.landmark[12]
            ring=hand.landmark[16]
            if index.y<thumb.y and index.y<pinky.y and index.y<middle.y and index.y<ring.y:
                return True
            else:
                return False
    def isThumbUp(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            index=hand.landmark[8]
            thumb=hand.landmark[4]
            pinky=hand.landmark[20]
            middle=hand.landmark[12]
            ring=hand.landmark[16]
            if thumb.y<index.y and thumb.y<pinky.y and thumb.y<middle.y and thumb.y<ring.y:
                return True
            else:
                return False
    def isMiddleUp(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            index=hand.landmark[8]
            thumb=hand.landmark[4]
            pinky=hand.landmark[20]
            middle=hand.landmark[12]
            ring=hand.landmark[16]
            if middle.y<index.y and middle.y<thumb.y and middle.y<pinky.y and middle.y<ring.y:
                return True
            else:
                return False
    def isRingUp(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            index=hand.landmark[8]
            thumb=hand.landmark[4]
            pinky=hand.landmark[20]
            middle=hand.landmark[12]
            ring=hand.landmark[16]
            if ring.y<index.y and ring.y<thumb.y and ring.y<middle.y and ring.y<pinky.y:
                return True
            else:
                return False
    def isPinkyUp(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            index=hand.landmark[8]
            thumb=hand.landmark[4]
            pinky=hand.landmark[20]
            middle=hand.landmark[12]
            ring=hand.landmark[16]
            if pinky.y<index.y and pinky.y<thumb.y and pinky.y<middle.y and pinky.y<ring.y:
                return True
            else:
                return False
    def isAllDown(self):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            index=hand.landmark[8]
            thumb=hand.landmark[4]
            middle=hand.landmark[12]
            ring=hand.landmark[16]
            pinky=hand.landmark[20]
            if not(self.isIndexUp() or self.isThumbUp() or self.isMiddleUp() or self.isRingUp() or self.isPinkyUp()):
                return True
            else:
                return False

# Example usage:
if __name__ == "__main__":
    # Load the hand detector
    detector = HandDetector(max_hands=2, detection_confidence=0.7)

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

        # Use the hand detector to find hands in the frame
        frame = detector.find_hands(frame)
        landmarks = detector.find_position(frame)

        # If hands are detected, you can extract landmarks and use them for further processing
        if landmarks:
            print(landmarks)  # Print the landmarks to the console

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
