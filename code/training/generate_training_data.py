'''
Generate training data using mediapipe hands
Author: Rajat Bisht
Version: 1.0
'''
import cv2
import mediapipe as mp
import csv
import time
import os

# Init Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence = 0.5, 
    min_tracking_confidence = 0.5)
mp_draw = mp.solutions.drawing_utils

# Output file
os.makedirs("data", exist_ok=True)
output_file = "data/gesture_data.csv"

# defining labels for gestures
labels ={
    0: "nothing_detected",
    1: "index_finger_up",
    2: "index_and_thumb_up",
    3: "index_and_middle_spaced",
    4: "index_and_middle_together",
    5: "index_thumb_middle_spaced",
    6: "index_thumb_middle_together" 
}


# start video capture
cap = cv2.VideoCapture(0)

print("+----------------------------------------------------")
print("| Press 'q'  to QUIT.")
print("| press '0-6' to label genstures while recording.")
print("+----------------------------------------------------")

mode = 'a'
# mode = 'w'

with open(output_file, mode=mode, newline='') as f:
    writer = csv.writer(f)
    # write header to the file
    if mode == 'w':
        header = ['label'] + [f'{i}_{coord}' for i in range(20) for coord in ('x', 'y', 'z')]
        writer.writerow(header)
        
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab camera frame.\nEXITING . . .")
            break
        
        # flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        result = hands.process(rgb_frame)
        
        # parse landmark data
        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = result.multi_handedness[idx].classification[0].label
                
                # Process only the right hand
                if handedness == "Right":
                    # draw landmarks on the frame
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # extract landmark coordinates
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # show the frame
                    cv2.imshow("Hand Gesture Recirder", frame)
                    
                    # wait for user-input to label gesture
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key in map(ord, '0123456'):        # '0' to '6'
                        label = int(chr(key))
                        print(f"Recording data for: {labels[label]}")
                        writer.writerow([label] + landmarks)
        
        # exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"Data saved to {output_file}")
                        
                            