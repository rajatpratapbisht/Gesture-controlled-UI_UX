import cv2
import mediapipe as mp

# MEDIAPIPE
###################################################
# Initialize MediaPipe hands module
# mp_drawing = mp.solutions.mediapipe.python.solution.drawing_utils     # render landmarks
# mp_hands = mp.solutions.mediapipe.python.solutions.hands               # get mediapipe-hands model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# WEBCAM
###################################################
# Start capturing video from the webcam
cam_width = 800
cam_height = 600

cap = cv2.VideoCapture(0)
cap.set(3, cam_width )
cap.set(4, cam_height)

# MAIN LOOP 
###################################################
# initialize model 
with mp_hands.Hands(
    max_num_hands = 2,
    static_image_mode = False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5 ) as hands:   

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Start Detection
        ###################################################
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # set image to be non-modifyable
        rgb_frame.flags.writeable = False
        
        # Process the frame to detect hands
        results = hands.process(rgb_frame)
        
        # set image back to writeable
        rgb_frame.flags.writeable = True
        
        
        # Checkout the results
        ###################################################
        ## 1. Print Handedness
        # if results.multi_handedness:
        #     for hand_info in results.multi_handedness:
        #         handedness = hand_info.classification[0].label
        #         print(f"Handedness: {handedness}")

        ## 2. Print Landmarks
        # print(f"landmarks: \n{results.multi_hand_landmarks}")
        if results.multi_hand_landmarks :
            # # a. Print landmarks
            # for hand_idx, landmark_info in enumerate(results.multi_hand_landmarks):
            #     print(f"Hand: {hand_idx + 1}")
            #     for idx, landmark in enumerate(landmark_info.landmark):
            #         print(f"Landmark {idx} |  \
            #                     x={landmark.x:.4f}, \
            #                     y={landmark.y:.4f}, \
            #                     z={landmark.z:.4f}")
            
            # b. Display landmarks
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # get handedness info (Left/Right hand)
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                    if hand_label == "Right":
                        print(f"Right hand Landmarks:")
                        for i, lm in enumerate(hand_landmarks.landmark):
                            print(f"Landmark{i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:0.3f}")
                            
                # draw landmark one by one
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
        # Display the resulting frame
        cv2.imshow("Hand Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
