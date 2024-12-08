# libraries
import torch
import cv2
import mediapipe as mp
# for loading latest checkpoint

import numpy as np
# from utils import find_latest_checkpoint, get_device
from utils import *
from joblib import load

import warnings
# warnings.filterwarnings("ignore")

print("\n\n+---------------------------------------------------------------------------------------------------------------+")        
# find model from the pickle
directory = 'checkpoints/pickles'
ckpt_file = 'model_gesture_recog'
model_ckpt = find_latest_checkpoint(directory, ckpt_file, tag = '1', ext='pkl')
print(f"| Model Checkpoint: {model_ckpt}")

# find scaler from the pickle
directory = 'checkpoints/pickles/scaler'
scaler_file = 'scaler_pickle'
scaler_ckpt = find_latest_checkpoint(directory, scaler_file, tag='1', ext='pkl')
print(f"| Scaler Checkpoint: {scaler_ckpt}")

# get torch device 
device = get_device()
print(f"| Using Device: {device}")
print("+---------------------------------------------------------------------------------------------------------------+")

# load model
model = load_pickle_model(model_ckpt).to(device)
model.eval()
# print(f"{model}")

# load scaler
scaler = load_pickle_scaler(scaler_ckpt)

# Init Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence = 0.7, 
    min_tracking_confidence = 0.7
    )
mp_draw = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, cam_width )
cap.set(4, cam_height)

# for FPS calcualtaion
prev_time = time.time()
# using model with meadiapipe:
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
        
    # flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # draw ROI for gesture rcognition:
    cv2.rectangle(frame, (int(cam_width * MIN_X), int(cam_height * MIN_Y)), 
                  (int(cam_width * MAX_X), int(cam_height * MAX_Y)), 
                              (0, 255, 0), 2)
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            
            if handedness.classification[0].label.lower() == 'right':
                
                # draw landmarks on hand
                # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                
                # only use the model when finger hand is in valid location:> improve throughput of the application
                if valid_position(hand_landmarks):
                    
                    # Convert to NumPy array and flatten
                    input_data = np.array(landmarks).flatten()
                    
                    try:
                        # normalize input data with running means and sd
                        normlaized_ip = scaler.transform(input_data.reshape(1, -1))
                        
                        # create tensor
                        input_tensor = torch.tensor(normlaized_ip, dtype=torch.float32)  # with running means and sd
                        # input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                        
                        # Make predictions with the loaded model
                        with torch.no_grad():
                            output = model(input_tensor.to(device))
                            predicted_class = torch.argmax(output, dim=1).item()
                        
                        # Display Prediction
                        print(f"Predicted Gesture (Right Hand): {finger_labels[predicted_class]}")
                        cv2.putText(img = frame,
                                    text=f'{action_labels[predicted_class]}',
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale= 2,
                                    org = (150, 50,),
                                    color=(0, 255, 0), 
                                    thickness=2)
                        mouse_op(predicted_class, hand_landmarks)
                        # print(find_index_tip(hand_landmarks))
                    except Exception as e:
                        print(f"Error in processing input data: {e}")
    
    # calculate FPS 
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    # print FPS:
    cv2.putText(img=frame,
                text=str(int(fps)),
                org=(20, 50), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale = 3, 
                color=(255, 0, 0), 
                thickness=3)
    # Show the frame
    cv2.imshow("MediaPipe Hands", frame)
    # wait for a milisec and check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):           
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
