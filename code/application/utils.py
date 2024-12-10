# import libraries
import glob
import os
import time
import torch
from training.ffn import GestureNet
from sklearn.preprocessing import StandardScaler
import pickle
import pyautogui
from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as KeyboardController, Key
import numpy as np


mouse  = Controller()
keyboard = KeyboardController()

screen_width, screen_height = pyautogui.size()        #  Size(width=1680, height=1050)
cam_width, cam_height = 640, 480
smoothening = 7             # higher value, less poiner speed
prev_index_x, prev_index_y = 0, 0
curr_index_x, curr_index_y = 0, 0
button_delay = 10           
button_counter = 0
scroll_sensitivity = 2    # higher value, more sensitive
scroll_threshold = 10

finger_labels ={
    0: "nothing_detected",
    1: "index_finger_up",
    2: "index_and_thumb_up",
    3: "index_and_middle_spaced",
    4: "index_and_middle_together",
    5: "index_thumb_middle_spaced",
    6: "index_thumb_middle_together" 
}

action_labels = {
    0: "S",                    # nothing_detected
    1: "track_mouse_pointer",           # index_finger_up
    2: "execute_left_click",            # index_and_thumb_up
    3: "execute_double_click",          # index_and_middle_spaced
    4: "scrolling_window",              # index_and_middle_together
    5: "execute_right_click",           # index_thumb_middle_spaced
    6: "scroll_switch_screen"           # index_thumb_middle_together
}

hand_dict = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20
}



##########################################################################################
##      gesture recognition model utils

# Get latest checkpoint file
def find_latest_checkpoint(directory, ckpt ,tag=None, ext='pth'):
        
    if tag:
        pattern = f"{directory}/{ckpt}_{tag}.{ext}"
    else:
        # check for all the checkpoint files
        pattern= f"{directory}/{ckpt}_*.{ext}"
    
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoints found.")
    
    # sort files by modificaiton time (latest first)
    if ext == 'pth':
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    return checkpoint_files[0]

# get device utility
def get_device(verbose=False):
    # Check for available devices
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Metal Performance Shaders (Apple GPU)
        if verbose:
            print("Using device: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU if available
        if verbose:
            print("Using device: CUDA")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        if verbose:
            print("Using device: CPU")
    
    return device

# load model from the presaved pickle of model
def load_pickle_model(pickle_path):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
    # print(f"model type: {type(model)}")
    return model

# load normalization scaler of running means and SDs
def load_pickle_scaler(pickle_path):
    with open(pickle_path, "rb") as f:
        scaler = pickle.load(f)
    # print(f"sclaer type: {type(scaler)}")
    return scaler


##########################################################################################
##      Gesture recognition
MIN_X, MAX_X = 0.20, 0.75
MIN_Y, MAX_Y = 0.20, 0.75
# is hand within detectable threshold location
def valid_position(hand_landmarks):
    index_location = find_index_tip(hand_landmarks)
    if MIN_X <= index_location.x <= MAX_X and MIN_Y<= index_location.y <= MAX_Y:
        return True
    else:
        return False

# return the normalized coordinates of index finger
def find_index_tip(hand_landmarks):
    return hand_landmarks.landmark[hand_dict["INDEX_FINGER_TIP"]]

def switch_screen(direction):
    if direction == 'left':
        with keyboard.pressed(Key.ctrl):
            keyboard.pressed(Key.left)
            keyboard.release(Key.left)
    elif direction == 'right':
        with keyboard.pressed(Key.ctrl):
            keyboard.pressed(Key.right)
            keyboard.release(Key.right)

def convert_norm_pos(index_tip):
    unnorm_x = np.interp(index_tip.x, [MIN_X, MAX_X], [-50, screen_width +10])
    unnorm_y = np.interp(index_tip.y, [MIN_Y, MAX_Y], [-50, screen_height + 10])
    return unnorm_x, unnorm_y
    

# mapping gestures to mouse operations
def mouse_op(prediction, hand_landmarks):
    
    # globals
    global prev_index_x, prev_index_y, curr_index_x, curr_index_y
    global button_counter, button_delay
    global scroll_sensitivity, scroll_threshold
    
    # reset global values if index moves out of detectable threshold
    if not valid_position(hand_landmarks):
        reset_globals()
    
    # mouse ops
    if prediction == 1:             # track index 
        index_tip = find_index_tip(hand_landmarks)
        # Scale x and y using np.interp
        x, y = convert_norm_pos(index_tip)
        # smoothening:
        curr_index_x = prev_index_x + (x  - prev_index_x) / smoothening
        curr_index_y = prev_index_y + (y  - prev_index_y) / smoothening
        
        # mouse.position = (x, y)
        mouse.position = (curr_index_x, curr_index_y)
        
        # update index tip locations
        prev_index_x, prev_index_y = curr_index_x, curr_index_y
        
    elif prediction == 2:            # left click
        button_counter += 1
        if button_counter > button_delay:
            button_counter = 0
        if button_counter == 0:
            mouse.click(Button.left, 1)

    elif prediction == 3:            # double-left click
        button_counter += 1
        if button_counter > button_delay:
            button_counter = 0
        if button_counter == 0:
            mouse.click(Button.left, 2)
    
    elif prediction == 4:            # scrolling
        _, y = convert_norm_pos(find_index_tip(hand_landmarks))
        delta = y - prev_index_y
        
        if abs(delta) > scroll_threshold:
            scroll_amt = int(delta * scroll_sensitivity)
            mouse.scroll(0, scroll_amt)
        prev_index_y = y
        
    elif prediction == 5:            # right click
        button_counter += 1
        if button_counter > button_delay:
            button_counter = 0
        if button_counter == 0:            
            mouse.click(Button.right, 1)
        
    elif prediction == 6:            # switch screens
        x = find_index_tip(hand_landmarks).x
        delta = x-prev_index_x
        
        if x > 0.5 :
            # switch window to right
            switch_screen("right")
            
        else :
            # switch window to left
            switch_screen("left") 
    else:                           # nothing detected
        reset_globals()

# reset global varirables
def reset_globals():
    global prev_index_x, prev_index_y, curr_index_x, curr_index_y, button_counter
    prev_index_x, prev_index_y = 0, 0
    curr_index_x, curr_index_y = 0, 0         
    button_counter = 0