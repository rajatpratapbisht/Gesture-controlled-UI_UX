# import sys
# sys.path.append(os.path.abspath(".."))
import glob
import os
import torch
from training.ffn import GestureNet
from sklearn.preprocessing import StandardScaler

import pickle


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
    0: "no_gesture",                    # nothing_detected
    1: "track_mouse_pointer",           # index_finger_up
    2: "select_objects_using_mouse",    # index_and_thumb_up
    3: "prepare_for_left_click",        # index_and_middle_spaced
    4: "execute_left_click",            # index_and_middle_together
    5: "prepare_for_right_click",       # index_thumb_middle_spaced
    6: "execute_right_click"            # index_thumb_middle_together
}


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

def load_pickle_model(pickle_path):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
    # print(f"model type: {type(model)}")
    return model

def load_pickle_scaler(pickle_path):
    with open(pickle_path, "rb") as f:
        scaler = pickle.load(f)
    # print(f"sclaer type: {type(scaler)}")
    return scaler