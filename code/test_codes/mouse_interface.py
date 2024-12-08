import pyautogui 
import time

scr_w, scr_h = pyautogui.size()

def drag_and_select(start_x, start_y, end_x, end_y):
    """
    Simulates a drag-and-select operation.
    :param start_x: Starting x-coordinate.
    :param start_y: Starting y-coordinate.
    :param end_x: Ending x-coordinate.
    :param end_y: Ending y-coordinate.
    """
    # Move the mouse to the starting position
    pyautogui.moveTo(start_x, start_y, duration=0.5)
    
    # Press the left mouse button
    pyautogui.mouseDown()
    
    # Drag the mouse to the ending position
    pyautogui.moveTo(end_x, end_y, duration=0.5)
    
    # Release the left mouse button
    pyautogui.mouseUp()

def test_mouse_op():
    print("+-----------------------------------------------------------")
    print("| Starting mouse operations test...")
    
    # get current mouse position
    print(f"| Current_location: {pyautogui.position()}")
    
    # Move mouse to a right corner location
    print(f"| Moving mouse to ({scr_w}, {scr_h})")
    pyautogui.moveTo(scr_w, scr_h)
    
    # perform Left click
    print(f"| Performing Left CLick")
    pyautogui.click()
    
    # perform RIght click
    print(f"| Performing RIght CLick")
    pyautogui.rightClick()
    
    # Double-click at the current position
    print("| Performing double click...")
    pyautogui.doubleClick()
    
    # Drag the mouse to another position
    print("| Dragging mouse to (0, 500)...")
    pyautogui.dragTo(x=50, y=500, duration=2, button='left')
    
    # Scroll up and down
    print("| Scrolling up...")
    pyautogui.scroll(500)
    time.sleep(1)
    print("| Scrolling down...")
    pyautogui.scroll(-500)
    
    print("Mouse operation tests complete.")

if __name__ == "__main__":
    test_mouse_op()
    drag_and_select(200, 100, 500, 500 )