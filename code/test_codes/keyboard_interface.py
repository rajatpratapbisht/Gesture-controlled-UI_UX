from pynput.keyboard import Controller, Key

keyb = Controller()

try: 
    keyb.press(Key.ctrl)
    try: 
        keyb.press(Key.ctrl)
    except Exception as e:
        print(e)
except Exception as e:
    print(e)


keyb.release(Key.left)
keyb.release(Key.ctrl)
 
