import cv2

cap = cv2.VideoCapture(0)

# setting up camera
cap.set(3, 300)
cap.set(4, 400)

while cap.isOpened():
    re, frame = cap.read()
    cv2.imshow('image', frame)
    
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
