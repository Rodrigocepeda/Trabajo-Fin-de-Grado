import cv2
import os.path

import numpy as np
import cv2

# THIS DIRECTORY HAS TO BE MANUALLY CONFIGURED!
DIRECTORY = os.path.join('/','home','domingo','anaconda3','envs','test','share','OpenCV','haarcascades')
FILENAME = 'haarcascade_frontalface_default.xml'
CLASSIFIER = os.path.join(DIRECTORY,FILENAME)
cap = cv2.VideoCapture(0)
print(CLASSIFIER)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # We define the classifier (these files have to be found)
    face_cascade = cv2.CascadeClassifier(CLASSIFIER)
    # Here we detect the faces 
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
