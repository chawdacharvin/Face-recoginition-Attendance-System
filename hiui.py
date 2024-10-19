import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

video = cv2.VideoCapture(0)
fa = cv2.CascadeClassifier('C:/Users/HP/Desktop/All in one/Python sums/face reco/haarcascade_frontalface.xml')

with open('face reco/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('face reco/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS['labels'])  # Assuming 'labels' contain the actual labels

img = cv2.imread('C:/users/hp/Desktop/All in one/Python sums/face reco/backgr_.png')

while True:
    ret, frame = video.read()
    fade = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = fa.detectMultiScale(fade, 1.3, 5)
    
    for (x, y, w, h) in det:
        crop = frame[y:y+h, x:x+w, :]
        resized = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized)
        
        if output[0] in LABELS['labels']:
            name = LABELS['names'][output[0]]  # Get the name associated with the label
        else:
            name = "Unknown"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 0, 255), -1)
        cv2.putText(frame, name, (x+50, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

    img_height, img_width, _ = img.shape
    frame_height, frame_width, _ = frame.shape
    start_x = (img_width - frame_width) // 2 + 50  # Adjusted to move slightly to the right
    start_y = (img_height - frame_height) // 2 - 30  # Adjusted to move upward more
    end_x = start_x + frame_width
    end_y = start_y + frame_height
    img[start_y:end_y, start_x:end_x] = frame
    cv2.imshow("Frame", img)
    k = cv2.waitKey(1)
    
    if k == ord(' '):
        break

video.release()
cv2.destroyAllWindows()
