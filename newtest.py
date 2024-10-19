import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
fa = cv2.CascadeClassifier('C:/Users/HP/Desktop/All in one/Python sums/face reco/haarcascade_frontalface.xml')
faces_data = []
name = input("Please enter your Name:")
i = 0

while True:
    ret, frame = video.read()
    fade = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = fa.detectMultiScale(fade, 1.3, 5)

    for (x, y, w, h) in det:
        crop = frame[y:y+h, x:x+w, :]
        resized = cv2.resize(crop, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Attendance", frame)
    k = cv2.waitKey(1)
    if k == ord(' ') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Load or initialize names
if 'names.pkl' not in os.listdir('C:/Users/HP/Desktop/All in one/Python sums/face reco'):
    names = [name] * 100  # Initialize with 100 repetitions of name
else:
    with open('C:/Users/HP/Desktop/All in one/Python sums/face reco/names.pkl', 'rb') as f:
        names = pickle.load(f)
    # Extend the list with 100 more repetitions of name
    names.extend([name] * 100)

# Save names
with open('C:/Users/HP/Desktop/All in one/Python sums/face reco/names.pkl', 'wb') as f:
    pickle.dump(names, f)

# Save face data
if 'faces_data.pkl' not in os.listdir('C:/Users/HP/Desktop/All in one/Python sums/face reco'):
    with open('C:/Users/HP/Desktop/All in one/Python sums/face reco/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('C:/Users/HP/Desktop/All in one/Python sums/face reco/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('C:/Users/HP/Desktop/All in one/Python sums/face reco/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
