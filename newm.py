import cv2
import pickle
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

video = cv2.VideoCapture(0)
fa = cv2.CascadeClassifier('C:/Users/HP/Desktop/All in one/Python sums/face reco/haarcascade_frontalface.xml')

with open('face reco/names.pkl', 'rb') as f:
    LABELS = np.array(pickle.load(f))
with open('face reco/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print("Loaded data shapes:")
print("Labels:", LABELS.shape)
print("Faces:", FACES.shape)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(LABELS)

print("Encoded labels:", encoded_labels)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, encoded_labels)

img = cv2.imread('C:/users/hp/Desktop/All in one/Python sums/face reco/backgr_.png')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = fa.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in det:
        crop = gray[y:y+h, x:x+w]
        resized = cv2.resize(crop, (50, 50))  # Resize the image
        resized_flat = resized.flatten().reshape(1, -1)  # Flatten the resized image
        
        print("Shape of resized_flat:", resized_flat.shape)
        print("Shape of FACES:", FACES.shape)
        
        # Check if the number of features matches the expected number
        if resized_flat.shape[1] == FACES.shape[1]:
            output = knn.predict(resized_flat)
            
            print("Predicted label index:", output[0])
            print("Predicted label:", label_encoder.inverse_transform([output])[0])
            
            if 0 <= output[0] < len(LABELS):
                label = label_encoder.inverse_transform([output])[0]
            else:
                label = "Unknown"
        else:
            label = "Unknown"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, str(label), (x+50, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

    img_height, img_width, _ = img.shape
    frame_height, frame_width, _ = frame.shape
    start_x = (img_width - frame_width) // 2 + 50
    start_y = (img_height - frame_height) // 2 - 30
    end_x = start_x + frame_width
    end_y = start_y + frame_height
    img[start_y:end_y, start_x:end_x] = frame
    cv2.imshow("Frame", img)
    
    k = cv2.waitKey(1)
    if k == ord(' '):
        break
video.release()
cv2.destroyAllWindows()
