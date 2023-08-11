import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('storage/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

enrol=input("Enter your Enrollment number: ")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)


if 'enroll.pkl' not in os.listdir('storage/'):
    enroll=[enrol]*100
    with open('storage/enroll.pkl', 'wb') as f:
        pickle.dump(enroll, f)
else:
    with open('storage/enroll.pkl', 'rb') as f:
        enroll=pickle.load(f)
    enroll=enroll+[enrol]*100
    with open('storage/enroll.pkl', 'wb') as f:
        pickle.dump(enroll, f)

if 'faces_data.pkl' not in os.listdir('Storage/'):
    with open('Storage/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('Storage/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('Storage/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)