from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import sqlite3


# Connect to the SQLite database
conn = sqlite3.connect('attendance.db')
c = conn.cursor()






video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('storage/haarcascade_frontalface_default.xml')

with open('storage/enroll.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('storage/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBack=cv2.imread("Main.png")

COL_NAMES = ['Enrollment', 'TIME']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        db_filename = "attendance.db"
        formatted_date = date.replace("-", "_")
        table_name = f"Attendance_{formatted_date}"

        if os.path.isfile(db_filename):
            # Create the table if it doesn't exist
            c.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    enrollment TEXT UNIQUE,
                    time TEXT
                )
            ''')
            conn.commit()
            






        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)


        attendance=[str(output[0]), str(timestamp)]
    imgBack[93:93+480,50:50+640] = frame
    cv2.imshow("Frame",imgBack)
    k=cv2.waitKey(1)
    if k == ord('B'):
    # Insert attendance into the database
        try:
            c.execute("INSERT INTO {} (enrollment, time) VALUES (?, ?)".format(table_name), (attendance[0], timestamp))
            conn.commit()
        except sqlite3.IntegrityError:
            print(f"Duplicate enrollment entry detected for {attendance[0]}. Skipping...")
        # Close the connection
        

    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
conn.close()


























































