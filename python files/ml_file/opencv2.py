# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data=[]
dataset_path='C:/Users/Admin/Desktop/python for ml/Face_Recoginition/data/'

file_name=input("ENTER THE NAME OF THE PERSON:")

while True:
    ret,frame=cap.read()

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    if len(faces)==0:
        continue

    faces=sorted(faces,key=lambda f:f[2]*f[3])

    #pick the last face as it is the largest
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #extract(crop ut the req face):region of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
 

        #storing every 10th frame for training data
        skip+=1
        if (skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
    
    

    cv2.imshow("video frame",frame)
    cv2.imshow("face section",face_section)

     

    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

#convert face list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
# #save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()