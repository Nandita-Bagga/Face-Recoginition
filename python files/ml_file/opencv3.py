# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.


# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name
import numpy as np
import cv2
import os

#KNN
def dist(v1,v2):
    return np.sqrt(sum(v1-v2)**2)
def knn(train,test,k=5):

    distance=[]
    for i in range(train.shape[0]):

        #get the vector and the label
        ix=train[i,:-1]
        iy=train[i,-1]
        #compute the dist from test_data
        d=dist(test,ix)
        distance.append([d,iy])

    #sort amongst distance and get top k
    dk=sorted(distance, key=lambda  x: x[0])[:k]
    #retrieve only the labels
    labels=np.array(dk)[:,-1]
    #get frequencies of each label
    output=np.unique(labels, return_counts=True)
    #find max freq and corresponding label
    index=np.argmax(output[1])
    return output[0][index]
    
#face_detection
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
dataset_path='C:/Users/Admin/Desktop/python for ml/Face_Recoginition/data/'

face_data=[]
labels=[]

#labels for given file
class_id=0
#mapping of names with id
names={}

#loading the training data
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        print("Loaded "+fx)
        data_item=np.load(dataset_path+fx)
        #saving in the list of training data
        face_data.append(data_item)

        #create labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        #saving in the list of labels of the training data
        labels.append(target)
        class_id=+1

        #combing all the training set data into one list
        face_dataset=np.concatenate(face_data,axis=0)
        face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

        print(face_dataset.shape)
        print(face_labels.shape)

        #converting the final list combing into one matrix as KNN takes matrix as input
        trainset=np.concatenate((face_dataset,face_labels),axis=1)
        print(trainset.shape)

#extractng the faces for testing
while True:
    ret,frame=cap.read()

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    if len(faces)==0:
        continue

    

    
    for face in faces:
        x,y,w,h=face

        #extract(crop out the req face):region of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        #use KNN to find the prediction of face 
        out=knn(trainset,face_section.flatten())
        
        #display the prediction on the screen and the name and rectangle out it
        pred_name=names[int(out)] 
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()