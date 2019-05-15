import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
subjects =["","Male","Female"]

def detectFace(img): #Builds up estimates of faces in all training examples - MTCNN tends to align fairly consistently
    detector = MTCNN()
    tmpResult = detector.detect_faces(img)
    bounding_box = tmpResult[0]['box']
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]

def prepareTrainingData(path): #Searches through all directories to find all faces
    dirs = os.listdir(path)
    faces = []
    labels = []
    for dir_name in dirs:
        if dir_name.startswith("trainingdata"): #Searches through trainingdata directory
            label = int(dir_name.replace("trainingdata","")) #blanks out to get just file name
            trainingPath = path + "/" + dir_name
            imagesFound = os.listdir(trainingPath)
            for imageName in imagesFound:
                if not imageName.startswith("."): #ensures only images loaded
                    imgPath = trainingPath + "/" + imageName
                    print(imgPath) #Helps with debugging
                    image = cv2.imread(imgPath)
                    face = detectFace(image)
                    face = cv2.resize(face,(256,256)) #Ensures constant face size for recogniser
                    faces.append(face)
                    labels.append(label)
    return faces,labels #Returns arrays of faces with labels of either male or female depending on directory

faces,labels = prepareTrainingData("train")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels)) #trains against labels to produce gender estimator
initialImg = cv2.imread("demo.jpg")
tmpImg = detectFace(initialImg)
tmpImg = cv2.resize(tmpImg, (256, 256))
label = face_recognizer.predict(tmpImg)
print("Gender of josh is:",label)
initialImg2 = cv2.imread("demo2.jpg")
tmpImg = detectFace(initialImg2)
tmpImg = cv2.resize(tmpImg, (256, 256))
label = face_recognizer.predict(tmpImg)
print("Gender of Orla is:",label)
face_recognizer.save("genderModel.xml")


