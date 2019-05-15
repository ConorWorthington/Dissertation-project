import tkinter as tk
import numpy
from tkinter.filedialog import *
from tkinter import messagebox
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN
from keras.models import Sequential
from tensorflow.python import keras
from keras import *
from keras.layers import *
from keras.models import *
import cv2
import webbrowser
import re
#Import all the necessary files

class mainApp:
    def donothing(self):
        j = 1;
        messagebox.showerror("Method not created",'Method is not yet implemented')
    #Opens up a webcam window and sets up variables so that the webcam will continue to update frames
    def webcamCap(self):
        try:
            if not self.cap.isOpened():
                self.capFrames = True #Setup update frames
                self.popupCamWindow = tk.Toplevel() #Build window
                self.popupCamWindow.resizable(False,False)
                self.popupCamWindow.wm_title("Webcam capture")
                self.popupCamWindow.lmain = tk.Label(self.popupCamWindow)
                self.popupCamWindow.lmain.grid(row=0, column=0,columnspan = 4)
                self.popupCamWindow.takePicBtn = tk.Button(self.popupCamWindow,text="Take photo",command=self.capFrame)
                self.popupCamWindow.takePicBtn.grid(row=2,column=0)
                self.popupCamWindow.exitWebcam = tk.Button(self.popupCamWindow,text="Exit Webcam",command=self.removeWindow)
                self.popupCamWindow.exitWebcam.grid(row=2,column=3)
                self.popupCamWindow.saveWebcam = tk.Button(self.popupCamWindow,text="Save Image",command = self.saveWebcam)
                self.popupCamWindow.saveWebcam.grid(row=2,column = 1)
                self.popupCamWindow.restartCam = tk.Button(self.popupCamWindow,text="Restart webcam",command = self.restartWebCam)
                self.popupCamWindow.restartCam.grid(row=2,column =2)
                self.cap = cv2.VideoCapture(0) #Begin capture
                _, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.popupCamWindow.lmain.imgtk = imgtk
                self.popupCamWindow.lmain.configure(image=imgtk)
                self.popupCamWindow.protocol('WM_DELETE_WINDOW', self.removeWindow)
            else:
                messagebox.showerror("Webcam running","Webcam is already running - it can not be started again!")
        except:
            self.removeWindow() #Destroy background window if no webcam is detected - prevents bug with popup
            messagebox.showerror("Webcam error", 'Webcam was unable to initalise')

    def removeWindow(self): #Handle popup webcam being closed
        self.popupCamWindow.destroy()
        self.cap.release()

    def capFrame(self): #Update call to capture the webcam frame
        self.capFrames = False
        _, frame = self.cap.read()
        self.webCamImage = cv2.flip(frame, 1) #webcam is initially inverted - this is corrected
        detector = MTCNN()
        tmpResult = detector.detect_faces(frame)
        tmpFaces = len(tmpResult)
        if (tmpFaces != 0): #Check image contains any faces
            for x in range(0,tmpFaces):
                bounding_box = tmpResult[x]['box']
                cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2) #Draw box around frame
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img) #Configure so temporary image doesn't draw box on live image that's being saved
            self.popupCamWindow.lmain.imgtk = imgtk
            self.popupCamWindow.lmain.configure(image=imgtk)
        else:
            messagebox.showerror("No faces were found","Error no faces were found, please move closer or adjust the lighting of the photo")
            self.capFrames = True #If no faces in image, alert user then automatically begin capturing frames again

    def saveWebcam(self): #Saves files as user wishes then proceeds to open file in main
        if self.capFrames == False:
            location = asksaveasfilename(confirmoverwrite='true', defaultextension='.jpg',
                                             filetypes=[("Image File", '.jpg')])
            if (location != ''):
                try:
                    cv2.imwrite(location, self.webCamImage)
                    self.openWebcamFile(location)
                except:
                    messagebox.showerror("File save error","File has not been saved succesfully - please try again")
            else:
                messagebox.showerror("No file Selected", 'Please provide a filename to allow the file to be saved')

    def openWebcamFile(self,location): #Sets up the enviroment to handle the newly loaded file whilst also closing the webcam window
        try:
            detector = MTCNN();
            self.filePath = location
            self.selectedFace = 0;
            tmpResult = detector.detect_faces(self.webCamImage)
            tmpFaces = len(tmpResult)
            self.modified = False;
            self.sourceImg = self.webCamImage
            self.mainImage = self.sourceImg.copy()
            self.result = tmpResult
            self.numFaces = len(self.result)
            self.keypointValues = np.zeros((self.numFaces, 15, 2))
            self.rightEyeBrowValues = np.zeros((2))
            self.leftEyeBrowValues = np.zeros((2))
            self.rightEyeValues = np.zeros((2))
            self.leftEyeValues = np.zeros((2))
            self.mouthValues = np.zeros((2))
            self.emotionallyPredicted = False
            self.undoImage = None
            self.redoImage = None
            self.redoAvailable = 0;
            self.checkAndUpdateSubImages()
            self.drawRectangles()
            self.keyPointPredictions()
            self.drawSubImage()
            mainFaceImage.configure(text="Source Photo - " + self.filePath)
            self.removeWindow()
            numFacesTxt.configure(text=self.numFaces)
            self.classifyGender()
            if self.gender[0] == 1:
                faceGenderTxt.configure(text="Male")
            else:
                faceGenderTxt.configure(text="Female")
        except:
            messagebox.showerror("Image open error","An error has occured opening the webcam image, please try opening manually or re-taking photo")

    def restartWebCam(self): #Begins capturing frames again and updating the webcam
        if self.capFrames == False:
            self.capFrames = True
        else:
            messagebox.showerror("Webcam already capturing", 'Webcam is currently capturing images, it can not start again!')

    def poll(self): #Method called by timing thread to scale window resolution and update webcam feed
        if self.cap.isOpened():
            if(self.popupCamWindow.winfo_exists()): #Ensure a webcam window exists before trying to update it
                if (self.capFrames == True):
                    self.updateFrame()
            else:
                self.cap.release() #Kills an open webcam capture if it's accidentally been left on somehow
        self.checkResolution() #Update screen resolution
        root.after(1,self.poll) #call itself constantly to update

    def updateFrame(self): #Update the image displayed on webcam popup window
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.popupCamWindow.lmain.imgtk = imgtk
        self.popupCamWindow.lmain.configure(image=imgtk)

    def openFile(self): #Handles opening of a file
        path= askopenfilename(filetypes=[("Image File",'.jpg')]) #Ensure file is an image
        if (path != ''):
            self.filePath = path
            self.selectedFace = 0;
            tmpImg = cv2.imread(path)
            detector = MTCNN();
            tmpResult = detector.detect_faces(tmpImg) #Find all available faces and build bounds for them
            tmpFaces = len(tmpResult) #get number of faces
            if(tmpFaces != 0): #Assuming a face is detected
                self.modified = False;
                self.emotionallyPredicted = False
                self.sourceImg = tmpImg.copy()
                self.mainImage = self.sourceImg.copy() #Setup images using copy so value is moved rather than address
                self.result = tmpResult
                self.numFaces = len(self.result)
                self.keypointValues = np.zeros((self.numFaces,15,2))
                self.rightEyeBrowValues = np.zeros((2))
                self.leftEyeBrowValues = np.zeros((2))
                self.rightEyeValues = np.zeros((2))
                self.leftEyeValues = np.zeros((2))
                self.mouthValues = np.zeros((2)) #Zero all variables until setup by other methods
                self.checkAndUpdateSubImages()
                self.undoImage = None
                self.redoImage = None #Prevent undoing into previous state from last image
                self.drawRectangles() #Draw boxes on every face
                self.keyPointPredictions() #Predict all points on faces
                self.estimateAngle() #provide an estimate of the facial angle
                self.drawSubImage() #Update the sub image to be loaded
                mainFaceImage.configure(text="Source Photo - "+self.filePath)
                self.classifyGender()
                numFacesTxt.configure(text=self.numFaces)
                if self.gender[0] == 1: #Make use of gender classifier for default face
                    faceGenderTxt.configure(text="Male")
                else:
                    faceGenderTxt.configure(text="Female")
            else:
                messagebox.showerror("No faces found",'The selected image file did not contain any faces which could be detected')
        else:
            messagebox.showerror("File not selected", 'No file was selected')

    def classifyGender(self):
        genderClassifier = cv2.face.FisherFaceRecognizer_create() #Load fisheface model that has been created
        genderClassifier.read("genderModel.xml")
        self.gender = np.zeros(self.numFaces)
        for x in range(0,self.numFaces):
            bounding_box = self.result[x]['box']
            tmpImg = self.sourceImg[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
            tmpImg = cv2.resize(tmpImg, (256, 256))
            tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY)
            label = genderClassifier.predict(tmpImg)
            self.gender[x] = label[0] #Forevery face load the area bound and classify the gender

    def resizeMain(self): #Update main image dimenisons
        (h,w) = self.mainImage.shape[:2]
        idealHeight  = int((self.windowHeight/1080)*900) #Scale all aspect ratios to designed UI of 1920X1080
        idealWidth  = int((self.windowWidth/1920)*1280) #Scale with ideal image size of 1280* 900
        newHeight = int((self.windowHeight/1080)*900)
        newWidth = int((self.windowWidth/1920)*1280)
        ratio = 0
        paddingY = 0
        paddingX = 0
        label1.grid(row=0,column=0,columnspan = 5,rowspan = 5, padx=2.5, pady=2.5)
        if(w>h): #If image is wider than tall resolve ratio and calculate new dims from width calculation
                ratio = newWidth / w
                newHeight = int(h*ratio)
                if (newHeight > idealHeight):
                    ratio = idealHeight/newHeight
                    newWidth = int(newWidth*ratio)
                    newHeight = idealHeight
                dim = (newWidth,newHeight)
                self.mainImage = cv2.resize(self.mainImage, dim, interpolation = cv2.INTER_AREA)
        elif (h>=w):#If image is taller than wide resolve ratio and calculate new dims from width calculation
                ratio = newHeight / h
                newWidth = int(w*ratio)
                if (newWidth > idealWidth):
                    ratio = idealWidth/newWidth
                    newHeight = int(newHeight*ratio)
                    newWidth = idealWidth
                dim = (newWidth,newHeight)
                self.mainImage = cv2.resize(self.mainImage, dim, interpolation = cv2.INTER_AREA)
        paddingX = idealWidth - newWidth; #Calculate padding that will ensure UI retains shape
        paddingY = idealHeight - newHeight;
        if(paddingX>=0 and paddingY>=0):
                label1.grid(row=0,column=0,columnspan = 7,rowspan = 5, padx=paddingX/2, pady=paddingY/2)


    def drawSubImage(self): #Re-apply main image code to dynamically scale sub image whilst also updating the image contents
        if(self.filePath != ''): #Setup before image as from source and when not modified set up after image as source too
            bounding_box = self.result[self.selectedFace]['box']
            self.beforeImg = self.sourceImg[int(bounding_box[1]):int(bounding_box[1] + bounding_box[3]),int(bounding_box[0]):int(bounding_box[0] + bounding_box[2])].copy()
            if(self.modified == False):
                self.croppedImg = self.sourceImg[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
        else:
            self.croppedImg = cv2.imread('default.jpg')
            self.beforeImg = self.sourceImg.copy()
        if (self.filePath != ''):
            self.keyPointAverages()
            if self.drawKeypointsVar.get() == 1:
                self.drawKeypoints() #draws keypoints on before image prior to rescaling
            if self.drawAngleVar.get() == 1:
                self.drawAngles() #Draws angles on before image prior to rescaling
        self.afterImg = self.croppedImg.copy()
        idealHeight  = int((self.windowHeight/1080)*350)
        idealWidth  = int((self.windowWidth/1920)*300)
        newHeight = idealHeight
        newWidth = idealWidth
        (h,w) = self.croppedImg.shape[:2]
        ratio = 0
        paddingY = 0
        paddingX = 0
        label2.grid(row=0,column=0,columnspan = 5,rowspan = 5, padx=2.5, pady=2.5)
        label3.grid(row=0,column=0,columnspan = 5,rowspan = 5, padx=2.5, pady=2.5)
        if(w>h): #Same size transformation as main image but for both before and after image
                ratio = newWidth / w
                newHeight = int(h*ratio)
                if (newHeight > idealHeight):
                    ratio = idealHeight/newHeight
                    newWidth = int(newWidth*ratio)
                    newHeight = idealHeight
                dim = (newWidth,newHeight)
                self.beforeImg = cv2.resize(self.beforeImg, dim, interpolation = cv2.INTER_AREA)
                self.afterImg = cv2.resize(self.afterImg, dim, interpolation = cv2.INTER_AREA)
        elif (h>=w):
                ratio = newHeight / h
                newWidth = int(w*ratio)
                if (newWidth > idealWidth):
                    ratio = idealWidth/newWidth
                    newHeight = int(newHeight*ratio)
                    newWidth = idealWidth
                dim = (newWidth,newHeight)
                self.beforeImg = cv2.resize(self.beforeImg, dim, interpolation = cv2.INTER_AREA)
                self.afterImg = cv2.resize(self.afterImg, dim, interpolation = cv2.INTER_AREA)
        paddingX = idealWidth - newWidth;
        paddingY = idealHeight - newHeight;
        if(paddingX>=0 and paddingY>=0):
                label2.grid(row=0,column=0,columnspan = 5,rowspan = 5, padx=paddingX/2, pady=paddingY/2)
                label3.grid(row=0,column=0,columnspan = 5,rowspan = 5, padx=paddingX/2, pady=paddingY/2)


        b,g,r = cv2.split(self.beforeImg)
        self.beforeImg = cv2.merge((r,g,b))
        im = Image.fromarray(self.beforeImg)
        photo2 = ImageTk.PhotoImage(image = im)
        label2.configure(image = photo2)
        label2.image = photo2
        b,g,r = cv2.split(self.afterImg)
        self.afterImg = cv2.merge((r,g,b))
        im = Image.fromarray(self.afterImg)
        photo2 = ImageTk.PhotoImage(image = im)
        label3.configure(image = photo2)
        label3.image = photo2
        if self.filePath != '':
            if self.emotionallyPredicted == False:
                self.updateEmotions() #If the images haven't been emotionally predicted since a change i.e transformation repredict now

    def estimateAngle(self):
        faceProjection = np.array([ #Make use of model and camera estimation from tutorial https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        shape = (3,self.numFaces)
        self.rotationVectors = np.zeros(shape)
        self.translationVectors = np.zeros(shape)
        for x in range(0,self.numFaces):
            bounding_box = self.result[x]['box']
            predicted2dPoints = np.array([
                (int((self.keypointValues[x][10][0]*bounding_box[2])/96),int((self.keypointValues[x][10][1]*bounding_box[3])/96)),  # Nose tip
                (int((self.keypointValues[x][13][0]*bounding_box[2])/96),int(bounding_box[3])),  # Chin
                (int((self.keypointValues[x][5][0]*bounding_box[2])/96), int((self.keypointValues[x][5][0]*bounding_box[3])/96)),  # Left eye left corner
                (int((self.keypointValues[x][3][0]*bounding_box[2])/96), int((self.keypointValues[x][3][0]*bounding_box[3])/96)),  # Right eye right corner
                (int((self.keypointValues[x][12][0]*bounding_box[2])/96),int((self.keypointValues[x][12][0]*bounding_box[3])/96)),  # Left Mouth corner
                (int((self.keypointValues[x][11][0]*bounding_box[2])/96), int((self.keypointValues[x][11][0]*bounding_box[3])/96))  # Right mouth corner
            ], dtype="double") #Build up array of all points from face scaled appropriately
            focalLength = bounding_box[2]
            centerPoint = (bounding_box[2]/2,bounding_box[3]/2)
            cameraMatrix = np.array([[focalLength, 0, centerPoint[0]],[0, focalLength, centerPoint[1]],[0, 0, 1]],dtype="double")
            distanceCoEfficient = np.zeros((4,1))
            (success, rotationVector, translationVector) = cv2.solvePnP(faceProjection, predicted2dPoints, cameraMatrix,distanceCoEfficient, flags=cv2.SOLVEPNP_ITERATIVE)
            #Resolve the solvePNP code to build up an estimation of angle
            for y in range(0,3):
                self.rotationVectors[y][x] = rotationVector[y] #Iterate through every angle and save for every face
                self.translationVectors[y][x] = translationVector[y]

    def updateEmotions(self): #Update code to setup GUI so that emotions as text are updated to new variables
        self.emotionallyPredicted = True
        bounding_box = self.result[self.selectedFace]['box']
        beforeResults = self.emotionalPred.emotionalPredict(self.sourceImg[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])])
        beforeEmotion = numpy.where(beforeResults == numpy.amax(beforeResults))
        predictedOldPct.configure(text=self.emotions[int(beforeEmotion[1])])
        angerPct.configure(text=format(beforeResults[0][0]*100,'f')+"%")
        disgustPct.configure(text=format(beforeResults[0][1]*100,'f')+"%")
        scaredPct.configure(text=format(beforeResults[0][2]*100,'f')+"%")
        happyPct.configure(text=format(beforeResults[0][3]*100,'f')+"%")
        sadPct.configure(text=format(beforeResults[0][4]*100,'f')+"%")
        suprisedPct.configure(text=format(beforeResults[0][5]*100,'f')+"%")
        neutralPct.configure(text=format(beforeResults[0][6]*100,'f')+"%")
        if self.modified == True:
            afterResults = self.emotionalPred.emotionalPredict(self.afterImg)
            afterEmotion = numpy.where(afterResults == numpy.amax(afterResults))
            predictedNewPct.configure(text=self.emotions[int(afterEmotion[1])])
            angerPctNew.configure(text=format(afterResults[0][0]*100,'f')+"%")
            disgustPctNew.configure(text=format(afterResults[0][1]*100,'f')+"%")
            scaredPctNew.configure(text=format(afterResults[0][2]*100,'f')+"%")
            happyPctNew.configure(text=format(afterResults[0][3]*100,'f')+"%")
            sadPctNew.configure(text=format(afterResults[0][4]*100,'f')+"%")
            suprisedPctNew.configure(text=format(afterResults[0][5]*100,'f')+"%")
            neutralPctNew.configure(text=format(afterResults[0][6]*100,'f')+"%")
        else:
            predictedNewPct.configure(text=self.emotions[int(beforeEmotion[1])])
            angerPctNew.configure(text=format(beforeResults[0][0]*100,'f')+"%")
            disgustPctNew.configure(text=format(beforeResults[0][1]*100,'f')+"%")
            scaredPctNew.configure(text=format(beforeResults[0][2]*100,'f')+"%")
            happyPctNew.configure(text=format(beforeResults[0][3]*100,'f')+"%")
            sadPctNew.configure(text=format(beforeResults[0][4]*100,'f')+"%")
            suprisedPctNew.configure(text=format(beforeResults[0][5]*100,'f')+"%")
            neutralPctNew.configure(text=format(beforeResults[0][6]*100,'f')+"%")

    def drawAngles(self): #Collapse 3d point into a 2d point and subsequently draw it from the nose indicating the rotation vector
        (h,w) = self.beforeImg.shape[:2]
        bounding_box = self.result[self.selectedFace]['box']
        rotationVector = np.array([[self.rotationVectors[0][self.selectedFace],self.rotationVectors[1][self.selectedFace],self.rotationVectors[2][self.selectedFace]]],dtype="float")
        translationVector = np.array([[self.translationVectors[0][self.selectedFace], self.translationVectors[1][self.selectedFace],self.translationVectors[2][self.selectedFace]]],dtype="float")
        focalLength = bounding_box[2]
        centerPoint = (bounding_box[2] / 2, bounding_box[3] / 2)
        cameraMatrix = np.array([[focalLength, 0, centerPoint[0]], [0, focalLength, centerPoint[1]], [0, 0, 1]],dtype="double")
        distanceCoefficients = np.zeros((4,1))
        (nosePoint2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 75.0)]), rotationVector,translationVector, cameraMatrix, distanceCoefficients)
        noseStart = (int((self.keypointValues[self.selectedFace][10][0]* (w/96))),int((self.keypointValues[self.selectedFace][10][1]* (h/96))))
        cv2.line(self.beforeImg, noseStart, (int(nosePoint2D[0][0][0]),int(nosePoint2D[0][0][1])), (255, 0, 0), 2)

    def drawKeypoints(self): #Place a circle at every single point which
        (h,w) = self.beforeImg.shape[:2]
        dims = np.zeros(2)
        for x in range(0,15):
            dims[0] = self.keypointValues[self.selectedFace][x][0]
            dims[1] = self.keypointValues[self.selectedFace][x][1]
            dims[0] = int(dims[0] * (w/96))
            dims[1] = int(dims[1] * (h/96))
            cv2.circle(self.beforeImg,(int(dims[0]),int(dims[1])),3,[255,255,255])
        cv2.circle(self.beforeImg,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),3,[255,0,0])
        cv2.circle(self.beforeImg,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),3,[255,0,0])
        cv2.circle(self.beforeImg,(int(self.mouthValues[0]),int(self.mouthValues[1])),3,[255,0,0])
        cv2.circle(self.beforeImg,(int(self.rightEyeValues[0]),int(self.rightEyeValues[1])),3,[255,0,0])
        cv2.circle(self.beforeImg,(int(self.leftEyeValues[0]),int(self.leftEyeValues[1])),3,[255,0,0])


    def checkAndUpdateSubImages(self): #ensures bounding box doesn't lay outside limits of image preventing crash
        width,height = self.sourceImg.shape[:2]
        for x in range(0,self.numFaces):
            bounding_box = self.result[x]['box']
            if(bounding_box[0] < 0):
                bounding_box[0] = 0;
            if(bounding_box[1] < 0):
                bounding_box[1] = 0;
            if(bounding_box[1]+bounding_box[3] > width):
                bounding_box[3] = width - bounding_box[1]
            if(bounding_box[0]+bounding_box[2] > height):
                bounding_box[2] = height - bounding_box[0]
            self.result[x]['box'] = bounding_box


    def drawRectangles(self): #Draws rectangles around every face that's been detected
        self.mainImage = self.sourceImg.copy()
        if (self.filePath != ''):
            for x in range(0,self.numFaces):
                bounding_box = self.result[x]['box']
                if (x == self.selectedFace):
                    cv2.rectangle(self.mainImage,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,255,0),
                          2)
                else:
                    cv2.rectangle(self.mainImage,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
        self.resizeMain()
        b,g,r = cv2.split(self.mainImage)
        self.mainImage = cv2.merge((r,g,b))
        im = Image.fromarray(self.mainImage)
        photo2 = ImageTk.PhotoImage(image = im)
        label1.configure(image = photo2)
        label1.image = photo2


    def nextFace(self): #loads next face by updating subimages, emotions and changing colour of rectangles
        self.modified = False;
        self.emotionallyPredicted = False
        if self.selectedFace < (self.numFaces-1):
            self.selectedFace = self.selectedFace + 1
            self.drawRectangles()
        else:
            self.selectedFace = 0
            self.drawRectangles()
        self.drawSubImage()
        if self.gender[self.selectedFace] == 1:
            faceGenderTxt.configure(text="Male")
        else:
            faceGenderTxt.configure(text="Female")

    def saveFile(self): #Saves face over current file
        if(self.filePath!=''):
            cv2.imwrite(self.filePath,self.sourceImg)
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to save')

    def saveAsFile(self): #Saves face over the new file location the user selected
        if(self.filePath!=''):
            location = asksaveasfilename(confirmoverwrite='true',defaultextension='.jpg',filetypes=[("Image File",'.jpg')])
            if(location!=''):
                cv2.imwrite(location,self.sourceImg)
                self.filePath = location
                mainFaceImage.configure(text="Source Photo - "+self.filePath)
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to save')

    def prevFace(self): #Moves to previous face and subsequently redraws all emotional state, sub images etc
        self.modified = False;
        self.emotionallyPredicted = False
        if (self.selectedFace-1) >= 0:
            self.selectedFace = self.selectedFace - 1
            self.drawRectangles()
        else:
            self.selectedFace = self.numFaces-1
            self.drawRectangles()
        self.drawSubImage()
        if self.gender[self.selectedFace] == 1:
            faceGenderTxt.configure(text="Male")
        else:
            faceGenderTxt.configure(text="Female")

    def keyPointPredictions(self): #Uses keypoints predictor to predict all points for each face within its bounds
        for x in range(0,self.numFaces):
            bounding_box = self.result[x]['box']
            self.croppedImg = self.sourceImg[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])]
            xyPredictions = self.keypointsPred.keypointPredict(self.croppedImg)
            self.keypointValues[x][:][:] = xyPredictions[:][:]

    def keyPointAverages(self): #Calculates average points for eyebrow, mouth and eyes which are used during emotional transformation
        (h,w) = self.beforeImg.shape[:2]
        self.rightEyeBrowValues[0] = (self.keypointValues[self.selectedFace][6][0] + self.keypointValues[self.selectedFace][7][0]) /2
        self.rightEyeBrowValues[1] = (self.keypointValues[self.selectedFace][6][1] + self.keypointValues[self.selectedFace][7][1]) /2
        self.rightEyeBrowValues[0] = int(self.rightEyeBrowValues[0] * (w/96))
        self.rightEyeBrowValues[1] = int(self.rightEyeBrowValues[1] * (h/96))
        self.leftEyeBrowValues[0] = (self.keypointValues[self.selectedFace][8][0] + self.keypointValues[self.selectedFace][9][0]) /2
        self.leftEyeBrowValues[1] = (self.keypointValues[self.selectedFace][8][1] + self.keypointValues[self.selectedFace][9][1]) /2
        self.leftEyeBrowValues[0] = int(self.leftEyeBrowValues[0] * (w/96))
        self.leftEyeBrowValues[1] = int(self.leftEyeBrowValues[1] * (h/96))
        self.mouthValues[0] = (self.keypointValues[self.selectedFace][11][0] + self.keypointValues[self.selectedFace][12][0]+self.keypointValues[self.selectedFace][13][0] + self.keypointValues[self.selectedFace][14][0]) /4
        self.mouthValues[1] = (self.keypointValues[self.selectedFace][11][1] + self.keypointValues[self.selectedFace][12][1]+self.keypointValues[self.selectedFace][13][1] + self.keypointValues[self.selectedFace][14][1]) /4
        self.mouthValues[0] = int(self.mouthValues[0] * (w/96))
        self.mouthValues[1] = int(self.mouthValues[1] * (h/96))
        self.rightEyeValues[0] = (self.keypointValues[self.selectedFace][0][0] + self.keypointValues[self.selectedFace][2][0]+self.keypointValues[self.selectedFace][3][0]) /3
        self.rightEyeValues[1] = (self.keypointValues[self.selectedFace][0][1] + self.keypointValues[self.selectedFace][2][1]+self.keypointValues[self.selectedFace][3][1]) /3
        self.rightEyeValues[0] = int(self.rightEyeValues[0] * (w/96))
        self.rightEyeValues[1] = int(self.rightEyeValues[1] * (h/96))
        self.leftEyeValues[0] = (self.keypointValues[self.selectedFace][1][0] + self.keypointValues[self.selectedFace][4][0]+self.keypointValues[self.selectedFace][5][0]) /3
        self.leftEyeValues[1] = (self.keypointValues[self.selectedFace][1][1] + self.keypointValues[self.selectedFace][4][1]+self.keypointValues[self.selectedFace][5][1]) /3
        self.leftEyeValues[0] = int(self.leftEyeValues[0] * (w/96))
        self.leftEyeValues[1] = int(self.leftEyeValues[1] * (h/96))

    def drawOntoMainImage(self): #Updates main image to have selected face reflect the now altered state
        if(self.filePath != ''):
            self.undoImage = self.sourceImg.copy()
            bounding_box = self.result[self.selectedFace]['box']
            self.sourceImg[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+bounding_box[2])] = self.croppedImg
            self.drawRectangles()
            self.drawSubImage()

    def updateAfterImg(self): #Ensures that next time frame is drawn the emotional state is updated
        self.emotionallyPredicted = False

    def angryEmotionTransformation(self): #Begins emotional transformation
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1: #If the genders male replace with masculine features
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleAngry.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA) #Gets full facial regions
                maskArray = np.array([ [70*propX,228*propY], [99*propX,215*propY], [125*propX,214*propY] , [156*propX,217*propY], [179*propX,231*propY], [183*propX,262*propY], [163*propX,274*propY], [131*propX,283*propY], [106*propX,279*propY], [80*propX,269*propY], [68*propX,251*propY]], np.int32)
                #Builds a full mask of the area we wish to take and subsequently scales it
                mask = np.zeros((h,w),emotFace.dtype) #Builds a pure black image of the size of the new image
                cv2.fillPoly(mask,[maskArray],(255,255,255)) #Fills new poly shape with white to mask where to transform
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                #Performs poisson blend
                if self.updateEyebrows.get() == 1: #Checks if eyebrow region should be updated
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[137 * propX, 107 * propY], [175 * propX, 103 * propY], [195 * propX, 101 * propY],
                         [229 * propX, 99 * propY], [244 * propX, 97 * propY], [242 * propX, 64 * propY],
                         [215 * propX, 58 * propY], [173 * propX, 61 * propY], [144 * propX, 66 * propY],
                         [132 * propX, 75 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    #Sets up new mask to target new area to transform
                    self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask, (int(self.rightEyeBrowValues[0]), int(self.rightEyeBrowValues[1])), cv2.NORMAL_CLONE)
                    #Transforms one eyebrow region
                    mask = np.zeros((h, w), emotFace.dtype)
                    maskArray = np.array(
                        [[15 * propX, 103 * propY], [36 * propX, 98 * propY], [63 * propX, 99 * propY],
                         [98 * propX, 107 * propY], [117 * propX, 117 * propY], [118 * propX, 58 * propY],
                         [20 * propX, 49 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    #Transforms other eyebrow region
            else: #Code identical to male anger - but for female mask instead
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('angry.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[102 * propX, 249 * propY], [130 * propX, 244 * propY], [174 * propX, 245 * propY],
                     [212 * propX, 263 * propY],[209 * propX, 289 * propY],[170 * propX, 305 * propY],[127 * propX, 303 * propY],[96 * propX, 288 * propY],[84 * propX, 266 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[161 * propX, 106 * propY], [215 * propX, 104 * propY], [269 * propX, 103 * propY],
                         [273 * propX, 69 * propY], [212 * propX, 61 * propY], [162 * propX, 69 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[32 * propX, 107 * propY], [79 * propX, 103 * propY], [143 * propX, 111 * propY],
                         [144 * propX, 67 * propY], [81 * propX, 60 * propY], [46 * propX, 74 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')
    def neutralEmotionTransformation(self): #Code identical to male anger - but for neutral mask instead
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1:
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleNeutral.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA)
                maskArray = np.array([ [88*propX,380*propY], [134*propX,347*propY], [201*propX,339*propY] , [262*propX,343*propY], [310*propX,377*propY], [269*propX,420*propY], [200*propX,436*propY], [124*propX,419*propY]], np.int32)
                mask = np.zeros((h,w),emotFace.dtype)
                cv2.fillPoly(mask,[maskArray],(255,255,255))
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[214 * propX, 162 * propY], [213 * propX, 99 * propY], [312 * propX, 81 * propY],
                         [374 * propX, 100 * propY], [370 * propX, 170 * propY], [291 * propX, 151 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask, (int(self.rightEyeBrowValues[0]), int(self.rightEyeBrowValues[1])), cv2.NORMAL_CLONE)
                    mask = np.zeros((h, w), emotFace.dtype)
                    maskArray = np.array(
                        [[22 * propX, 182 * propY], [90 * propX, 159 * propY], [172 * propX, 168 * propY],
                         [173 * propX, 102 * propY], [85 * propX, 87 * propY], [25 * propX, 101 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            else:
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('neutral.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[91 * propX, 340 * propY], [140 * propX, 308 * propY], [200 * propX, 299 * propY],
                     [258 * propX, 305 * propY],[303 * propX, 336 * propY],[276 * propX, 376 * propY],[209 * propX, 387 * propY],[115 * propX, 376 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[216 * propX, 145 * propY], [364 * propX, 142 * propY], [357 * propX, 84 * propY],
                         [219 * propX, 84 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask, (int(self.rightEyeBrowValues[0]), int(self.rightEyeBrowValues[1])), cv2.NORMAL_CLONE)
                    mask = np.zeros((h, w), emotFace.dtype)
                    maskArray = np.array(
                        [[18 * propX, 151 * propY], [168 * propX, 147 * propY], [165 * propX, 87 * propY],
                         [27 * propX, 98 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')
    def scaredEmotionTransformation(self): #Code identical to male anger - but for scared mask instead
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1:
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleScared.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA)
                maskArray = np.array([ [64*propX,299*propY], [87*propX,268*propY], [149*propX,269*propY] , [221*propX,276*propY], [229*propX,311*propY], [192*propX,338*propY], [140*propX,342*propY], [94*propX,333*propY]], np.int32)
                mask = np.zeros((h,w),emotFace.dtype)
                cv2.fillPoly(mask,[maskArray],(255,255,255))
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[156 * propX, 117 * propY], [209 * propX, 121 * propY], [249 * propX, 132 * propY],
                         [282 * propX, 148 * propY], [268 * propX, 126 * propY], [268 * propX, 93 * propY], [225 * propX, 75 * propY], [157 * propX, 72 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask, (int(self.rightEyeBrowValues[0]), int(self.rightEyeBrowValues[1])), cv2.NORMAL_CLONE)
                    mask = np.zeros((h, w), emotFace.dtype)
                    maskArray = np.array(
                        [[143 * propX, 120 * propY], [142 * propX, 69* propY], [41 * propX, 74 * propY],
                         [19 * propX, 115 * propY], [26 * propX, 151 * propY], [63 * propX, 123 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            else:
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('scared.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[105 * propX, 451* propY], [156 * propX, 407 * propY], [218 * propX, 401 * propY],
                     [284 * propX, 409 * propY],[331 * propX, 450 * propY],[300 * propX, 506 * propY],[224 * propX, 521 * propY],[141 * propX, 505 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[234 * propX, 188 * propY], [228 * propX, 126 * propY], [405 * propX, 132 * propY],
                         [420 * propX, 210 * propY], [337 * propX, 184 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask, (int(self.rightEyeBrowValues[0]), int(self.rightEyeBrowValues[1])), cv2.NORMAL_CLONE)
                    mask = np.zeros((h, w), emotFace.dtype)
                    maskArray = np.array(
                        [[46 * propX, 191 * propY], [56 * propX, 129 * propY], [206 * propX, 123 * propY],
                         [209 * propX, 180 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')
    def disgustEmotionTransformation(self): #Code identical to male anger - but for disgust mask instead
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1:
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleDisgust.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA)
                maskArray = np.array([ [123*propX,407*propY], [174*propX,353*propY], [259*propX,339*propY] , [344*propX,343*propY], [393*propX,388*propY], [389*propX,444*propY], [307*propX,486*propY], [211*propX,492*propY], [151*propX,464*propY]], np.int32)
                mask = np.zeros((h,w),emotFace.dtype)
                cv2.fillPoly(mask,[maskArray],(255,255,255))
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[269 * propX, 167 * propY], [343 * propX, 162 * propY], [419 * propX, 170 * propY],
                         [447 * propX, 174 * propY], [444 * propX, 115 * propY], [423 * propX, 76 * propY],
                         [341 * propX, 64 * propY], [242 * propX, 70 * propY], [240 * propX, 122 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[16 * propX, 187 * propY], [102 * propX, 177 * propY], [161 * propX, 168 * propY],
                         [223 * propX, 167 * propY], [227 * propX, 128 * propY], [217 * propX, 71 * propY],
                         [96 * propX, 70 * propY], [20 * propX, 92 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)

            else:
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('disgust.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[48 * propX, 177* propY], [67 * propX, 163 * propY], [102 * propX, 163 * propY],
                     [128 * propX, 167 * propY],[146 * propX, 186 * propY],[145 * propX, 207 * propY],[113 * propX, 219 * propY],[74 * propX, 219 * propY],[48 * propX, 199 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[114 * propX, 81 * propY], [138 * propX, 78 * propY], [169 * propX, 77 * propY],
                         [183 * propX, 84 * propY], [180 * propX, 52 * propY], [160 * propX, 43 * propY],
                         [131 * propX, 44 * propY], [112 * propX, 51 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[17 * propX, 82 * propY], [54 * propX, 76 * propY], [93 * propX, 84 * propY],
                         [90 * propX, 52 * propY], [50 * propX, 42 * propY], [17 * propX, 52 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')

    def sadEmotionTransformation(self): #Code identical to male anger - but for sad mask instead
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1:
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleSad.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA)
                maskArray = np.array([ [34*propX,127*propY], [52*propX,113*propY], [71*propX,109*propY] , [93*propX,110*propY], [109*propX,122*propY], [106*propX,140*propY], [73*propX,140*propY], [50*propX,151*propY], [37*propX,142*propY]], np.int32)
                mask = np.zeros((h,w),emotFace.dtype)
                cv2.fillPoly(mask,[maskArray],(255,255,255))
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[71 * propX, 44 * propY], [104 * propX, 46 * propY], [127 * propX, 57 * propY],
                         [128 * propX, 39 * propY], [113 * propX, 22 * propY], [88 * propX, 15 * propY],
                         [70 * propX, 16 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[64 * propX, 44 * propY], [63 * propX, 17 * propY], [42 * propX, 18 * propY],
                         [17 * propX, 29 * propY], [3 * propX, 58 * propY], [16 * propX, 62 * propY],
                         [36 * propX, 48 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            else:
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('sad.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[70 * propX, 257* propY], [87 * propX, 232 * propY], [117 * propX, 220 * propY],
                     [171 * propX, 213 * propY],[209 * propX, 231 * propY],[218 * propX, 259 * propY],[185 * propX, 283 * propY],[145 * propX, 286 * propY],[91 * propX, 289 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[133 * propX, 96 * propY], [251 * propX, 89 * propY], [248 * propX, 54 * propY],
                         [133 * propX, 59 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[19 * propX, 114 * propY], [19 * propX, 78 * propY], [115 * propX, 65 * propY],
                         [116 * propX, 103 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')

    def surprisedEmotionTransformation(self): #Code identical to male anger - but for surprised mask instead
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1:
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleSurprise.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA)
                maskArray = np.array([ [174*propX,709*propY], [229*propX,645*propY], [303*propX,592*propY] , [372*propX,582*propY], [449*propX,590*propY], [509*propX,625*propY], [567*propX,710*propY], [515*propX,782*propY], [447*propX,826*propY], [368*propX,830*propY], [282*propX,815*propY], [213*propX,765*propY]], np.int32)
                mask = np.zeros((h,w),emotFace.dtype)
                cv2.fillPoly(mask,[maskArray],(255,255,255))
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[372 * propX, 143 * propY], [374 * propX, 281 * propY], [483 * propX, 255 * propY],
                         [591 * propX, 259 * propY], [658 * propX, 283 * propY], [656 * propX, 152 * propY],
                         [518 * propX, 123 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[43 * propX, 310 * propY], [169 * propX, 257 * propY], [321 * propX, 259 * propY],
                         [317 * propX, 145 * propY], [181 * propX, 129 * propY], [52 * propX, 171 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            else:
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('surprised.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[56 * propX, 194* propY], [75 * propX, 178 * propY], [102 * propX, 169 * propY],
                     [129 * propX, 176 * propY],[144 * propX, 194 * propY],[137 * propX, 217 * propY],[118 * propX, 232 * propY],[84 * propX, 233 * propY],[62 * propX, 216 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[113 * propX, 48 * propY], [112 * propX, 83 * propY], [150 * propX, 76 * propY],
                         [183 * propX, 84 * propY], [178 * propX, 50 * propY], [165 * propX, 37 * propY],
                         [132 * propX, 37 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[16 * propX, 90 * propY], [47 * propX, 76 * propY], [86 * propX, 86 * propY],
                         [81 * propX, 39 * propY], [52 * propX, 37 * propY], [18 * propX, 47 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')

    def happyEmotionTransformation(self): #Code identical to male anger - but for happy mask instead
        if(self.filePath != ''):
            self.clear()
            if self.gender[self.selectedFace] == 1:
                self.modified = True
                self.emotionallyPredicted = False
                h,w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('maleHappy.jpg')
                faceHeight,faceWdith  = emotFace.shape[:2]
                propX = w/faceWdith
                propY = h/faceHeight
                emotFace  = cv2.resize(emotFace ,(int(faceWdith *propX),int(faceHeight*propY)), interpolation = cv2.INTER_AREA)
                maskArray = np.array([ [63*propX,276*propY], [86*propX,252*propY], [151*propX,253*propY] , [205*propX,250*propY], [239*propX,260*propY], [233*propX,296*propY], [191*propX,327*propY], [154*propX,332*propY], [94*propX,316*propY]], np.int32)
                mask = np.zeros((h,w),emotFace.dtype)
                cv2.fillPoly(mask,[maskArray],(255,255,255))
                self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.mouthValues[0]),int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[172 * propX, 128 * propY], [235 * propX, 119 * propY], [272 * propX, 129 * propY],
                         [280 * propX, 100 * propY], [260 * propX, 73 * propY], [171 * propX, 77 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[18 * propX, 141 * propY], [49 * propX, 119 * propY], [125 * propX, 124 * propY],
                         [127 * propX, 82 * propY], [46 * propX, 76 * propY], [3 * propX, 100 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            else:
                self.modified = True
                self.emotionallyPredicted = False
                h, w = self.croppedImg.shape[:2]
                emotFace  = cv2.imread('happy.jpg')
                faceHeight, faceWdith  = emotFace.shape[:2]
                propX = w / faceWdith
                propY = h / faceHeight
                emotFace  = cv2.resize(emotFace , (int(faceWdith  * propX), int(faceHeight * propY)),
                                       interpolation=cv2.INTER_AREA)
                maskArray = np.array(
                    [[74 * propX, 305 * propY], [94 * propX, 353 * propY], [141 * propX, 377 * propY],
                     [204 * propX, 385 * propY],[239 * propX, 377 * propY],[283 * propX, 319 * propY],[253 * propX, 297 * propY],[181 * propX, 304 * propY],[114 * propX, 291 * propY]], np.int32)
                mask = np.zeros((h, w), emotFace.dtype)
                cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                self.croppedImg = cv2.seamlessClone(emotFace , self.croppedImg, mask,(int(self.mouthValues[0]), int(self.mouthValues[1])),cv2.NORMAL_CLONE)
                if self.updateEyebrows.get() == 1:
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[195 * propX, 82 * propY], [195 * propX, 141 * propY], [323 * propX, 124 * propY],
                         [366 * propX, 139 * propY], [351 * propX, 75 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.rightEyeBrowValues[0]),int(self.rightEyeBrowValues[1])),cv2.NORMAL_CLONE)
                    mask = np.zeros((h,w),emotFace.dtype)
                    maskArray = np.array(
                        [[23 * propX, 135 * propY], [70 * propX, 116 * propY], [168 * propX, 143 * propY],
                         [165 * propX, 79 * propY], [48 * propX, 66 * propY], [14 * propX, 101 * propY]], np.int32)
                    cv2.fillPoly(mask, [maskArray], (255, 255, 255))
                    self.croppedImg = cv2.seamlessClone(emotFace ,self.croppedImg,mask,(int(self.leftEyeBrowValues[0]),int(self.leftEyeBrowValues[1])),cv2.NORMAL_CLONE)
            self.drawSubImage()
            self.updateAfterImg()
        else:
            messagebox.showerror("No file open", 'Please open a file before attempting to transform a face')


    def checkResolution(self): #As there's no resolution update call manually check resolution and if updated redraw UI
        str = root.winfo_geometry()
        dims = re.findall(r'\d+',str)
        if(dims[0] != '1' and dims[1] != '1'):
            if (int(dims[0]) != self.windowWidth or int(dims[1]) != self.windowHeight):
                    self.windowWidth = int(dims[0])
                    self.windowHeight = int(dims[1])
                    self.drawRectangles()
                    self.drawSubImage()
    def clear(self): #Reset face to before image
        self.modified = False
        self.emotionallyPredicted = False
        self.drawSubImage()

    def clearChangesPressed(self): #Calls clear method but allows it to be called multiple ways
        if self.modified == True:
            self.clear()
        else:
            tk.messagebox.showerror("Can't clear","The image is the same as the original, there are no changes to clear")

    def exitApp(self): #Kills root and exits code
        global root
        root.destroy()
        exit()

    def checkboxTick(self): #As booleans controlled in checkboxes only needs to be redrawn and handled in method
        self.drawSubImage()

    def undoBtnPress(self): #Resets image to state as seen in undo - sets up redo button
        if self.undoImage is not None:
            self.redoImage = self.sourceImg.copy()
            self.sourceImg = self.undoImage.copy()
            self.undoImage = None
            self.emotionallyPredicted = False
            self.drawRectangles()
            self.drawSubImage()
        else:
            tk.messagebox.showerror("Nothing to undo","No changes committed to main image to be undone")
    def loadUserGuide(self): #Opens web link to a video tutorial on using software
        webbrowser.open("https://youtu.be/4inEahRSWQ4")

    def redoBtnPress(self): #Update main image and sub images accordingly - set up undo button
        if self.redoImage is not None:
            self.undoImage = self.sourceImg.copy()
            self.sourceImg = self.redoImage.copy()
            self.redoImage = None
            self.emotionallyPredicted = False
            self.drawRectangles()
            self.drawSubImage()
        else:
            tk.messagebox.showerror("Nothing to redo","No changes committed to main image have been undone")
    def updateEyebrowRegion(self): #As flag is set in checkbox this merely alerts users that if the image is modified it won't currently update on tick
        if self.modified == True:
            tk.messagebox.showerror("Eyebrow update","Eyebrows on currently modified image will not change but will on future images")

    def __init__(self): #setup object so that processing can begin
        global result
        self.emotions = ["Anger","Disgust","Fear","Happy","Sad","Suprise","Neutral"]
        self.windowHeight = 1080
        self.windowWidth = 1920
        self.selectedFace = 0;
        self.emotionalPred = emotionPredictor()
        self.keypointsPred = keypointModel()
        self.updateEyebrows = tk.IntVar(value = 1)
        self.drawKeypointsVar = tk.IntVar()
        self.drawAngleVar = tk.IntVar()
        self.filePath = ''
        self.capFrames = True
        self.modified = False
        self.popupCamWindow = tk.Toplevel();
        self.popupCamWindow.destroy()
        tmpImg = cv2.imread('default.jpg')
        detector = MTCNN();
        tmpResult = detector.detect_faces(tmpImg)
        tmpFaces = len(tmpResult)
        self.sourceImg = tmpImg.copy()
        self.webCamImage = tmpImg.copy()
        self.mainImage = self.sourceImg.copy()
        self.result = tmpResult
        self.numFaces = len(self.result)
        self.checkAndUpdateSubImages()
        try: #Open camera briefly so that cap is initialised and can be checked if open
            self.cap = cv2.VideoCapture(0)
            self.cap.release()
        except:
            tk.messagebox.showerror("No webcam was found","No webcam was found, please insert one to enable image capture features")
        self.poll() #Start timing code



class emotionPredictor: #Builds up emotion prediction neural network
    def buildEmotionModel(self): #Builds up emotional model and returns object
        emotionModel = Sequential()
        emotionModel.add(BatchNormalization(input_shape=(48, 48, 1)))
        emotionModel.add(Convolution2D(96, 4, 4, border_mode='same', init='he_normal'))
        emotionModel.add(Activation('relu'))
        emotionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        emotionModel.add(Conv2D(128, 2, 2, init='he_normal', activation='relu'))
        emotionModel.add(Conv2D(128, 2, 2, init='he_normal', activation='relu'))
        emotionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        emotionModel.add(Conv2D(128, 3, 3, init='he_normal', activation='relu'))
        emotionModel.add(Conv2D(128, 3, 3, init='he_normal', activation='relu'))
        emotionModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        emotionModel.add(GlobalAveragePooling2D());
        emotionModel.add(Dense(7, activation='softmax', name='predictions'))
        adams = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        emotionModel.compile(optimizer=adams, loss='categorical_crossentropy', metrics=['accuracy'])
        return emotionModel

    def loadEmotionModel(self): #Initialises emotional model
        model = self.buildEmotionModel()
        model.load_weights('emotion_model.h5')
        return model

    def emotionalPredict(self,img): #Scales image to correct colour schema then provides predictions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = (gray / 255)
        roi_rescaled = cv2.resize(gray, (48, 48))
        predictions = self.predictor.predict(roi_rescaled[np.newaxis, :, :, np.newaxis])
        return predictions

    def __init__(self): #Initialises model by building object
        self.predictor = self.loadEmotionModel()

class keypointModel: #Object for building keypoints model and providing predictions
    def buildKeypointModel(self):#Builds keypoints model and returns it as an object
        keypointModel = Sequential()
        keypointModel.add(BatchNormalization(input_shape=(96, 96, 1)))
        keypointModel.add(Convolution2D(96, 4, 4, border_mode='same', init='he_normal', activation='relu'))
        keypointModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        keypointModel.add(Conv2D(128, 2, 2, init='he_normal', activation='relu'))
        keypointModel.add(Conv2D(128, 2, 2, init='he_normal', activation='relu'))
        keypointModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        keypointModel.add(Conv2D(128, 3, 3, init='he_normal', activation='relu'))
        keypointModel.add(Conv2D(128, 3, 3, init='he_normal', activation='relu'))
        keypointModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        keypointModel.add(Conv2D(256, 5, 5, init='he_normal', activation='relu'))
        keypointModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
        keypointModel.add(GlobalAveragePooling2D());
        keypointModel.add(Dense(256, activation='relu'))
        keypointModel.add(Dense(128, activation='relu'))
        keypointModel.add(Dense(64, activation='relu'))
        keypointModel.add(Dense(30))
        adams = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        keypointModel.compile(optimizer=adams, loss='mse', metrics=['accuracy'])
        return keypointModel

    def loadKeypointModel(self): #Calls for model to be built and specifies weights
        model = self.buildKeypointModel()
        model.load_weights('keypoints_model.h5')
        return model

    def keypointPredict(self,img): #Performs greyscaling on image and provides predictions as a 15x2 array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = (gray / 255)
        roi_rescaled = cv2.resize(gray, (96, 96))
        predictions = self.predictor.predict(roi_rescaled[np.newaxis, :, :, np.newaxis])
        xyPredictions = (predictions).reshape(15, 2)
        return xyPredictions

    def __init__(self): #Initialises and builds model
        self.predictor = self.loadKeypointModel()


def askQuit(): #Ensures users do not accidentally quit application
    if tk.messagebox.askokcancel("Quit", "Are you sure you want to quit? Unsaved changes will be lost."):
        root.quit()
        exit()

global root #Defines a global UI
root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", askQuit)
root.title('Emotion transformation software')
root.geometry("1920x1079") #Sets initial window size
if sys.platform == "darwin": #Sets up OS-X command keys
    CmdKey="Command-"
    root.bind('<Command-o>', lambda event: app.openFile())
    root.bind('<Command-s>', lambda event: app.saveFile())
    root.bind('<Command-r>', lambda event: app.clearChangesPressed())
    root.bind('<Command-Shift-s>', lambda event: app.saveAsFile())
    root.bind('<Command-w>', lambda event: app.webcamCap())
    root.bind('<Command-z>', lambda event: app.undoBtnPress())
    root.bind('<Command-y>', lambda event: app.redoBtnPress())

    root.minsize(1200, 670) #Sets min size based on how OS renders text
else:
    CmdKey="Control-" #Sets up command keys for linux / windows
    root.bind('<Control-o>', lambda event: app.openFile())
    root.bind('<Control-s>', lambda event: app.saveFile())
    root.bind('<Control-r>', lambda event: app.clearChangesPressed())
    root.bind('<Control-Shift-s>', lambda event: app.saveAsFile())
    root.bind('<Control-w>', lambda event: app.webcamCap())
    root.bind('<Control-z>', lambda event: app.undoBtnPress())
    root.bind('<Control-y>', lambda event: app.redoBtnPress())
    root.minsize(720,480)
app = mainApp()
menubar = tk.Menu(root) #Sets up menus
filemenu = tk.Menu(menubar, tearoff=0)
editmenu = tk.Menu(menubar,tearoff=0)
helpmenu = tk.Menu(menubar,tearoff=0)
settingsmenu = tk.Menu(menubar,tearoff=0)
webcammenu = tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)
menubar.add_cascade(label="Edit",menu=editmenu)
menubar.add_cascade(label="Settings",menu=settingsmenu)
menubar.add_cascade(label="Webcam",menu=webcammenu)
menubar.add_cascade(label="Help",menu=helpmenu)
mainImageOptionsmenu = tk.Menu(editmenu,tearoff=0)
subImageOptionsmenu = tk.Menu(editmenu,tearoff=0)
filemenu.add_command(label="Open", command=app.openFile,accelerator=CmdKey+'O') #Uses command key so that guide for hotkeys is OS relevant
filemenu.add_command(label="Save", command=app.saveFile,accelerator=CmdKey+'S')
filemenu.add_command(label="Save as", command=app.saveAsFile,accelerator=CmdKey+'Shift-'+'S')
filemenu.add_separator()
filemenu.add_command(label="Exit", command=askQuit)
editmenu.add_cascade(label="Main Image options:",menu=mainImageOptionsmenu)
editmenu.add_separator()
editmenu.add_cascade(label="Sub-Image options:",menu=subImageOptionsmenu)
mainImageOptionsmenu.add_command(label="Undo",command=app.undoBtnPress,accelerator=CmdKey+'Z')
mainImageOptionsmenu.add_command(label="Redo",command=app.redoBtnPress,accelerator=CmdKey+'Y')
subImageOptionsmenu.add_command(label="Reset image",command=app.clearChangesPressed,accelerator=CmdKey+'R')
webcammenu.add_command(label="Open webcam",command=app.webcamCap,accelerator=CmdKey+'W')
settingsmenu.add_checkbutton(label="Display keypoints",variable=app.drawKeypointsVar,command=app.checkboxTick)
settingsmenu.add_checkbutton(label="Display face angle",variable=app.drawAngleVar,command=app.checkboxTick)
settingsmenu.add_checkbutton(label="Update eyebrow region",variable=app.updateEyebrows,command=app.updateEyebrowRegion)
helpmenu.add_command(label="User guide",command=app.loadUserGuide)
mainFaceImage = tk.LabelFrame(root, text="Source Photo - File not loaded")
fileOptions = tk.LabelFrame(root,text="File options")
mainButtonGroup = tk.LabelFrame(root,text="Image options")
croppedFaceImage = tk.LabelFrame(root,text="Selected face")
croppedModifiedFaceImage = tk.LabelFrame(root, text="Altered face")
oldFaceLabelFrameButtons = tk.LabelFrame(root,text="Transformation options")
oldEmotions = tk.LabelFrame(root,text="Original image emotions")
newEmotions = tk.LabelFrame(root,text="Modified image emotions")
emotionButtons = tk.LabelFrame(oldFaceLabelFrameButtons,text="Modified image emotions")
modifiedFileOptions = tk.LabelFrame(oldFaceLabelFrameButtons, text="File Options")
imageStats = tk.LabelFrame(root,text="Image details")
photo = ImageTk.PhotoImage(file="default.jpg")
label1 = tk.Label(mainFaceImage, image=photo)
label1.image = photo
photo = ImageTk.PhotoImage(file="default.jpg")
label2 = tk.Label(croppedFaceImage, image=photo)
label2.image = photo
label3 = tk.Label(croppedModifiedFaceImage, image=photo)
label3.image = photo
prevBtn = tk.Button(mainButtonGroup, text="Previous", command=app.prevFace)
nextBtn = tk.Button(mainButtonGroup, text="Next", command=app.nextFace)
webcamBtn = tk.Button(fileOptions, text="Open Webcam", command=app.webcamCap)
openFile = tk.Button(fileOptions, text="Open File", command=app.openFile)
save = tk.Button(fileOptions, text="Save", command=app.saveFile)
saveAs = tk.Button(fileOptions, text="Save As", command=app.saveAsFile)
angerBtn = tk.Button(emotionButtons, text="Anger", command=app.angryEmotionTransformation)
disgustBtn = tk.Button(emotionButtons, text="Disgust", command=app.disgustEmotionTransformation)
scaredBtn = tk.Button(emotionButtons, text="Scared", command=app.scaredEmotionTransformation)
sadBtn = tk.Button(emotionButtons, text="Sad", command=app.sadEmotionTransformation)
happyBtn = tk.Button(emotionButtons, text="Happy", command=app.happyEmotionTransformation)
suprisedBtn = tk.Button(emotionButtons, text="Surprised", command=app.surprisedEmotionTransformation)
neutralBtn = tk.Button(emotionButtons, text=" Neutral ", command=app.neutralEmotionTransformation)
sendToMain = tk.Button(modifiedFileOptions, text="Update Image", command=app.drawOntoMainImage)
undoSub = tk.Button(mainButtonGroup, text="Undo", command=app.undoBtnPress)
redoSub = tk.Button(mainButtonGroup, text="Redo", command=app.redoBtnPress)
clearChanges = tk.Button(modifiedFileOptions,text="Reset Image",command=app.clearChangesPressed)
updateEyebrowBtn = tk.Checkbutton(modifiedFileOptions,text="Update eyebrow region",variable=app.updateEyebrows,command=app.updateEyebrowRegion)
suprisedOld = tk.Label(oldEmotions,text="Suprised:")
predictedOld = tk.Label(oldEmotions, text="Emotion:")
angerOld = tk.Label(oldEmotions, text="Anger:")
disgustOld = tk.Label(oldEmotions, text="Disgust:")
scaredOld = tk.Label(oldEmotions, text="Scared:")
happyOld = tk.Label(oldEmotions, text="Happy:")
sadOld = tk.Label(oldEmotions, text="Sad:")
neutralOld = tk.Label(oldEmotions, text="Neutral:")
predictedOldPct = tk.Label(oldEmotions, text="Null")
angerPct = tk.Label(oldEmotions, text="0%")
disgustPct = tk.Label(oldEmotions, text="0%")
scaredPct = tk.Label(oldEmotions, text="0%")
happyPct = tk.Label(oldEmotions, text="0%")
sadPct = tk.Label(oldEmotions, text="0%")
suprisedPct = tk.Label(oldEmotions, text="0%")
neutralPct = tk.Label(oldEmotions, text="0%")
predictedNewPct = tk.Label(newEmotions, text="Null")
angerPctNew = tk.Label(newEmotions, text="0%")
disgustPctNew = tk.Label(newEmotions, text="0%")
scaredPctNew = tk.Label(newEmotions, text="0%")
happyPctNew = tk.Label(newEmotions, text="0%")
sadPctNew = tk.Label(newEmotions, text="0%")
suprisedPctNew = tk.Label(newEmotions, text="0%")
neutralPctNew = tk.Label(newEmotions, text="0%")
predictedNew = tk.Label(newEmotions, text="Emotion:")
angerNew = tk.Label(newEmotions, text="Anger:")
disgustNew = tk.Label(newEmotions, text="Disgust:")
scaredNew = tk.Label(newEmotions, text="Scared:")
happyNew = tk.Label(newEmotions, text="Happy:")
sadNew = tk.Label(newEmotions, text="Sad:")
suprisedNew = tk.Label(newEmotions, text="Suprised:")
neutralNew = tk.Label(newEmotions, text="Neutral:")
faceGender = tk.Label(imageStats,text="Predicted gender:")
faceGenderTxt = tk.Label(imageStats,text="Null")
numFacesDesc = tk.Label(imageStats,text="Number of faces:")
numFacesTxt = tk.Label(imageStats,text="Null")
displayKeypoints = tk.Checkbutton(imageStats,text="Display keypoints",variable=app.drawKeypointsVar,command=app.checkboxTick)
displayAngleEst = tk.Checkbutton(imageStats,text="Display face angle",variable=app.drawAngleVar,command=app.checkboxTick)
#Sets up grid elements so UI looks nice
mainFaceImage.grid(row=0, column=0, columnspan=7, rowspan=5)
fileOptions.grid(row=5, column=0, columnspan=3)
mainButtonGroup.grid(row=5, column=3, columnspan=4)
croppedFaceImage.grid(row=0, column=7, columnspan=5, sticky=tk.N)
croppedModifiedFaceImage.grid(row=0, column=12, columnspan=5, sticky=tk.N)
imageStats.grid(row=1, column=7, columnspan=10, sticky=tk.N + tk.W + tk.E)
oldFaceLabelFrameButtons.grid(row=2, column=7, columnspan=10, sticky=tk.N + tk.W + tk.E)
oldEmotions.grid(row=3, column=7, columnspan=5, sticky=tk.N + tk.W + tk.E)
newEmotions.grid(row=3, column=12, columnspan=5, sticky=tk.N + tk.W + tk.E)
emotionButtons.grid(row=0, column=0, columnspan=5, rowspan=2, sticky=tk.N + tk.W + tk.E)
modifiedFileOptions.grid(row=0, column=5, columnspan=5, rowspan=1, sticky=tk.N + tk.W + tk.E)
label1.grid(row=0, column=0, columnspan=5, rowspan=5, padx=5, pady=5)
label2.grid(row=1, column=0, columnspan=5, padx=2, pady=2, sticky=tk.N)
label3.grid(row=1, column=0, columnspan=5, padx=2, pady=2, sticky=tk.N)
prevBtn.grid(row=0, column=0)
nextBtn.grid(row=0, column=1)
openFile.grid(row=0, column=0)
save.grid(row=0, column=1)
saveAs.grid(row=0, column=2)
webcamBtn.grid(row=0, column=3)
angerBtn.grid(row=0, column=0, columnspan=2)
disgustBtn.grid(row=0, column=2, columnspan=2)
scaredBtn.grid(row=0, column=4, columnspan=2)
sadBtn.grid(row=0, column=6, columnspan=2)
happyBtn.grid(row=1, column=1, columnspan=2)
suprisedBtn.grid(row=1, column=3, columnspan=2)
neutralBtn.grid(row=1, column=5, columnspan=2)
sendToMain.grid(row=0, column=0)
updateEyebrowBtn.grid(row=1, column=0,columnspan = 2)
undoSub.grid(row=0, column=2)
redoSub.grid(row=0, column=3)
clearChanges.grid(row=0,column=1)
predictedOld.grid(row=0, column=0)
angerOld.grid(row=1, column=0, sticky=tk.W)
disgustOld.grid(row=2, column=0, sticky=tk.W)
scaredOld.grid(row=3, column=0, sticky=tk.W)
happyOld.grid(row=4, column=0, sticky=tk.W)
sadOld.grid(row=5, column=0, sticky=tk.W)
suprisedOld.grid(row=6, column=0, sticky=tk.W)
neutralOld.grid(row=7, column=0, sticky=tk.W)
predictedOldPct.grid(row=0, column=1)
angerPct.grid(row=1, column=1, sticky=tk.W)
disgustPct.grid(row=2, column=1, sticky=tk.W)
scaredPct.grid(row=3, column=1, sticky=tk.W)
happyPct.grid(row=4, column=1, sticky=tk.W)
sadPct.grid(row=5, column=1, sticky=tk.W)
suprisedPct.grid(row=6, column=1, sticky=tk.W)
neutralPct.grid(row=7, column=1, sticky=tk.W)
predictedNewPct.grid(row=0, column=1, sticky=tk.W)
angerPctNew.grid(row=1, column=1, sticky=tk.W)
disgustPctNew.grid(row=2, column=1, sticky=tk.W)
scaredPctNew.grid(row=3, column=1, sticky=tk.W)
happyPctNew.grid(row=4, column=1, sticky=tk.W)
sadPctNew.grid(row=5, column=1, sticky=tk.W)
suprisedPctNew.grid(row=6, column=1, sticky=tk.W)
neutralPctNew.grid(row=7, column=1, sticky=tk.W)
predictedNew.grid(row=0, column=0, sticky=tk.W)
angerNew.grid(row=1, column=0, sticky=tk.W)
disgustNew.grid(row=2, column=0, sticky=tk.W)
scaredNew.grid(row=3, column=0, sticky=tk.W)
happyNew.grid(row=4, column=0, sticky=tk.W)
sadNew.grid(row=5, column=0, sticky=tk.W)
suprisedNew.grid(row=6, column=0, sticky=tk.W)
neutralNew.grid(row=7, column=0, sticky=tk.W)
faceGender.grid(row=0,column = 0,columnspan = 3)
faceGenderTxt.grid(row =0,column = 3,columnspan = 2)
numFacesDesc.grid(row=0,column=5,columnspan = 3)
numFacesTxt.grid(row=0,column=8,columnspan =2)
displayKeypoints.grid(row=1,column=0,columnspan=5)
displayAngleEst.grid(row=1,column=5,columnspan=5)
root.config(menu=menubar)
root.geometry("1920x1080")
root.mainloop()
exit()
