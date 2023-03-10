# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AddFace.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import cv2
import os
import random
from PyQt5 import QtCore, QtGui, QtWidgets
haar_file = 'cascades/data/haarcascade_frontalface_default.xml'
datasets = "images/ORL"
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1200, 800)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(50, 30, 401, 91))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lbNameFace = QtWidgets.QLabel(Form)
        self.lbNameFace.setFont(font)
        self.lbNameFace.setText("Name Face:")
        self.lbNameFace.setGeometry(QtCore.QRect(240, 120, 321, 31))

        self.txtNameNewFace = QtWidgets.QLineEdit(Form)
        self.txtNameNewFace.setGeometry(QtCore.QRect(400, 120, 321, 31))
        self.txtNameNewFace.setObjectName("txtNameNewFace")
        self.btnNameNewFace = QtWidgets.QPushButton(Form)
        self.btnNameNewFace.setGeometry(QtCore.QRect(750, 120, 200, 31))
        self.btnNameNewFace.setObjectName("btnNameNewFace")
        self.btnNameNewFace.setText("Set Name Face")
        self.btnNameNewFace.setFont(font)
        self.btnNameNewFace.clicked.connect(self.setNameFace)
        self.path = ''
        self.btnOpenWebcam = QtWidgets.QPushButton(Form)
        self.btnOpenWebcam.setGeometry(QtCore.QRect(480, 170, 201, 71))
        self.btnOpenWebcam.setFont(font)
        self.btnOpenWebcam.setObjectName("btnOpenWebcam")
        self.btnOpenWebcam.clicked.connect(self.OpenWebCam)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.subdata = ""
    def setNameFace (self):
        self.valueNameFace = self.txtNameNewFace.text()
        self.path = os.path.join(datasets,self.valueNameFace)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        print(self.path)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Add new face"))
        self.btnOpenWebcam.setText(_translate("Form", "Get Image"))
    def OpenWebCam (self):
        self.Image_Input = []
        camera = cv2.VideoCapture(0)
        for i in range(5):
            return_value , image = camera.read()
            self.Image_Input.append(image)
        del(camera)
        cv2.imshow("Hung",self.Image_Input[len(self.Image_Input) - 1])
        cv2.waitKey(0)
        self.DetectedFaceLocation(self.Image_Input[len(self.Image_Input) - 1])
    def DetectedFaceLocation(self,image):
        face_cascase = cv2.CascadeClassifier(haar_file)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face = face_cascase.detectMultiScale(gray,1.3,4)
        if face == () :
            print("No face detected")
            return
        x = face[0][0]
        y = face[0][1]
        w = face[0][2]
        h = face[0][3]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h,x:x+w]
        face_resize = cv2.resize(face,(130,100))
        cv2.imwrite('% s/% s_% s_fa.jpg' % (self.path, random.sample(range(100000),10000)[0], random.sample(range(100000),10000)[0]), face_resize)
        cv2.imshow("Image datasets",face_resize)
        cv2.waitKey(0)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
