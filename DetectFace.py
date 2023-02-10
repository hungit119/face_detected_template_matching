import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import math
import numpy as np
import cv2
import random
import os 
from matplotlib import pyplot as plt
from matplotlib.image import imread
TRAIN_IMG_FOLDER = 'images/ORL/hung/'
TEST_IMG_FOLDER = 'images/ORL/Test/'

class Ui_FormDetectFace(object):
    def setupUi(self, Form):
        self.magnitude = 1
        Form.setObjectName("Form")
        Form.resize(1217, 821)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(40, 10, 271, 61))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.btnChooseImage = QtWidgets.QPushButton(Form)
        self.btnChooseImage.setGeometry(QtCore.QRect(40, 80, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.btnChooseImage.setFont(font)
        self.btnChooseImage.setObjectName("btnChooseImage")
        self.btnChooseImage.clicked.connect(self.linkTo)
        self.lbShowImage = QtWidgets.QLabel(Form)
        self.lbShowImage.setGeometry(QtCore.QRect(290, 140, 901, 661))
        self.lbShowImage.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.lbShowImage.setAcceptDrops(False)
        self.lbShowImage.setFrameShape(QtWidgets.QFrame.Panel)
        self.lbShowImage.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbShowImage.setText("")
        self.lbShowImage.setPixmap(QtGui.QPixmap("cach-them-user-va-xoa-user-khoi-group-linux-1024x698.jpg"))
        self.lbShowImage.setScaledContents(False)
        self.lbShowImage.setObjectName("lbShowImage")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(10, 220, 131, 241))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.btnSubScale = QtWidgets.QPushButton(self.frame)
        self.btnSubScale.setGeometry(QtCore.QRect(30, 140, 61, 61))
        self.btnSubScale.clicked.connect(self.SubScaleImage)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.btnSubScale.setFont(font)
        self.btnSubScale.setObjectName("btnSubScale")
        self.btnPlusScale = QtWidgets.QPushButton(self.frame)
        self.btnPlusScale.setGeometry(QtCore.QRect(30, 50, 61, 61))
        self.btnPlusScale.clicked.connect(self.PlusScaleImage)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.btnPlusScale.setFont(font)
        self.btnPlusScale.setObjectName("")
        self.frame_2 = QtWidgets.QFrame(Form)
        self.frame_2.setGeometry(QtCore.QRect(150, 220, 131, 241))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.txtNewWidth = QtWidgets.QLineEdit(self.frame_2)
        self.txtNewWidth.setGeometry(QtCore.QRect(10, 80, 113, 22))
        self.txtNewWidth.setObjectName("txtNewWidth")
        self.txtNewHeight = QtWidgets.QLineEdit(self.frame_2)
        self.txtNewHeight.setGeometry(QtCore.QRect(10, 130, 113, 22))
        self.txtNewHeight.setObjectName("txtNewHeight")
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setGeometry(QtCore.QRect(10, 60, 55, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setGeometry(QtCore.QRect(10, 110, 55, 16))
        self.label_6.setObjectName("label_6")
        self.btnResizeImage = QtWidgets.QPushButton(self.frame_2)
        self.btnResizeImage.setGeometry(QtCore.QRect(20, 170, 93, 28))
        self.btnResizeImage.setObjectName("btnResizeImage")
        self.txtShowPathImage = QtWidgets.QLineEdit(Form)
        self.txtShowPathImage.setGeometry(QtCore.QRect(200, 80, 991, 41))
        self.txtShowPathImage.setObjectName("txtShowPathImage")
        self.btnFaceDetect = QtWidgets.QPushButton(Form)
        self.btnFaceDetect.setGeometry(QtCore.QRect(40, 560, 211, 181))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.btnFaceDetect.setFont(font)
        self.btnFaceDetect.setObjectName("btnFaceDetect")
        self.btnFaceDetect.clicked.connect(self.onClickBtnDetected)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Face Recognition"))
        self.btnChooseImage.setText(_translate("Form", "Choose Image"))
        self.label_3.setText(_translate("Form", "Scale Image"))
        self.btnPlusScale.setText(_translate("Form", "+"))
        self.btnSubScale.setText(_translate("Form", "-"))
        self.label_4.setText(_translate("Form", "Resize Image"))
        self.label_5.setText(_translate("Form", "Width"))
        self.label_6.setText(_translate("Form", "Height"))
        self.btnResizeImage.setText(_translate("Form", "Resize"))
        self.btnResizeImage.clicked.connect(self.reSizeImage)
        self.btnFaceDetect.setText(_translate("Form", "Face Detect"))
    def linkTo (self):
        self.link = QFileDialog.getOpenFileName(filter='*jpg *png')
        self.lbShowImage.setPixmap(QPixmap(self.link[0]))
        self.txtShowPathImage.setText(self.link[0])
    def onClickBtnDetected (self):
        if self.txtShowPathImage.text() == "":
            pass
        else:
            self.caculateMeanFace()
            self.pathInputImage = self.txtShowPathImage.text()
            self.arrayInputImage = cv2.imread(self.pathInputImage)
            self.grayArrayInputImage = cv2.cvtColor(self.arrayInputImage,cv2.COLOR_BGR2GRAY)
            if self.mean_face.dtype != self.grayArrayInputImage.dtype:
                self.mean_face = self.mean_face.astype(self.grayArrayInputImage.dtype)
            self.mean_face_template = self.mean_face.reshape(self.height,self.width)
            self.w,self.h = self.mean_face_template.shape[::-1]
            self.res = cv2.matchTemplate(self.grayArrayInputImage,self.mean_face_template,cv2.TM_CCOEFF_NORMED)
            self.threshold = 0.57
            self.locations = np.where(self.res >= self.threshold)
            for pt in zip(*self.locations[::-1]):
                cv2.rectangle(self.arrayInputImage, pt, (pt[0] + self.w, pt[1] + self.h), (0,255,0), 2)

            # Display the resulting image
            cv2.imshow('Detected faces',self.arrayInputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def caculateMeanFace(self):
        self.training_set_files = os.listdir(TRAIN_IMG_FOLDER)
        self.test_set_files = os.listdir(TEST_IMG_FOLDER)

        self.width = 128
        self.height = 128
        self.training_id_file = set([f.split("_")[0] for f in self.training_set_files])
        self.test_id_file = set([f.split("_")[0] for f in self.test_set_files])
        # Hiển thị tập ảnh huấn luyện và tập ảnh thử nghiệm
        print("Training Images:")
        self.train_image_names = os.listdir(TRAIN_IMG_FOLDER)
        self.training_tensor = np.ndarray(shape=(len(self.train_image_names),self.height * self.width),dtype=np.float64)
        for i in range(len(self.train_image_names)):
            img = cv2.resize(plt.imread(TRAIN_IMG_FOLDER + self.train_image_names[i]),(self.height,self.width))
            self.training_tensor[i,:] = np.array(img, dtype='float64').flatten()
            # Tính khuôn mặt trung bình
        self.mean_face = np.zeros((1,self.height*self.width))
        for i in self.training_tensor:
            self.mean_face = np.add(self.mean_face,i)

        self.mean_face = np.divide(self.mean_face,float(len(self.train_image_names))).flatten()
        plt.imshow(self.mean_face.reshape(self.height,self.width),cmap="gray")
        plt.tick_params(labelleft="off",labelbottom="off",bottom="off",top="off",right="off",left="off",which="both")
        plt.show()
    def PlusScaleImage (self):
        imageCur = cv2.imread(self.link[0],1)
        self.ScaleImage(imageCur,self.magnitude + 0.5)
    def SubScaleImage (self):
        imageCur = cv2.imread(self.link[0],1)
        self.ScaleImage(imageCur,self.magnitude - 0.5)
    def ScaleImage(self,image,magnitude):
        width,height,chan = image.shape
        print(width,height,chan)
        scale_x = magnitude
        scale_y = magnitude
        w1 = width
        h1 = height
        w2 = int(math.floor(w1*scale_x))
        h2 = int(math.floor(h1*scale_y))
        img_bl = np.empty((w2,h2,chan),dtype=np.uint8)
        x_ratio = float(1/float(scale_x))
        y_ratio = float(1/float(scale_y))
        for i in range(0,w2):
            for j in range (0,h2):
                orir = i * x_ratio
                oric = j * y_ratio
                img_bl[i,j] = self.GetbilinearPixel(image,oric,orir)
        self.path = os.path.join("images","scalePlus")
        randomImageNumber = random.sample(range(1000),1)[0]
        cv2.imwrite('% s/% s.png' % (self.path, randomImageNumber), img_bl)
        print('% s/% s.png' % (self.path, randomImageNumber))
        self.lbShowImage.setPixmap(QPixmap('% s/% s.png' % (self.path, randomImageNumber)))
        self.txtShowPathImage.setText('% s/% s.png' % (self.path, randomImageNumber))
        self.link = ['% s/% s.png' % (self.path, randomImageNumber)]
    def GetbilinearPixel (self,imArr,posX,posY):
        out = []
        modXi = int(posX)
        modYi = int(posY)
        modXf = posX - modXi
        modYf = posY - modYi
        modXiPlusOneLim = min(modXi+1,imArr.shape[1]-1)
        modYiPlusOneLim = min(modYi+1,imArr.shape[0]-1)
        for chan in range(imArr.shape[2]):
                bl = imArr[modYi, modXi, chan]
                br = imArr[modYi, modXiPlusOneLim, chan]
                tl = imArr[modYiPlusOneLim, modXi, chan]
                tr = imArr[modYiPlusOneLim, modXiPlusOneLim, chan]
                b = modXf * br + (1. - modXf) * bl
                t = modXf * tr + (1. - modXf) * tl
                pxf = modYf * t + (1. - modYf) * b
                out.append(int(pxf+0.5))
        return out
    def reSizeImage (self):
        ImageCurr = cv2.imread(self.link[0],1)
        width = int(self.txtNewWidth.text())
        height = int(self.txtNewHeight.text())
        img_height,img_width,chan = ImageCurr.shape
        resized = np.empty((height, width,chan),dtype=np.uint8)

        x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
        y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

        for i in range(height):
            for j in range(width):
                x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

                x_weight = (x_ratio * j) - x_l
                y_weight = (y_ratio * i) - y_l

                a = ImageCurr[y_l, x_l]
                b = ImageCurr[y_l, x_h]
                c = ImageCurr[y_h, x_l]
                d = ImageCurr[y_h, x_h]

                pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                resized[i][j] = pixel
        self.pathResize = os.path.join("images","resizeds")
        if not os.path.isdir(self.pathResize):
            os.mkdir(self.pathResize)
        randomImageNumber = random.sample(range(1000),1)[0]
        cv2.imwrite('% s/% s.png' % (self.pathResize, randomImageNumber), resized)
        print('% s/% s.png' % (self.pathResize, randomImageNumber))
        self.lbShowImage.setPixmap(QPixmap('% s/% s.png' % (self.pathResize, randomImageNumber)))
        self.txtShowPathImage.setText('% s/% s.png' % (self.pathResize, randomImageNumber))
        self.link = ['% s/% s.png' % (self.pathResize, randomImageNumber)]

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_FormDetectFace()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
