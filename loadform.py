import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import (QFileDialog, QLabel)
from PyQt5.QtGui import QPixmap, QPolygon
import cv2
from PyQt5 import Qt
from PyQt5.QtCore import QPoint

Form, _ = uic.loadUiType('./LoadForm.ui')

class Ui(QtWidgets.QDialog, Form):
    def __init__(self):
        super(Ui, self).__init__()
        self.initProperties()
        self.initEvents()
        self.label.mousePressEvent = self.get_pos
        self.massDots = []

        self.massLines = []

        self.massPolygon = []
        self.massPolygonZone1 = []
        self.massPolygonZone2 = []
        self.massPolygonZone3 = []
        self.massPolygonZone4 = []
        self.currentPolygon = self.massPolygon
        self.massPolygonCount = 0
        self.switchPenCount = 0

        
        



    def switchPen(self):
        
        if self.switchPenCount == 0:
            pen = Qt.QPen(Qt.QColor(255, 0, 0), 10)
            self.switchPenCount+=1
        elif self.switchPenCount == 1:
            pen = Qt.QPen(Qt.QColor(7, 16, 250), 10)
            self.switchPenCount+=1
        elif self.switchPenCount == 2:
            pen = Qt.QPen(Qt.QColor(7, 250, 104), 10)
            self.switchPenCount+=1
        elif self.switchPenCount == 3:
            pen = Qt.QPen(Qt.QColor(250, 209, 7), 10)
            self.switchPenCount+=1
        elif self.switchPenCount == 4:
            pen = Qt.QPen(Qt.QColor(0, 236, 252), 10)
            self.switchPenCount=0
        return pen

    def initProperties(self):
        self.setupUi(self)
        self.setWindowTitle("Object Detection")
        self.label = QLabel()

    def initEvents(self):
        self.BrowseButton.clicked.connect(self.GetFilePath)
        self.EnterButton.clicked.connect(self.DrawFirstFrame)
        self.startObjectDetection.clicked.connect(self.startDetection)
        self.poligonButton.clicked.connect(self.drawPolygon)
             

    def drawPolygon(self):
        
        qp = Qt.QPainter()
        qp.begin(self.label.pixmap())
        qp.setPen(self.switchPen())

        coords = []
        for i in self.currentPolygon:
            for j in i:
                coords.append(j)
   
        coords = QPolygon(coords)

        qp.drawPolygon(coords)
        self.label.update()
        self.massPolygonCount+=1

        if self.massPolygonCount == 1:
            
            self.currentPolygon = self.massPolygonZone1

        elif self.massPolygonCount == 2:
            
            self.currentPolygon = self.massPolygonZone2

        elif self.massPolygonCount == 3:
            
            self.currentPolygon = self.massPolygonZone3

        elif self.massPolygonCount == 4:
            
            self.currentPolygon = self.massPolygonZone4
        elif self.massPolygonCount == 5:
            self.currentPolygon = self.currentPolygon
            print('self.massPolygon = ', self.massPolygon)
            print('self.massPolygonZone1 = ', self.massPolygonZone1)
            print('self.massPolygonZone2 = ', self.massPolygonZone2)
            print('self.massPolygonZone3 = ', self.massPolygonZone3)
            print('self.massPolygonZone4 = ', self.massPolygonZone4)
            print('self.massLines', self.massLines)
            self.poligonButton.hide()


    def DrawLine(self):
        
        
        qp = Qt.QPainter()
        qp.begin(self.label.pixmap())
        qp.setPen(self.switchPen())
        
        qp.drawLine(self.massDots[0][0], self.massDots[0][1], self.massDots[1][0], self.massDots[1][1])
        self.label.update()
        print(self.massLines)

    def GetFilePath(self):
        self.fpath = QFileDialog.getOpenFileName(self, 'Open file', './')
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        self.lineEdit.setText(str(self.fpath[0]))

    def DrawFirstFrame(self):
        pathTo = self.lineEdit.text()
        cap = cv2.VideoCapture(pathTo)
        _, frame = cap.read()
        width = 960  # int(img.shape[1] * scale_percent / 100)
        height = 960  # int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("frame%d.jpg" % 1, img)
        self.label.setPixmap(QPixmap("./frame%d.jpg" % 1))
        self.FirstFrame.setWidget(self.label)
        self.massDots = []
        self.massLines = []
        self.massPoligon = []

    def drawPoints(self, pos):
        print("Call draw points")
        pen = Qt.QPen(Qt.QColor(255, 255, 255), 10)
        qp = Qt.QPainter()
        qp.begin(self.label.pixmap())
        qp.setPen(pen)
        qp.drawPoint(pos.x(), pos.y())
        self.label.update()

    def get_pos(self, event):
        
        X = event.pos().x()
        y = event.pos().y()
        
        if len(self.massLines) == 8:
            self.currentPolygon.append((X,y))

        print('x,y = ', X, y)
        self.drawPoints(event.pos())
        self.massDots.append((X,y))
        
        
        if len(self.massDots) == 2 and len(self.massLines) < 8:
            self.massLines.append(self.massDots[0])
            self.massLines.append(self.massDots[1])
            self.DrawLine()
            self.massDots.clear()
            


    def startDetection(self):
        from vehicleCounting import DrowVehicles
        drowVehicles = DrowVehicles(videoPath=self.fpath, massLines = self.massLines, massPolygon = self.massPolygon, massPolygonZone1 = self.massPolygonZone1, massPolygonZone2 = self.massPolygonZone2, massPolygonZone3 = self.massPolygonZone3, massPolygonZone4 = self.massPolygonZone4)
        
            

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    # show window dialog
    loadForm = Ui()
    loadForm.show()
    sys.exit(app.exec_())