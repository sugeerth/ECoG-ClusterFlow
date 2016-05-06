import sys
from PySide import QtGui, QtCore

""" TODO : Adding selectable boxes """
syllables = ['baa','boo','daa','doo','paa','poo']
Group = QtGui.QButtonGroup()

class syllablePic(QtGui.QFrame):
    syllabeSignal = QtCore.Signal(int)

    def __init__(self, syllableNo, parent=None):
        super(syllablePic, self).__init__(parent)        
        self.syllableNo = int(syllableNo)
        # self.syllabeSignal = syllabeSignal
        self.button = QtGui.QRadioButton("", self)
        self.button.clicked.connect(self.ButtonClicked)

        Group.addButton(self.button)
        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(2)
        hbox.addWidget(self.button)
        self.setLayout(hbox)

    def paintEvent(self, event):
            qp = QtGui.QPainter()
            qp.begin(self)
            self.drawText(event, qp)
            qp.end()

    def drawText(self, event, qp):
            qp.setPen(QtGui.QColor(255, 255, 255))
            qp.setFont(QtGui.QFont('Decorative', 40))
            qp.drawText(event.rect(), QtCore.Qt.AlignCenter, syllables[self.syllableNo-1])       

    def ButtonClicked(self):
        self.update()
        self.syllabeSignal.emit(self.syllableNo)

    def mousePressEvent(self, event):
        self.button.setChecked(True)
        self.update()
        self.syllabeSignal.emit(self.syllableNo)


class Syllable(QtGui.QWidget):

    selectedSyllableChanged = QtCore.Signal(int)

    def __init__(self):
        super(Syllable, self).__init__()
        
        self.initUI()
        
    def initUI(self):      
        self.square = [None]*7
        self.col = QtGui.QColor(1, 0, 0)       
        syllabeSignal = self.selectedSyllableChanged 

        for i in range(1,7):
            self.square[i] = syllablePic(i,self)
            self.square[i].setGeometry(30 + (i-1)*220, 25, 150, 100)
            self.square[i].syllabeSignal.connect(syllabeSignal)
            # self.square[i].setBackground(black)
            self.square[i].setStyleSheet("QWidget { background-color: rgb(1, 1,1); }") 
            # new = QtGui.QPushButton(self.square[i])
            # self.addLayout(self.square[i]
            Pallette = self.square[i].palette();
            Pallette.setColor(QtGui.QPalette.Window, QtCore.Qt.red);
            Pallette.setColor(QtGui.QPalette.Base, QtCore.Qt.blue);
            Pallette.setColor(QtGui.QPalette.Button, QtCore.Qt.green);
            self.square[i].setPalette(Pallette);
            self.square[i].setAutoFillBackground(True);
            self.square[i].hide()

        # Group.buttonClicked.connect(self.ButtonGroupChecked) 

        self.setGeometry(400, 300, 720, 150)
        self.setMinimumSize(350, 160)
        self.setWindowTitle('Syllable selection')

    # def ButtonGroupChecked(self,Id):
    #     print Id

