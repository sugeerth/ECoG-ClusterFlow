import csv
import sys
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from sys import platform as _platform
import weakref
import cProfile
import os
TotalNo = 62

class SmallMultipleLayout(QtGui.QWidget):
    def __init__(self,Electrode, CustomWebView):
        super(SmallMultipleLayout,self).__init__()
        self.Electrode = Electrode

        self.setMinimumSize(1425,851)
        self.setMaximumSize(1425,851)

        self.PutAllLayout = True

        self.CustomWebView = CustomWebView 

        self.LayoutDesign()
        self.SignalForWebView()
        self.setContentsMargins(0,0,0,0)

    def LayoutDesign(self): 
        self.widget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout(self.widget)
        
        self.layout.setSpacing(1)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)

        self.scroll = QtGui.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setWidget(self.widget)

        self.LayoutChangesOnSliceChange(self.Electrode.slices)
        self.layout.addWidget(self.CustomWebView, 1, 0,1,15, QtCore.Qt.AlignLeft)

        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        
        self.grid1 = QtGui.QGridLayout()
        self.grid1.addWidget(self.scroll,3,0)

        self.setLayout(self.grid1)

    """
    Slots coming in from other visualization 
    views
    """
    def sliceLayoutChanged(self, slices):
        print "Slices", slices

    def LayoutChangesOnSliceChange(self, sliceValues):
        k = 0
        
        IterationNo = int(math.floor(TotalNo/sliceValues))
        IterationList = [i for i in range(IterationNo)]
        i = 0 
        j = 0
        for l in range(len(IterationList)):
            if k % 2 == 0: 
                WidgetTemp=self.Electrode.SmallMultipleElectrode[k]
                self.layout.setContentsMargins(0, 0, 0, 0)
                if self.PutAllLayout:
                    self.layout.addWidget(WidgetTemp, 0, i, QtCore.Qt.AlignTop)
                    # WidgetTemp.show()
                else:
                    C = self.layout.itemAtPosition(0, i)
                    if C is not None:
                        Widget = C.widget()
                        self.layout.removeWidget(Widget)
                        Widget.hide()
                self.layout.setContentsMargins(0, 0, 0, 0)
                i=i+1
            else: 
                WidgetTemp=self.Electrode.SmallMultipleElectrode[k]
                self.layout.setContentsMargins(0, 0, 0, 0)
                if self.PutAllLayout:
                    self.layout.addWidget(WidgetTemp, 2, j, QtCore.Qt.AlignBottom)
                    # WidgetTemp.show()
                else: 
                    C = self.layout.itemAtPosition(2, j)
                    if C is not None:
                        Widget = C.widget()
                        self.layout.removeWidget(Widget)
                        Widget.hide()
                j = j+1
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.setContentsMargins(0, 0, 0, 0)
            k=k+1 

        A = self.layout.itemAtPosition(0, i)
        B = self.layout.itemAtPosition(2, j)

        while (A is not None):   
            Widget = A.widget()
            self.layout.removeWidget(Widget)
            Widget.hide()
            i = i + 1 
            A = self.layout.itemAtPosition(0, i)

        while (B is not None): 
            Widget = B.widget()
            self.layout.removeWidget(Widget)
            Widget.hide()
            j = j + 1
            B = self.layout.itemAtPosition(2, j)

        horizontalScrollBar = self.scroll.horizontalScrollBar()
        self.lengthPixels = horizontalScrollBar.maximum() + horizontalScrollBar.minimum() + horizontalScrollBar.pageStep()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_L:
            self.PutAllLayout= not(self.PutAllLayout)
            self.LayoutChangesOnSliceChange(self.Electrode.slices)

    def SignalForWebViewE(self, Start, End, Slice):
        self.Start = Start
        self.End = End
        
        StartSlice = Start/self.Electrode.slices
        EndSlice = End/self.Electrode.slices
        gettingSmallMultipleNumber = (End-Start)/Slice
        self.LayoutChangesOnSliceChange(self.Electrode.slices)

        if not(self.PutAllLayout):
            print StartSlice,EndSlice,gettingSmallMultipleNumber
            if EndSlice - StartSlice > 0: 
                q = StartSlice 
                IterationNo = EndSlice-StartSlice
                if IterationNo ==0: 
                    IterationNo = 1
                for i in range(IterationNo):
                    WidgetTemp=self.Electrode.SmallMultipleElectrode[q]
                    WidgetTemp.show()
                    WidgetTemp.setMinimumSize(QtCore.QSize(275,275))
                    WidgetTemp.setMaximumSize(QtCore.QSize(275,275))
                    if i < 5: 
                        self.layout.addWidget(WidgetTemp, 0, i ,QtCore.Qt.AlignCenter)
                    else:
                        self.layout.addWidget(WidgetTemp, 2, (i-5) ,QtCore.Qt.AlignCenter)
                    q=q+1
                C = self.layout.itemAtPosition(0, (EndSlice-StartSlice)+1)

                while (C is not None):   
                    Widget = C.widget()
                    self.layout.removeWidget(Widget)
                    Widget.hide()
                    i = i + 1 
                    C = self.layout.itemAtPosition(0, i)

                A = QtGui.QWidget()
                A.setMinimumSize(QtCore.QSize(275,275))
                A.setMaximumSize(QtCore.QSize(275,275))
                for i in range(5):
                    self.layout.addWidget(A, 2, i)
            else: 
                q = StartSlice 
                WidgetTemp=self.Electrode.SmallMultipleElectrode[q]
                self.layout.addWidget(WidgetTemp, 0, i ,QtCore.Qt.AlignCenter)
                A = QtGui.QWidget()
                A.setMinimumSize(QtCore.QSize(275,275))
                A.setMaximumSize(QtCore.QSize(275,275))
                for i in range(5):
                    self.layout.addWidget(A, 2, i)

    def SignalForWebView(self):
        self.CustomWebView.sendTimeStepFromSankey.connect(self.SignalForWebViewE)
            # self.Electrode.SmallMultipleElectrode[i].EmitSelectedElectrodeView.connect(self.CustomWebView.EmitTimestepRanges)

