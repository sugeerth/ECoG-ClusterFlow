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

SliceSizes= 250
Width = 1425
Height = 851

class SmallMultipleLayout(QtGui.QWidget):
    def __init__(self,Electrode, CustomWebView):
        super(SmallMultipleLayout,self).__init__()
        self.Electrode = Electrode

        self.setMinimumSize(Width,Height)
        self.setMaximumSize(Width,Height)

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
        self.layout.addWidget(self.CustomWebView, 1, 0,1,25, QtCore.Qt.AlignLeft)

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
            if EndSlice - StartSlice > -1: 
                q = StartSlice 
                IterationNo = EndSlice-StartSlice
                if IterationNo ==0: 
                    IterationNo = 1
                for i in range(IterationNo):
                    WidgetTemp=self.Electrode.SmallMultipleElectrode[q]
                    WidgetTemp.show()
                    WidgetTemp.setMinimumSize(QtCore.QSize(SliceSizes,SliceSizes))
                    WidgetTemp.setMaximumSize(QtCore.QSize(SliceSizes,SliceSizes))
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
                A.setMinimumSize(QtCore.QSize(SliceSizes,SliceSizes))
                A.setMaximumSize(QtCore.QSize(SliceSizes,SliceSizes))
                for i in range(5):
                    self.layout.addWidget(A, 2, i)


    def SignalForWebView(self):
        self.CustomWebView.sendTimeStepFromSankey.connect(self.SignalForWebViewE)
