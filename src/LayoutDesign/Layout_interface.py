import csv
import sys
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from sys import platform as _platform
import weakref
import cProfile
import os

class LayoutInit(QtGui.QWidget):
    def __init__(self,widget,quantTable,Ui,VisualizerUI,dataSetLoader,screenshot, Brain_image_filename=None,Electrode_mat_filename=None):
        super(LayoutInit,self).__init__()

        self.Ui = Ui
        self.VisualizerUI = VisualizerUI

        self.classVariable(widget, Ui, dataSetLoader,screenshot, VisualizerUI)
        self.widgetChanges()
        self.dialogueConnect()

        Node_Label= QtGui.QLabel('Edge Weight Threshold')
        
        VisualizerUI.highlightEdges.stateChanged.connect(widget.changeHighlightedEdges)
        VisualizerUI.colorEdges.stateChanged.connect(widget.changeTitle)
        VisualizerUI.preservePositions.stateChanged.connect(widget.changeSpringLayout)
        VisualizerUI.springLayout.clicked.connect(widget.LayoutCalculation)
        VisualizerUI.Layout.activated[str].connect(widget.SelectLayout)
        VisualizerUI.correlation.activated[str].connect(widget.SelectNodeColor)
        # VisualizerUI.hover.stateChanged.connect(widget.hoverChanged)
        VisualizerUI.transparent.stateChanged.connect(widget.changeTransparency)
        VisualizerUI.NodeSize.activated[str].connect(widget.setNodeSizeOption)
        
        # VisualizerUI.snapshot.clicked.connect(widget.captureSnapshot)
        # VisualizerUI.Thickness.valueChanged[int].connect(widget.changeEdgeThickness)
        # VisualizerUI.communityLevel.valueChanged[int].connect(widget.changeDendoGramLevel)
        # VisualizerUI.communityLevel.setToolTip("Level: %0.2f" % (widget.level))

        VisualizerUI.dataSet.clicked.connect(self.openFileDialog)
        VisualizerUI.getSnapshots.clicked.connect(self.getSnapshots)
        VisualizerUI.snapshot.clicked.connect(self.captureSnapshot)
        VisualizerUI.quantTable.addWidget(quantTable)
        VisualizerUI.AreaGlyph.hide()
        VisualizerUI.BarGlyph.hide()
        VisualizerUI.three1.hide()
        VisualizerUI.four1.hide()
        VisualizerUI.five1.hide()
        VisualizerUI.six1.hide()

        # VisualizerUI.Save1.hide()
        VisualizerUI.label_28.hide()
        VisualizerUI.Max1.hide()
        VisualizerUI.MaxButtonCheck.hide()
        VisualizerUI.timeInterval1.hide()
        VisualizerUI.PreCompute1.hide()
        VisualizerUI.ElectrodeImages.hide()
        VisualizerUI.Glyphs.hide()
        VisualizerUI.ElecNodes1.hide()
        VisualizerUI.ElectrodeSize.hide()
        VisualizerUI.nodeSize1.hide()

        VisualizerUI.TrElectrode.hide()
        VisualizerUI.TrCommunity.hide()

        self.horizontalLayout = QtGui.QGridLayout()
        self.horizontalLayout.setSpacing(1)
        self.horizontalLayout.addWidget(Node_Label,0,0)
        self.horizontalLayout.addWidget(widget.slider1,0,1)
        self.horizontalLayout.addWidget(widget.Lineditor,0,101,0,102)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        # widget.SelectNodeColor("something")
        # ELECTRODE
        Node_Label.hide()
        widget.slider1.hide()
        widget.Lineditor.hide()

        hbox = QtGui.QVBoxLayout()
        hbox.addWidget(widget)
        # hbox.addLayout(self.horizontalLayout)
        hbox.setContentsMargins(0, 0, 0, 0)

        bbbox = QtGui.QHBoxLayout()
        bbbox.setContentsMargins(0, 0, 0, 0)

        # Ui.setMinimumSize(211,726)
        # bbbox.addWidget(VisualizerUI.Graph.setLayout)
        bbbox.setContentsMargins(0, 0, 0, 0)

        bbbox.addLayout(hbox)
        hbox.setContentsMargins(0, 0, 0, 0)
        bbbox.setContentsMargins(0, 0, 0, 0)
        # Toggling Graph Layouts
        # VisualizerUI.Graph.setLayout(bbbox)
        # self.setLayout(bbbox)

    def classVariable(self,widget,Ui,dataSetLoader,screenshot,VisualizerUI):
        self.widget = widget
        self.Ui = Ui
        self.VisualizerUI = VisualizerUI 
        self.dataSetLoader = dataSetLoader
        self.screenshot= screenshot

    def widgetChanges(self):
        self.widget.setMinimumSize(550, 550)
        self.widget.slider_imple()
        self.widget.NodeSlider()
        self.widget.lineEdit()

    @staticmethod
    def captureSnapshot():
        """ Logic to capture all the parameters for the data visualization """
        
        # Code graciously taken from the Qt website 
        msgBox = QtGui.QMessageBox()
        msgBox.setText("Capturing the snapshot.")
        msgBox.setInformativeText("Do you want to capture the parameters of the visualization?")
        msgBox.setStandardButtons(QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtGui.QMessageBox.Save)
        ret = msgBox.exec_()

        if ret == QtGui.QMessageBox.Save:
            pass

    def getSnapshots(self):
        """ Logic to retrieve the snapshots from the output file """
        self.screenshot.show()

    """ Dataset specific functions """
    def openFileDialog(self):
        """
        Opens a file dialog and sets the label to the chosen path
        """
        self.dataSetLoader.show()
        self.setPathForData()

    def clickLineEdit(self,Flag):
        path, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", os.getcwd())
        if Flag == "centres_abbreviation": 
            self.centres_abbreviation = path
        else: 
            exec("%s='%s'" % (('self.'+Flag+'_filename'), path))
        self.setPathForData()

    def dialogueConnect(self):
        self.dataSetLoader.matrix.clicked.connect(lambda: self.clickLineEdit("matrix"))
        self.dataSetLoader.center.clicked.connect(lambda: self.clickLineEdit("centre"))
        self.dataSetLoader.abbrev.clicked.connect(lambda: self.clickLineEdit("centres_abbreviation"))
        self.dataSetLoader.parcel.clicked.connect(lambda: self.clickLineEdit("parcelation"))
        self.dataSetLoader.template1.clicked.connect(lambda: self.clickLineEdit("template"))

    def LevelValueInSlider(self,level):
        self.Ui.communityLevel.setToolTip("Level: %0.2f" % (self.widget.level))

    def SliderValueChanged(self,MaxDendoGramDepth):
        self.Ui.communityLevel.setMaximum(MaxDendoGramDepth+1)

    def setPathForData(self):
        self.dataSetLoader.elecPath.setText(self.Electrode_data_filename)
        self.dataSetLoader.brainPath.setText(self.Brain_image_filename)
        self.dataSetLoader.posPath.setText(self.Electrode_mat_filename)
        """Logic to send the new files to the start of the applicaiton """