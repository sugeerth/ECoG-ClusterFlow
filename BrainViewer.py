import csv

import math
from collections import defaultdict
import time
# from PySide import QtCore, QtGui
from sys import platform as _platform
import weakref
import cProfile
from PySide import QtWebKit
import sys
import pprint
from PySide import QtCore, QtGui , QtUiTools

import warnings
warnings.filterwarnings("ignore")

import matplotlib
# matplotlib.use('Agg')
import numpy
from optparse import OptionParser
import os
import sys

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

# import time'
import warnings
with warnings.catch_warnings(): 
           warnings.simplefilter("ignore", category=RuntimeWarning)

import community as cm
try:
    # ... reading NIfTI 
    import nibabel as nib
    import numpy as np
    import pyqtgraph as pgp
    # import h5py
    # ... graph drawing
    import networkx as nx

except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise

app = QtGui.QApplication(sys.argv)

### BrainViewer packages
from GraphView.correlation_table import CorrelationTable, CorrelationTableDisplay, CommunityCorrelationTableDisplay
from UIFiles.ProcessUi import ProcessQuantTable
from QuantTable.quantTable import quantTable
from QuantTable.quantData import QuantData
from Data.color_table import CreateColorTable
from CommunityAnalysis.CommunitiesAcrossTimeStep import CommunitiesAcrossTimeStep
from CommunityAnalysis.communityDetectionEngine import ColorBasedOnBackground
from ConsensusClster.cluster import ConsensusCluster
from GraphView.Graph_interface import GraphWidget
from GraphView.GraphDataStructure import GraphVisualization
from LayoutDesign.Layout_interface import LayoutInit
from Electrode.Electrode import ImageLabel
from CommunityAnalysis.syllableChoice import Syllable
from Data.dataProcessing import dataProcessing
from interface.SignalInterface import Interface
from interface.ElectrodeInterface import ElectrodeInterface
from interface.CommunitiesAcrossTimestepInterrface import CommunitiesAcrossTimeStepInterface
from WebViewServer.CustomWebView import CustomWebView
from LayoutDesign.SmallMulitplesLayoutDesign import SmallMultipleLayout
from PathFiles import *

"""
1) Gray color distinguishableColors[0] == gray 

Visaully perceptive dark colors for opacity thresholding

"""
# distinguishableColors = [(255,0,0),(31,120,180),(252,205,229),(247,129,191),\
# (186,186,186),(135,135,135),(0,0,0),(251,0,0), (255,0,0),\
#  (0,0,0), (251,154,153), (227,26,28),(145,207,96), (0,0,0), (216,218,235), (146,197,222),\
#  (128,128,128), (148,255,181), (143, 124, 0), (157, 204, 0),\
#  (216,191,216),(194, 0, 136), (0, 51, 128),\
#  (255, 168, 187), (66, 102, 0),(255, 0, 16), (94, 241, 242), (0, 153, 143), (224, 255, 102),\
# (116, 10, 255), (153, 0, 0), (255, 192, 203), (153,0,0), (242,170,121), (210,217,108),\
# (38,51,43), (115,191,230), (0,0,77), (51,0,0),(140,98,70),(226,230,172),\
# (0,153,82), (0,136,255), (22,0,166), (115,57,57), (230,195,172), \
# (122,153,0), (61,242,182), (0,48,89), (123,108,217), (140,0,94), (229,122,0),\
# (82,102,0), (57,115,96), (38,45,51), (77,35,140), (89,22,67), (204,153,153), (127,68,0),\
# (195,230,57), (182,242,222), (0,97,242), (141,124,166), (51,13,38), (229,31,0), (255,196,128),\
# (122,230,0), (0,153,122),(0,66,166),(116,0,217),(217,54,141),(204,116,102), (128,113,96),\
# (135,153,115), (0,217,202),(96,134,191),(54,0,102),(191,143,169)]

# ['rgb(252,141,89)','rgb(255,255,191)','rgb(145,191,219)']
# ['rgb(27,158,119)','rgb(217,95,2)','rgb(117,112,179)']
# ['rgb(102,194,165)','rgb(252,141,98)','rgb(141,160,203)']
# ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)']
distinguishableColors = [(0,0,0),(147, 196, 125), (162, 196, 201),\
 (228, 0, 0),\
  (252,141,89), (0,0,0),(145,191,219), (252,141,89),\
   (255,255,191),(145,191,219), (143, 124, 0),\
   (157, 204, 0),(194, 0, 136), (0, 51, 128),\
 (255, 168, 187), (94, 241, 242), (0, 153, 143), (224, 255, 102),\
(116, 10, 255), (252,141,89), (255,255,191),(145,191,219), (242,170,121), (210,217,108),\
(38,51,43), (115,191,230), (0,0,77), (51,0,0),(140,98,70),(226,230,172),\
(0,153,82), (0,136,255), (22,0,166), (115,57,57), (230,195,172), \
(122,153,0), (61,242,182), (140,0,94), (229,122,0),]

FontBgColor = [ColorBasedOnBackground(r,g,b) for r,g,b in iter(distinguishableColors)]

LAYOUT=QtGui.QHBoxLayout()
# FIX ME 

colorTableName = 'blue_lightblue_red_yellow'
selectedColor = (0, 100, 0, 255)

CorrelationTableShowFlag = False
MainWindowShowFlag = False
GraphWindowShowFlag = False
ElectrodeWindowShowFlag = True

print "Creating files that will read the paths"
execfile('BrainViewerDataPathsArtificial.py')
print "Creating correlation table display"

if ElectrodeWindowShowFlag:
    print "Processing electrodeFiles"
    dataProcess = dataProcessing(Brain_image_filename,SelectedElectrodes_filename,Electrode_ElectrodeData_filename,Electrode_mat_filename, ElectrodeSignals, Electrode_Ids_filename, Electrode_data_filename)
    correlationTable = CorrelationTable(dataProcess)
else:
    correlationTable = CorrelationTable(matrix_filename,centres_abbreviation)

colorTable = CreateColorTable(colorTableName)
colorTable.setRange(correlationTable.valueRange())

Visualizer.setContentsMargins(0,0,0,0)
ElectrodeLayout = QtGui.QHBoxLayout()
ElectrodeLayout.setContentsMargins(0,0,0,0)

print "Creating main GUI."

Counter = len(correlationTable.data)
DataColor = np.zeros(Counter+1)

# Offset for the table widgets and everything 
if Counter < 50:
        Offset= Counter/2 - Counter/28
else: 
        Offset = 4

# Layout for the tablewidget 
BoxTableWidget =QtGui.QWidget()

# Layout for the graph widget 
BoxGraphWidget =QtGui.QWidget()

# Layout for the electrode
BoxElectrodeWidget = QtGui.QWidget() 

print "Setting GraphDataStructure"
Tab_2_AdjacencyMatrix = GraphVisualization(correlationTable.data)

print "Setting CorrelationTable for communities"
Tab_2_CorrelationTable = CommunityCorrelationTableDisplay(correlationTable, colorTable,Tab_2_AdjacencyMatrix)
Tab_2_CorrelationTable.setMinimumSize(390, 460)

print "Setting CorrelationTable"
Tab_1_CorrelationTable = CorrelationTableDisplay(correlationTable, colorTable,Tab_2_AdjacencyMatrix)
Tab_1_CorrelationTable.setMinimumSize(390, 460)

# Method to add graphs inside this
print "Setting Graph Widget"

""" Controlling graph widgets  """
widget = GraphWidget(Tab_2_AdjacencyMatrix,Tab_2_CorrelationTable,correlationTable,colorTable,selectedColor,BoxGraphWidget,BoxTableWidget,Offset,distinguishableColors,FontBgColor, ui, electrodeUI,dataProcess, Visualizer)

""" Controlling Quant Table """
print "Initializing Quant Table"
quantData=QuantData(widget)
quantTableObject = quantTable(quantData,widget)

print "Setting Graph interface"
Graph_Layout=LayoutInit(widget,quantTableObject,ui,Visualizer,dataSetLoader,screenshot)

"""Window for correlation Table"""
window_CorrelationTable =QtGui.QWidget()
Box = QtGui.QHBoxLayout()
Box.addWidget(Tab_1_CorrelationTable)
Box.setContentsMargins(0, 0, 0, 0)

window_CorrelationTable.setLayout(Box)
window_CorrelationTable.setWindowTitle("CorrelationTable")
window_CorrelationTable.resize(Offset*(Counter)-0,Offset*(Counter)+170)

Tab_2_CorrelationTable.hide()
BoxTable = QtGui.QHBoxLayout()
BoxTable.setContentsMargins(0, 0, 0, 0)
BoxTable.addWidget(window_CorrelationTable)
BoxTable.addWidget(Tab_2_CorrelationTable)
BoxTable.addWidget(widget.wid)
BoxTableWidget.setLayout(BoxTable)

if CorrelationTableShowFlag:
    BoxTableWidget.hide()

print "Setting Graph Layout_interface"

Graph = QtGui.QHBoxLayout()
Graph.setContentsMargins(0, 0, 0, 0)
Graph.addWidget(widget.wid)
BoxGraphWidget.setLayout(Graph)

if GraphWindowShowFlag:
    BoxGraphWidget.hide()

# For image label 
print "Setting up Electrode data"
if ElectrodeWindowShowFlag: 
    Electrode = ImageLabel(dataProcess, correlationTable, colorTable, selectedColor, Counter, widget, electrodeUI, Visualizer)
    communitiesAcrossTimeStep = CommunitiesAcrossTimeStep(widget, Electrode, electrodeUI, AcrossTimestep, Visualizer)
    Electrode.CommunitiesAcrossTimeStep = communitiesAcrossTimeStep 

    communitiesAcrossTimeStep.AcrossTimestepUI = AcrossTimestep
    CommunitiesLayout = CommunitiesAcrossTimeStepInterface(AcrossTimestep, communitiesAcrossTimeStep)

    syllable = Syllable()

    InterfaceSignals= Interface(widget,ui,Electrode,electrodeUI,\
        communitiesAcrossTimeStep,Tab_1_CorrelationTable,\
        Tab_2_CorrelationTable, Visualizer, quantData, quantTableObject, Graph_Layout)

    ElectrodeInterface = ElectrodeInterface(widget,ui,Electrode,electrodeUI,\
        communitiesAcrossTimeStep,Tab_1_CorrelationTable,Tab_2_CorrelationTable, Visualizer)
    
    Electrode.ElectrodeInterface = ElectrodeInterface

    FinalLayout = QtGui.QHBoxLayout()
    # FIX ME appropriate naming of the widgets 
  
    vbox = QtGui.QVBoxLayout()
    plotLayout = QtGui.QHBoxLayout()

    # Faster debugging
    def debug():
        """
        Easy to debug module, streaming data can be simulated 
        here
        """

        widget.SelectNodeColor("communities")
        Visualizer.correlation.setCurrentIndex(1)

        Electrode.ClusterActivated("ConsensusPreComputed")
        Visualizer.NodeSize1.setCurrentIndex(6)

        Electrode.TowValueChanged("5")
        Visualizer.Tow1.setValue(5)

        Electrode.MultipleTimeGlyph(True) 
        Visualizer.Glyphs.setChecked(True)

        # Electrode.SyllableOneClicked()

        # communitiesAcrossTimeStep.flushData()
        # Electrode.playButtonFunc() 
    debug()

"""Window for correlation Table"""
view = CustomWebView()

Electrode.StopAnimationSignal.connect(view.reload)
MainSmallMultipleInterface = SmallMultipleLayout(Electrode, view)
ElectrodeInterface.connectCustomWebView(view, MainSmallMultipleInterface)

view.SmallMultipleLayout = MainSmallMultipleInterface 
Visualizer.SliceInterval.valueChanged[int].connect(MainSmallMultipleInterface.LayoutChangesOnSliceChange)
# Visualizer.SliceInterval.


vbox.setContentsMargins(0, 0, 0, 0)
vbox.addWidget(MainSmallMultipleInterface)
vbox.setContentsMargins(0, 0, 0, 0)
    
# Slider widgets
FinalLayout.addLayout(vbox)
FinalLayout.setContentsMargins(0, 0, 0, 0)

ElectrodeLayout.setContentsMargins(0, 0, 0, 0)
ElectrodeLayout.addWidget(MainSmallMultipleInterface)
ElectrodeLayout.setContentsMargins(0, 0, 0, 0)

Visualizer.setContentsMargins(0,0,0,0)
Visualizer.setLayout(ElectrodeLayout)
Visualizer.setContentsMargins(0,0,0,0)

Visualizer.show()
sys.exit(app.exec_())
