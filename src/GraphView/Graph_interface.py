import csv
import colorsys
import math
import colorsys
import time
from collections import defaultdict
from PySide import QtCore, QtGui
from PySide.QtCore import *
from sys import platform as _platform
import weakref
import cProfile
import pprint
# import time
import community as cm
try:
    # ... reading NIfTI 
    import nibabel as nib
    import numpy as np
    # ... graph drawing
    import networkx as nx

except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise
from GraphDataStructure import GraphVisualization
from CommunityAnalysis.communityDetectionEngine import communityDetectionEngine

from Edge import Edge 
from  Node import Node

Timestep = 12
def ColorToInt(color):
    r, g, b, a = map(np.uint32, color)
    return a << 24 | r << 16 | g << 8 | b

class GraphWidget(QtGui.QGraphicsView):
    
    regionSelected = QtCore.Signal(int)
    EdgeWeight = QtCore.Signal(int)
    CommunityColor = QtCore.Signal(list)
    CalculateColors1 = QtCore.Signal(int)
    CommunityColorAndDict = QtCore.Signal(list,dict)
    CommunityMode = QtCore.Signal(bool)
    ThresholdChange = QtCore.Signal(bool)
    DendoGramDepth = QtCore.Signal(int)
    CalculateFormulae = QtCore.Signal(bool)
    WidgetUpdate = QtCore.Signal(bool)
    propagateLevelValue = QtCore.Signal(int)

    def __init__(self,Graph_data,Tab_2_CorrelationTable,correlationTable,\
        colortable,selectedColor,BoxGraphWidget,BoxTableWidget,Offset,\
        distinguishableColors,FontBgColor ,Ui , electrodeUI, dataProcess, VisualizerUI):
        QtGui.QGraphicsView.__init__(self)
        global Timestep 
        Timestep = dataProcess.timestep
        self.OverallTimestep = Timestep

        self.communityMultiple = defaultdict(list)
        self.colortable=colortable
        self.CalculateColors = self.CalculateColors1
        self.CalculateFormulae = self.CalculateFormulae
        self.BoxGraphWidget = BoxGraphWidget
        self.electrodeUI = electrodeUI
        self.BoxTableWidget = BoxTableWidget
        self.distinguishableColors = distinguishableColors
        self.Ui = Ui
        self.VisualizerUI = VisualizerUI
        self.DisplayOnlyEdges = False
        self.level = -1
        
        self.selectedColor = selectedColor
        self.Graph_data = weakref.ref(Graph_data)
        self.Tab_2_CorrelationTable = weakref.ref(Tab_2_CorrelationTable)
        self.ColorNodesBasedOnCorrelation =True
        self.communityObject = None
        self.correlationTable = weakref.ref(correlationTable)
        self.correlationTableObject = self.correlationTable()
        self.partition =[]

        self.MaxDepthLevel = 2
        self.TimeStep = 0
        self.sortedValues = None
        self.setTransp = True
        self.Syllable = 0
        self.communityPos = dict()
        self.Matrix = []
        self.oneLengthCommunities=[]
        self.hoverRender = True
        self.Centrality = []
        self.Betweeness = []
        self.dendogramObject = None
        self.dendogram = dict()
        self.LoadCentrality = [] 
        self.EigenvectorCentralityNumpy = []
        self.ClosenessCentrality = []
        self.EigenvectorCentrality = []
        self.nodeSizeFactor = "Centrality"
        self.counter = len(correlationTable.data)+1
        self.width =  Offset*(self.counter-1)+45
        self.Min=correlationTable.data.min()
        self.Max = self.Min
        self.Check = False
        self.DataColor = np.zeros(self.counter)
        self.EdgeColor = np.zeros(self.counter * self.counter)
        self.ColorToBeSentToVisit = list() 
        self.EdgeSliderValue = 0.0
        self.nodesize = 7
        self.grayOutNodes = True
        self.PositionPreserve = True
        self.Graph_Color(-1)

        # initializing with an arbitrary layout option 
        self.Layout = 'fdp'

        # initializing the scene
        scene = QtGui.QGraphicsScene(self)
        scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
        self.setScene(scene)

        self.Scene_to_be_updated = scene
        self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.setInteractive(True)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtGui.QGraphicsView.NoAnchor)

        self.NodeIds = []

        self.wid = QtGui.QWidget()
        self.hbox = QtGui.QVBoxLayout()

        self.First= True
        self.node = None

        self.HighlightedId = None
        self.EdgeIds = []
        self.scale(0.8, 0.8)
        self.setMinimumSize(400, 600)
        self.setWindowTitle(self.tr("Node Visualization"))

        i = 0

        for node in  self.Graph_data().g.nodes():
            i = i + 1
            node_value=Node(self,i,correlationTable)
            self.NodeIds.append(node_value)
            scene.addItem(node_value)

        k = 0 
        Min1 = np.min(self.Graph_data().data)
        Max1 = np.max(self.Graph_data().data)

        # WARNING minimum value changed to 0
        self.Min1 = 0.0

        for i in range(1, self.counter):
            for j in range(1, self.counter):
                if (i-1 >= j-1): 
                    continue
                try:
                    t = self.correlationTable().value(i-1,j-1)
                    self.EdgeColor[k] = ColorToInt(self.colortable.getColor(t))
                    scene.addItem(Edge(self,self.NodeIds[i-1],self.NodeIds[j-1],k,i,j,self.Max,self.Graph_data().data[i-1][j-1]))
                except KeyError:
                    continue
                k = k + 1 

        self.edges = [weakref.ref(item) for item in self.scene().items() if isinstance(item, Edge)]
        self.nodes = [weakref.ref(item) for item in self.scene().items() if isinstance(item, Node)]
     
    def CalculateColorsFunction(self,state):
        self.CalculateColors.emit(self.communityDetectionEngine.TowValue)

    @Slot(int)
    def ComputeUpdatedClusters(self,cluster):
        print cluster

    def NewNodeSelected(self,idx):
        self.HighlightedId = idx 

    def ColorForVisit(self,partition):
        self.ColorToBeSentToVisit = []
        for key,value in partition.items():
            self.ColorToBeSentToVisit.append(self.communityDetectionEngine.ColorVisit[value])

    """Select the nodes colors"""
    @Slot(bool)
    def SelectNodeColor(self,state):
        nodes1 = [item for item in self.scene().items() if isinstance(item, Node)]
        if state == "Correlation Strength": 
            self.ColorNodesBasedOnCorrelation = True 
            for node in nodes1:
                node.NodeColor()
            self.Tab_2_CorrelationTable().hide()
            self.CommunityMode.emit(False)
        else: 
            self.ColorNodesBasedOnCorrelation = False 
            if not(self.level == -1):
                self.communityDetectionEngine.ChangeCommunityColor(self.level)
            else: 
                self.communityDetectionEngine.ChangeCommunityColor()
            self.CommunityMode.emit(True)
            self.CommunityColor.emit(self.ColorToBeSentToVisit)
        self.Scene_to_be_updated.update()
        del nodes1

    @Slot(bool)
    def ToggleAnimationMode(self,state):
        self.communityDetectionEngine.AnimationMode = False

    @Slot(int)
    def changeStuffDuetoTowChange(self,value):
        self.communityDetectionEngine.TowChanged = True 
        self.communityDetectionEngine.TowValue = value
        self.DeriveNewCommunities(self.Min1)
    
    def ClusterChangeHappening(self,ClusteringAlgorithm):
        self.communityDetectionEngine.ClusteringAlgorithm = ClusteringAlgorithm
        self.changeStuffDuetoTowChange(self.communityDetectionEngine.TowValue)

    def changeTimeStepSyllable(self,Syllable, TimeStep):
        self.communityDetectionEngine.AnimationMode = True
        self.TimeStep = TimeStep
        self.Syllable = Syllable

        self.correlationTable().changeTableContents(Syllable,TimeStep)
        self.communityDetectionEngine.ChangeGraphDataStructure()
        self.communityDetectionEngine.ChangeGraphWeights()
        self.DeriveNewCommunities(self.Min1)

    def changeTitleSetColorMap(self, state):
        if state: 
            for edge in self.edges:
                edge().setColorMap(True)
        self.Scene_to_be_updated.update()

    def UpdateThresholdDegree(self):
        self.communityDetectionEngine.TimeStepNetworkxGraphData =  self.Graph_data().DrawHighlightedGraph(self.EdgeSliderValue)

    def Refresh(self):
        for edge in self.edges:
            edge().update()

        for node in self.nodes:
            node().update()

        self.CommunityMode.emit(True)
        self.Scene_to_be_updated.update()

    def DeriveNewCommunities(self,value):
        """Changing the value of the communities"""
        self.EdgeSliderValue = value

        if not(self.ColorNodesBasedOnCorrelation):
            """Community Mode"""
            if not(self.PositionPreserve):
                """position is not preserved"""
                """recalculate stuff in position preserve"""
                pass
            if not(self.level == -1):
                self.communityDetectionEngine.ChangeCommunityColor(self.level)
            else:
                self.communityDetectionEngine.ChangeCommunityColor(-1)

        if not(self.communityDetectionEngine.AnimationMode): 
            self.UpdateThresholdDegree()
            self.Scene_to_be_updated.update()

    """
    Manufactures the colors that can be further used for further analysis 
    in other classes of the files
    """
    def Graph_Color(self,Annote, maxval = -1,minval = 1):
        if Annote != 0: 
            Annote = Annote -1
        if maxval == -1: 
            maxval = self.counter

        for i in range(minval,maxval):
            if i != (Annote+1):
                t = self.correlationTable().value(Annote, i-1)
                self.DataColor[i] = ColorToInt(self.colortable.getColor(t))
            else:
                self.DataColor[i] = ColorToInt(self.selectedColor)

    def Edge_Color(self):
        k= 0 
        for i in range(1,self.counter):
            for j in range(1,self.counter):
                if (i-1 >= j-1): 
                    continue
                try:
                    t = self.correlationTable().value(i-1,j-1)
                    self.EdgeColor[k] = ColorToInt(self.colortable.getColor(t))
                    k = k + 1 
                except KeyError:
                    continue
