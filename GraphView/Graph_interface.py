### Standard Python packages
#-*- coding: utf-8 -*-
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
from Dendogram.dendogram import dendogram, DendoNode
from CommunityAnalysis.AdditionalMetricsCustomizable import AdditionalMetricsCustomizable
from CommunityAnalysis.communityDetectionEngine import communityDetectionEngine

#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
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
        # start_time = time.time()
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
        self.ClusteringAlgorithm = 0
        self.TowChanged = False
        self.selectedColor = selectedColor
        self.Graph_data = weakref.ref(Graph_data)
        self.Tab_2_CorrelationTable = weakref.ref(Tab_2_CorrelationTable)
        self.ColorNodesBasedOnCorrelation =True
        self.communityObject = None
        self.correlationTable = weakref.ref(correlationTable)
        self.correlationTableObject = self.correlationTable()
        self.partition =[]
        self.AnimationMode = False
        self.MaxDepthLevel = 2
        self.TimeStep = 0
        self.sortedValues = None
        self.AdditionalMetricsCustomizable = AdditionalMetricsCustomizable(self)

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
        # self.Max=correlationTable.data.max()
        self.Min=correlationTable.data.min()
        self.Max = self.Min
        self.Check = False
        self.DataColor = np.zeros(self.counter)
        self.EdgeColor = np.zeros(self.counter * self.counter)
        self.ColorToBeSentToVisit = list() 
        # self.EdgeSliderValue = self.Max - 0.01
        self.EdgeSliderValue = self.Min
        self.nodesize = 7
        self.grayOutNodes = True
        self.PositionPreserve = True
        self.TowValue = -1
        self.Graph_Color(-1)

        # initializing with an arbitrary layout option 
        self.Layout = 'fdp'

        # initializing the scene
        scene = QtGui.QGraphicsScene(self)

        # Black background 
        # bgColor = QtGui.QColor(0, 0, 0)
        scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
        # scene.setBackgroundBrush(bgColor)


        #scene.setSceneRect(0, 0, 8000, 6000);
        self.setScene(scene)

        self.Scene_to_be_updated = scene
        self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.setInteractive(True)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtGui.QGraphicsView.NoAnchor)
        self.scaleView(2.0)

        self.ColorVisit = []
        self.NodeIds = []

        self.wid = QtGui.QWidget()
        self.hbox = QtGui.QVBoxLayout()

        self.First= True
        #selfpos1=self.Graph_data.g.pos
        self.node = None
        self.clut= np.zeros(self.counter)
        self.FontBgColor = np.zeros(self.counter)
        self.HighlightedId = None
        self.EdgeIds = []
        # self.interval = self.Graph_data.interval
        # Setting the posiitons of the node for the different graphs              
        self.scale(0.8, 0.8)
        self.setMinimumSize(400, 600)
        self.setWindowTitle(self.tr("Node Visualization"))
        i = 0

        # for node in  self.Graph_data().g.nodes():
        #     pass
        #     i = i + 1
        #     node_value=Node(self,i,correlationTable)
        #     self.NodeIds.append(node_value)
        #     scene.addItem(node_value)

        k = 0 
        Min1 = np.min(self.Graph_data().data)
        Max1 = np.max(self.Graph_data().data)

        # for i in range(1, self.counter):
        #     for j in range(1, self.counter):
        #         pass
        #         if (i-1 >= j-1): 
        #             continue
        #         try:
        #             t = self.correlationTable().value(i-1,j-1)
        #             self.EdgeColor[k] = ColorToInt(self.colortable.getColor(t))
        #             scene.addItem(Edge(self,self.NodeIds[i-1],self.NodeIds[j-1],k,i,j,self.Max,self.Graph_data().data[i-1][j-1]))
        #         except KeyError:
        #             continue
        #         k = k + 1 

        self.edges = [weakref.ref(item) for item in self.scene().items() if isinstance(item, Edge)]
        self.nodes = [weakref.ref(item) for item in self.scene().items() if isinstance(item, Node)]
     
        # print("Adding the edges and nodes into Qpainter --- %f seconds ---" % (time.time() - start_time1))
     
        # self.setLayout('fdp')
        self.g =  self.Graph_data().DrawHighlightedGraph(self.EdgeSliderValue)

        # print("Init of GraphWidget --- %f seconds ---" % (time.time() - start_time))

        self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
        self.setScene(self.Scene_to_be_updated)
        self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
        # self.scaleView(math.pow(2.0, -500 / 1040.0))

        self.communityDetectionEngine = communityDetectionEngine(self,distinguishableColors,FontBgColor)
        self.communityDetectionEngine.CalculateColors.connect(self.CalculateColors)
        self.communityDetectionEngine.CalculateFormulae.connect(self.CalculateFormulae)

        print "done"
        # self.SelectNodeColor("Something")

    def CalculateColorsFunction(self,state):
        self.CalculateColors.emit(TowValue)

    def changeHighlightedEdges(self,state):
        if state: 
            # edges = [item for item in self.scene().items() if isinstance(item, Edge)]
            for edge in self.edges:
                 edge().setHighlightedColorMap(False)
        else: 
            # edges = [item for item in self.scene().items() if isinstance(item, Edge)]
            for edge in self.edges:
                 edge().setHighlightedColorMap(True)

        self.Scene_to_be_updated.update()

    def LayoutCalculation(self):
        self.setLayout('spring')

    def NewNodeSelected(self,idx):
        self.HighlightedId = idx 

    def changeSpringLayout(self,state):
        if state: 
            self.PositionPreserve = True
        else: 
            self.PositionPreserve = False
        #print self.PositionPreserve

    def changeTitle(self, state):
        self.DisplayOnlyEdges = not(self.DisplayOnlyEdges)
        self.NodeSelected(self.HighlightedId) 
        self.Scene_to_be_updated.update()

    def ColorForVisit(self,partition):
        self.ColorToBeSentToVisit = []
        for key,value in partition.items():
            self.ColorToBeSentToVisit.append(self.ColorVisit[value])
        # Matrix = nx.to_numpy_matrix(self.induced_graph)
        # print("Find_InterModular_Edge_correlativity CorrelationTable --- %f seconds ---" % (time.time() - start_time))

    """
    Defines the initial positions needed for the spring layout algorithm
    """
    def Find_Initial_Positions(self):
        # return Pos
        # start_time = time.time()
        self.communityPos.clear()

        count = 0
        nodes1 = [item for item in self.scene().items() if isinstance(item, Node)]

        for community in set(self.partition.values()):
            Sumx = 0
            Sumy = 0

            list_nodes = [nodes for nodes in self.partition.keys() if self.partition[nodes] == community]
            for node in nodes1:
                if node.counter-1 in list_nodes:
                    Sumx = Sumx + self.pos[node.counter-1][0] 
                    Sumy = Sumy + self.pos[node.counter-1][1]
                    # print Sumx,Sumy,self.pos[node.counter-1][0],node.counter-1
            centroidx = Sumx/len(list_nodes)
            centroidy = Sumy/len(list_nodes)
            # print centroidx,centroidy
            self.communityPos[community] = (centroidx,centroidy)
            count = count + 1

        # print("ChangeColrs CorrelationTable --- %s seconds ---" % (time.time() - start_time))
        return self.communityPos

    def changeGrayOutNodes(self,state):
        self.grayOutNodes = not(self.grayOutNodes)
        if not(self.level == -1):
            self.communityDetectionEngine.ChangeCommunityColor(self.level)
        else: 
            self.communityDetectionEngine.ChangeCommunityColor()

    @Slot(bool)
    def ToggleAnimationMode(self,state):
        self.AnimationMode = False

    @Slot(int)
    def changeStuffDuetoTowChange(self,value):
        # -1 says it is coming for Tow change
        self.TowChanged = True 
        self.TowValue = value
        # print self.TowValue, self.TowChanged, "Tow changes"
        # self.communityDetectionEngine.ChangeGraphDataStructure()
        # self.communityDetectionEngine.ChangeGraphWeights()
        self.ChangeValue2(0)
        # self.Refresh()
        # self.updateElectrodeView()
        # self.scaleView(1.0001)
    
    def ClusterChangeHappening(self,ClusteringAlgorithm):
        # print ClusteringAlgorithm
        self.ClusteringAlgorithm = ClusteringAlgorithm
        self.changeStuffDuetoTowChange(self.TowValue)

    def changeTimeStepSyllable(self,Syllable, TimeStep):
        #Refrshes the entrire graph structure, 1) changes the layout, 2)dendograms, 3) reomputes the colors
        #idetifies communities 4) recomputes everything
        # if TimeStep > Timestep-2:
        #     print "I am in the termination cluster" 
        #     self.AnimationMode = False
        #     return    
        self.AnimationMode = True
        #* Changed index
        self.TimeStep = TimeStep
        self.Syllable = Syllable

        self.correlationTable().changeTableContents(Syllable,TimeStep)
        self.communityDetectionEngine.ChangeGraphDataStructure()
        self.communityDetectionEngine.ChangeGraphWeights()

        # comment for a reason as no real time communties 
        self.ChangeValue2(0)
        # print "Refreshing "
        # refresh the data
        # self.Refresh()
        # self.scaleView(1.0001)
        # self.Scene_to_be_updated.update()
        # refresh the 

    """Sending dendogram values"""
    @Slot(int)
    def changeDendoGramLevel(self,level):
        self.level = level-1
        self.propagateLevelValue.emit(self.level)

        print "LEVEL", self.level
        self.communityDetectionEngine.ChangeCommunityColor(self.level)

    """Select the nodes colors"""
    @Slot(bool)
    def SelectNodeColor(self,state):
        # start_time = time.time()
        nodes1 = [item for item in self.scene().items() if isinstance(item, Node)]
        if state == "Correlation Strength": 
            self.ColorNodesBasedOnCorrelation = True 
            for node in nodes1:
                node.NodeColor()
            self.Tab_2_CorrelationTable().hide()
            self.CommunityMode.emit(False)
            self.wid.hide()
            if self.HighlightedId:
                self.regionSelected.emit(self.HighlightedId)
                # Sending data to the CPP files
            else:
                self.regionSelected.emit(3)
            self.resizeTheWholeGraphWidget(True)
            # self.Scene_to_be_updated.update()
        else: 
            # print "Going in Communtiy Node color"
            self.Tab_2_CorrelationTable().show()
            self.wid.show()
            self.ColorNodesBasedOnCorrelation = False 
            if not(self.level == -1):
                self.communityDetectionEngine.ChangeCommunityColor(self.level)
            else: 
                self.communityDetectionEngine.ChangeCommunityColor()
            self.CommunityMode.emit(True)
            self.CommunityColor.emit(self.ColorToBeSentToVisit)
            self.resizeTheWholeGraphWidget(False) 
        self.Scene_to_be_updated.update()

        del nodes1
        # print("Node Color --- %f seconds ---" % (time.time() - start_time))

    def changeTitleSetColorMap(self, state):
        if state: 
            # edges = [weakref.ref(item) for item in self.scene().items() if isinstance(item, Edge)]
            for edge in self.edges:
                edge().setColorMap(True)
        self.Scene_to_be_updated.update()

    def resizeTheWholeGraphWidget(self,state):

        if state: 
            # Entering the correlation mode initiate the window to resize to normal (Workaround not adrdressing the root problem)
            self.BoxGraphWidget.resize(550,550)
            self.BoxTableWidget.resize(self.width,self.width) 
        else: 
            # Entering the community mode initiate the window to have an earlier version
            newwidth = self.width+self.width
            self.BoxTableWidget.resize(newwidth,self.width)

    def setLayout(self,Layout='sfdp'):
        # start_time = time.time()
        # scale = self.counter*100
        # print "Changing layout" 
        Layout = (Layout.encode('ascii','ignore')).replace(' ','')
        self.g =  self.Graph_data().DrawHighlightedGraph(self.EdgeSliderValue)

        if not(self.ColorNodesBasedOnCorrelation):
            partition=cm.best_partition(self.g)
            size = float(len(set(partition.values())))
            # modularity = cm.modularity(partition, self.g)
            induced_graph = cm.induced_graph(partition,self.g)
            if not(self.level == -1): 
                dendo=cm.generate_dendogram(self.g)
                g = cm.partition_at_level(dendo,self.level)
                partition = g
            print "New Colors GENERATING"
            self.communityDetectionEngine.GenerateNewColors(len(set(partition.values())))
        #communities=list(nx.k_clique_communities(self.g, 3))
        #print communities 
        if (Layout == "circular") or (Layout == "shell") or (Layout == "random") or (Layout == "fruchterman_reingold_layout") or (Layout == "spring") or (Layout == "spectral"):
            if (Layout == "spring"): 
               # print self.Graph_data.G.edge['weight']
                #print self.pos
                # print "setLayout" 
                if self.First:
                    self.First =False
                    self.neewPos=nx.spring_layout(self.g,weight = 'weight', k = 0.55, iterations = 20,scale =500)
                    self.pos=self.neewPos
                else: 
                    self.neewPos=nx.spring_layout(self.g,pos=self.pos,weight = 'weight',scale = 500)
                    self.pos=self.neewPos
                count = 0 
                Factor = 1

                    #self.CommunityColor.emit(self.ColorToBeSentToVisit)
            elif (Layout == "random") or (Layout == "shell") or (Layout == "neato"):
                self.neewPos=eval('nx.'+Layout+'_layout(self.g)')
                self.pos=self.neewPos
                Factor = 700
            else: 
                self.neewPos=eval('nx.'+Layout+'_layout(self.g)')
                self.pos=self.neewPos
                Factor = 500
            if not(self.ColorNodesBasedOnCorrelation): 
                    self.ColorNodesBasedOnCorrelation = False 
                    if not(self.level == -1):
                        self.communityDetectionEngine.ChangeCommunityColor(self.level)
                    else: 
                        self.communityDetectionEngine.ChangeCommunityColor()
        else:
            if Layout != "circo":
                # print "computing the circles" 
                self.pos=nx.graphviz_layout(self.g,prog=Layout,args='-Gsep=.25,-GK=20-Eweight=2')
                Factor = 0.60 + self.counter/100
                if Layout == 'sfdp':
                    Factor = 0.55
            else:
                print "Before Circo" 
                self.pos=nx.graphviz_layout(self.g,prog=Layout)
                print "After Circo" 
                #print self.pos
                Factor = 0.30
            if not(self.ColorNodesBasedOnCorrelation): 
                self.ColorNodesBasedOnCorrelation = False 
                if not(self.level == -1):
                    self.communityDetectionEngine.ChangeCommunityColor(self.level)
                else: 
                    self.communityDetectionEngine.ChangeCommunityColor()
              #self.CommunityColor.emit(self.ColorToBeSentToVisit)

        # Degree Centrality for the the nodes involved
        self.Centrality=nx.degree_centrality(self.g)
        self.Betweeness=nx.betweenness_centrality(self.g)  
        self.LoadCentrality = nx.load_centrality(self.g)
        self.ParticipationCoefficient = self.AdditionalMetricsCustomizable.participation_coefficient(self.g,True)
        self.ClosenessCentrality = nx.closeness_centrality(self.g)
        # self.EigenvectorCentrality = nx.eigenvector_centrality(self.g)
        self.EigenvectorCentralityNumpy= nx.eigenvector_centrality_numpy(self.g)

        for i in range(len(self.ParticipationCoefficient)):
            if (str(float(self.ParticipationCoefficient[i])).lower() == 'nan'):
                   self.ParticipationCoefficient[i] = 0
        i = 0 
        
        """ Calculate rank and Zscore """
        MetrixDataStructure=eval('self.'+self.nodeSizeFactor)
        from collections import OrderedDict
        self.sortedValues = OrderedDict(sorted(MetrixDataStructure.items(), key=lambda x:x[1]))

        self.average = np.average(self.sortedValues.values())
        self.std = np.std(self.sortedValues.values())

        for item in self.scene().items():
            if isinstance(item, Node):
                x,y=self.pos[i]
                item.setPos(QtCore.QPointF(x,y)*Factor)
                Size = eval('self.'+self.nodeSizeFactor+'[i]')
                rank, Zscore = self.calculateRankAndZscore(i)
                item.setNodeSize(Size,self.nodeSizeFactor,rank,Zscore)
                i = i + 1

        # print "Will calculateforces now" 
        for edge in self.edges:
            edge().adjust()

        # self.Refresh()

        if not(self.PositionPreserve):
            self.Scene_to_be_updated.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
            self.setScene(self.Scene_to_be_updated)


        self.changeViewinGraph()

        # self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
        # self.setScene(self.Scene_to_be_updated)
        # self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
        # # self.scaleView(math.pow(2.0, -500 / 1040.0))
        # # self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
        # self.Scene_to_be_updated.update()
        # self.Refresh()
        # print("setLayout --- %s seconds ---" % (time.time() - start_time))

    def UpdateThresholdDegree(self):
        self.g =  self.Graph_data().DrawHighlightedGraph(self.EdgeSliderValue)

        
        # # Degree Centrality for the the nodes involved
        # self.Centrality=nx.degree_centrality(self.g)
        # self.Betweeness=nx.betweenness_centrality(self.g)  
        # self.ParticipationCoefficient = self.AdditionalMetricsCustomizable.participation_coefficient(self.g,True)
        # self.LoadCentrality = nx.load_centrality(self.g)
        # self.ClosenessCentrality = nx.closeness_centrality(self.g)
        # self.EigenvectorCentralityNumpy= nx.eigenvector_centrality_numpy(self.g)

        # for i in range(len(self.ParticipationCoefficient)):
        #     if (str(float(self.ParticipationCoefficient[i])).lower() == 'nan'):
        #            self.ParticipationCoefficient[i] = 0

        # i = 0
        # """ Calculate rank and Zscore """
        # MetrixDataStructure=eval('self.'+self.nodeSizeFactor)

        # from collections import OrderedDict
        # self.sortedValues = OrderedDict(sorted(MetrixDataStructure.items(), key=lambda x:x[1]))
        
        # self.average = np.average(self.sortedValues.values())
        # self.std = np.std(self.sortedValues.values())
        
        # for item in self.scene().items():
        #     if isinstance(item, Node):
        #         Size = eval('self.'+self.nodeSizeFactor+'[i]')
        #         rank, Zscore = self.calculateRankAndZscore(i)
        #         item.setNodeSize(Size,self.nodeSizeFactor,rank,Zscore)    
        #         i = i + 1

        # self.ThresholdChange.emit(True)
        # if not(self.ColorNodesBasedOnCorrelation): 
        #     self.DendoGramDepth.emit(self.MaxDepthLevel)
        
        # self.Refresh()

    def calculateRankAndZscore(self,counter):

        """Zscore and Rank"""
        Rank =  abs(self.sortedValues.keys().index(counter)-(self.counter-1))
        Zscore = (self.sortedValues[counter] - self.average)/(self.std)

        return (Rank,Zscore)

    def setNodeSizeOption(self,state):
        self.nodeSizeFactor = state
        self.UpdateThresholdDegree()

    def SelectLayout(self, Layout):
        self.setLayout(Layout)
        self.Layout = Layout

    def changeViewinGraph(self):
        self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
        self.setScene(self.Scene_to_be_updated)
        self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
        # self.scaleView(math.pow(2.0, -500 / 1040.0))
        # self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
        self.Scene_to_be_updated.update()
        # self.Refresh()
        self.update()
    # def NormalizeWeights(self,Weights):
    #     # (((self.Graph_data().data[i-1][j-1]-Min1)/(Max1 - Min1))*10) 

    def NodeSelected(self,NodeId):
        # start_time = time.time()
        if isinstance(self.sender(),Node) or isinstance(self.sender(),GraphWidget):
            return

        if not(isinstance(self.sender(),GraphWidget)):
            for node in self.nodes:
                if node().counter-1 == NodeId:
                    node().SelectedNode(NodeId,True)
                    node().setSelected(True)
                    node().update()

        if self.DisplayOnlyEdges:
            for edge in self.edges:
                edge().ColorOnlySelectedNode(True)
        else:
           for edge in self.edges:
                edge().ColorOnlySelectedNode(False) 
        # print("Graph interface  --- %f seconds ---" % (time.time() - start_time1))
        self.HighlightedId = NodeId
        self.Scene_to_be_updated.update()
        # print("Node Selected --- %f seconds ---" % (time.time() - start_time))

    def Refresh(self):
        for edge in self.edges:
            edge().update()

        for node in self.nodes:
            node().update()

        # Using community signal as general Update procedure 
        self.CommunityMode.emit(True)

        self.Scene_to_be_updated.update()

    def communityGraphUpdate(self):
        for edge in self.communityObject.edges:
            edge.update()

        for node in self.communityObject.nodes:
            node.update()

        self.communityObject.Scene_to_be_updated.update()

    def NodeSlider(self):
        self.slider2 = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.slider2.setRange(0, 10)
        self.slider2.setValue(0)
        self.slider2.setToolTip("Node Size: %0.2f" % (self.nodesize))
        self.slider2.setTracking(False)
        self.slider2.valueChanged[int].connect(self.ChangeValue)
        self.slider2.hide()

    def ChangeValue(self,value):
        self.nodesize =  value
        self.slider2.setToolTip("Node Size: %0.2f" % (self.nodesize))
        nodes = [item for item in self.scene().items() if isinstance(item, Node)]
        for node in nodes:
            if node.counter == (self.HighlightedId+1):
                continue
            node.ChangeNodeSize(value)

    # @Slot(bool)
    # def hoverChanged(self,state):
    #     self.hoverRender = not(self.hoverRender)
    #     # Enable/Disable hover events for the entire tool
    #     nodes = [item for item in self.scene().items() if isinstance(item, Node)]
    #     for node in nodes:
    #         node.setAcceptHoverEvents(self.hoverRender)
    #         node.update()

    #     if self.communityObject:
    #         nodes = [item for item in self.communityObject.scene().items() if isinstance(item, Node)]
    #         for node in nodes:
    #             node.setAcceptHoverEvents(self.hoverRender)
    #             node.update()

    #         edges = [item for item in self.communityObject.scene().items() if isinstance(item, Edge)]
    #         for edge in edges:
    #             edge.setAcceptHoverEvents(self.hoverRender)
    #             edge.update()

    #         DendoNodes = [item for item in self.dendogramObject.scene.items() if isinstance(item, DendoNode)]
    #         AllEdges = [item for item in self.dendogramObject.scene.items() if isinstance(item, Edge)]
            
    #         for node in DendoNodes:
    #             node.setAcceptHoverEvents(self.hoverRender)
    #             node.update()

    #         for edge in AllEdges:
    #             edge.setAcceptHoverEvents(self.hoverRender)
    #             edge.update()

    @Slot(bool)
    def changeTransparency(self,state):
        pass
        # self.setTransp = not(self.setTransp)
        # nodes = [item for item in self.scene().items() if isinstance(item, Node)]
        # for node in nodes:
        #     node.setTransp = self.setTransp
        #     node.unsetOpaqueNodes()
        #     # not Highly efficient
        #     # if node.isSelected():
        #     #     node.SelectedNode(node.counter-1,False, 1)
        #     node.update()   
        # self.Refresh()

    def ChangeValue2(self,value):
        """Changing the value of the communities"""

        value_for_slider = float(value) / 1000 
        self.EdgeSliderValue = value_for_slider
        self.slider1.setValue = self.EdgeSliderValue
        # self.Lineditor.setText(str(self.EdgeSliderValue))
        # self.slider1.setToolTip("Edge Weight: %0.2f" % (self.EdgeSliderValue))
        # print "In between things"
        if not(self.PositionPreserve):
            pass
            # self.setLayout(self.Layout)
        for edge in self.edges:
            #FIX ME CHANGED EDGETHRESHOLD VALUE
            pass
            # edge().Threshold(0)


        if not(self.ColorNodesBasedOnCorrelation):
            """Community Mode"""
            if not(self.PositionPreserve):
                """position is not preserved"""
                """recalculate stuff in position preserve"""
                pass
            # else:
            # prin self.level
            if not(self.level == -1):
                self.communityDetectionEngine.ChangeCommunityColor(self.level)
            else:
                self.communityDetectionEngine.ChangeCommunityColor(-1)

            # self.CommunityColor.emit(self.ColorToBeSentToVisit)
        if not(self.AnimationMode): 
            self.UpdateThresholdDegree()
            self.Scene_to_be_updated.update()

    def ComputeUpdatedClusters(self,cluster):
        print cluster

    @Slot()
    def LineEditChanged(self):
        """Handling Line Edit changes"""
        text = (self.Lineditor.text().encode('ascii','ignore')).replace(' ','')
        value = float(text)*1000
        self.EdgeWeight.emit(int(value))

    def slider_imple(self):
        """implementation of Edge threshold sliders"""
        self.slider1 = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.slider1.setTracking(False)
        self.slider1.setRange( self.Graph_data().data.min()*1000 ,  self.Graph_data().data.max()* 1000 )
        self.slider1.setValue(self.EdgeSliderValue*1000)
        self.slider1.setToolTip("Edge Weight: %0.2f" % (self.EdgeSliderValue))
        self.interval=((0.1-5)/10)*(-1)
        self.slider1.valueChanged[int].connect(self.ChangeValue2)
        self.slider1.hide()

    def changeEdgeThickness(self,value):
        NormValue = float(value)
        edgeThickness = 1 + float(NormValue)
        edges = [item for item in self.scene().items() if isinstance(item, Edge)]
        for edge in edges:
            edge.setEdgeThickness(edgeThickness)
            # print edge.edgeThickness

    def lineEdit(self):
        """Drawing the Line editor for the purpose of manualling entering the edge threshold"""
        self.Lineditor = QtGui.QLineEdit()
        self.Lineditor.setText(str(self.EdgeSliderValue))
        self.Lineditor.returnPressed.connect(self.LineEditChanged)
        self.EdgeWeight.connect(self.slider1.setValue)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Up:
            # pprint.pprint(self.HighlightedId)
            self.NodeIds[self.HighlightedId].moveBy(0, -20)
        elif key == QtCore.Qt.Key_Down:
            self.NodeIds[self.HighlightedId].moveBy(0, 20)
        elif key == QtCore.Qt.Key_Left:
            self.NodeIds[self.HighlightedId].moveBy(-20, 0)
        elif key == QtCore.Qt.Key_Right:
            self.NodeIds[self.HighlightedId].moveBy(20, 0)
        elif key == QtCore.Qt.Key_Plus:
            self.nodesize = self.nodesize + 1
            self.ChangeValue(self.nodesize)
        elif key == QtCore.Qt.Key_Minus:
            self.nodesize = self.nodesize - 2
            self.ChangeValue(self.nodesize)
        elif key == QtCore.Qt.Key_D:
            self.DisplayOnlyEdges = not(self.DisplayOnlyEdges)
            self.NodeSelected(self.HighlightedId) 
            self.Scene_to_be_updated.update()
        elif key == QtCore.Qt.Key_M:
            self.ModularityBehaviour()
        elif key == QtCore.Qt.Key_I:
            pprint.pprint(self.Scene_to_be_updated.itemsBoundingRect())
        elif key == QtCore.Qt.Key_Space or key == QtCore.Qt.Key_Enter:
            for item in self.scene().items():
                if isinstance(item, Node):
                    item.setPos(-150 + QtCore.qrand() % 300, -150 + QtCore.qrand() % 300)
            self.Scene_to_be_updated.update()
        else:
            QtGui.QGraphicsView.keyPressEvent(self, event)
        self.Refresh()
        
    def wheelEvent(self, event):
        self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

    def scaleView(self, scaleFactor):
        factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 5, 5)).width()
        if factor < 0.07 or factor > 100:
            return
        self.scale(scaleFactor, scaleFactor)
        del factor

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
